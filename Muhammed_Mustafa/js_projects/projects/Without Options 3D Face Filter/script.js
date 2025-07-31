const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];

const config = {
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`,
};

const solutionOptions = {
  selfieMode: true,
  enableFaceGeometry: true,
  maxNumFaces: 3,
  refineLandmarks: false,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
};

const modelConfigs = {
  glasses: {
    path: './models/glasses/scene.gltf',
    scale: [1.5, 1.5, 1.5],
    rotation: [0, -Math.PI / 2, 0],
    position: [0, -2.5, 2],
  },
  mask: {
    path: './models/mask/scene.gltf',
    scale: [18, 18, 18],
    rotation: [0, 0, 0],
    position: [0, -10, 0],
  },
};

class EffectRenderer {
  VIDEO_DEPTH = 500;
  FOV_DEGREES = 63;
  NEAR = 1;
  FAR = 10000;
  scene;
  renderer;
  faceGroups = [];
  camera;
  model = null;
  modelConfig;

  constructor(modelName = 'glasses') {
    const config = modelConfigs[modelName];
    this.modelConfig = config;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xadd8e6);

    this.renderer = new THREE.WebGLRenderer({ canvas: canvasElement });
    this.renderer.outputEncoding = THREE.sRGBEncoding;

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
    hemiLight.position.set(0, 100, 0);
    this.scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff);
    dirLight.position.set(-30, 100, -5);
    dirLight.castShadow = true;
    this.scene.add(dirLight);

    const loader = new THREE.GLTFLoader();
    loader.load(config.path, (gltf) => {
      this.model = gltf.scene;
      const [sx, sy, sz] = config.scale;
      const [rx, ry, rz] = config.rotation;
      this.model.scale.set(sx, sy, sz);
      this.model.rotation.set(rx, ry, rz);
    });
  }

  render(results) {
    this.onCanvasDimsUpdate();
    const imagePlane = this.createGpuBufferPlane(results.image);
    this.scene.add(imagePlane);

    this.faceGroups.forEach(group => this.scene.remove(group));
    this.faceGroups = [];

    for (const faceGeometry of results.multiFaceGeometry) {
      if (!this.model) continue;

      const poseTransformMatrixData = faceGeometry.getPoseTransformMatrix?.();
      if (!poseTransformMatrixData) continue;

      const newGroup = new THREE.Group();
      newGroup.matrixAutoUpdate = false;

      const clone = this.model.clone(true);
      const [px, py, pz] = this.modelConfig.position;
      clone.position.set(px, py, pz);
      newGroup.add(clone);

      newGroup.matrix.fromArray(poseTransformMatrixData.getPackedDataList());
      newGroup.visible = true;

      this.scene.add(newGroup);
      this.faceGroups.push(newGroup);
    }

    this.renderer.render(this.scene, this.camera);
    this.scene.remove(imagePlane);
  }

  createGpuBufferPlane(gpuBuffer) {
    const depth = this.VIDEO_DEPTH;
    const fov = this.camera.fov;
    const width = canvasElement.width;
    const height = canvasElement.height;
    const aspect = width / height;
    const viewportHeightAtDepth = 2 * depth * Math.tan(THREE.MathUtils.degToRad(0.5 * fov));
    const viewportWidthAtDepth = viewportHeightAtDepth * aspect;

    const texture = new THREE.CanvasTexture(gpuBuffer);
    texture.minFilter = THREE.LinearFilter;
    texture.encoding = THREE.sRGBEncoding;

    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: texture })
    );
    plane.scale.set(viewportWidthAtDepth, viewportHeightAtDepth, 1);
    plane.position.set(0, 0, -depth);
    return plane;
  }

  onCanvasDimsUpdate() {
    this.camera = new THREE.PerspectiveCamera(
      this.FOV_DEGREES,
      canvasElement.width / canvasElement.height,
      this.NEAR,
      this.FAR
    );
    this.renderer.setSize(canvasElement.width, canvasElement.height);
  }
}

const effectRenderer = new EffectRenderer('glasses');

function onResults(results) {
  effectRenderer.render(results);
}

const faceMesh = new FaceMesh(config);
faceMesh.setOptions(solutionOptions);
faceMesh.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceMesh.send({ image: videoElement });
  },
  width: 1280,
  height: 720,
});
camera.start();