// src/core/gpu/HypercubeGPUContext.ts
var HypercubeGPUContext = class {
  static _device = null;
  static _adapter = null;
  static _isSupported = null;
  /**
   * Vérifie si le navigateur supporte WebGPU.
   */
  static get isSupported() {
    if (this._isSupported === null) {
      this._isSupported = typeof navigator !== "undefined" && "gpu" in navigator;
    }
    return this._isSupported;
  }
  /**
   * Retourne le GPUDevice actif. Renvoie une erreur si non initialisé.
   */
  static get device() {
    if (!this._device) {
      throw new Error("[Hypercube GPU] GPUDevice non initialis\xE9. Appelez HypercubeGPUContext.init() d'abord.");
    }
    return this._device;
  }
  /**
   * Initialise l'interface WebGPU (Adapter + Device).
   * @returns boolean true si succès, false si non supporté.
   */
  static async init(options) {
    if (!this.isSupported) {
      console.warn("[Hypercube GPU] WebGPU n'est pas support\xE9 par ce navigateur.");
      return false;
    }
    if (this._device) {
      return true;
    }
    try {
      this._adapter = await navigator.gpu.requestAdapter(options);
      if (!this._adapter) {
        console.warn("[Hypercube GPU] Aucun adaptateur WebGPU trouv\xE9.");
        return false;
      }
      this._device = await this._adapter.requestDevice();
      this._device.lost.then((info) => {
        console.error(`[Hypercube GPU] WebGPU Device perdu: ${info.message}`);
        this._device = null;
        this._adapter = null;
      });
      return true;
    } catch (error) {
      console.error("[Hypercube GPU] Erreur d'initialisation WebGPU:", error);
      return false;
    }
  }
  /**
   * Compile un code source WGSL en Compute Pipeline WebGPU.
   * @param wgslCode Code source WGSL
   * @param entryPoint Nom de la fonction d'entrée (ex: 'compute_main')
   */
  static createComputePipeline(wgslCode, entryPoint) {
    const shaderModule = this.device.createShaderModule({ code: wgslCode });
    return this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint
      }
    });
  }
  /**
   * Crée et initialise un Storage Buffer WebGPU à partir d'un Float32Array.
   */
  static createStorageBuffer(data) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      // STORAGE pour la lecture/écriture dans le shader
      // COPY_SRC pour relire le résultat depuis le CPU
      // COPY_DST pour mettre à jour les données depuis le CPU
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  }
};

// src/core/HypercubeChunk.ts
var HypercubeChunk = class {
  mapSize;
  faces = [];
  gpuBuffer = null;
  // Un seul buffer contigu pour le GPU (V3 GodMode)
  offset;
  stride;
  // Exposé pour la WorkerPool
  engine = null;
  x;
  y;
  masterBuffer;
  constructor(x, y, mapSize, masterBuffer, numFaces = 6) {
    this.x = x;
    this.y = y;
    this.masterBuffer = masterBuffer;
    this.mapSize = mapSize;
    const allocation = masterBuffer.allocateCube(mapSize, numFaces);
    this.offset = allocation.offset;
    this.stride = allocation.stride;
    const floatCount = mapSize * mapSize;
    for (let i = 0; i < numFaces; i++) {
      this.faces.push(
        new Float32Array(
          masterBuffer.buffer,
          this.offset + i * this.stride,
          floatCount
        )
      );
    }
  }
  setEngine(engine) {
    this.engine = engine;
  }
  /**
   * Initialise la contrepartie GPU (VRAM) de ce Cube Logiciel.
   * Appelé par le HypercubeGrid lors de sa création en mode 'webgpu'.
   */
  initGPU() {
    if (!this.engine) return;
    const totalSize = this.faces.length * this.stride;
    this.gpuBuffer = HypercubeGPUContext.device.createBuffer({
      size: totalSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    HypercubeGPUContext.device.queue.writeBuffer(
      this.gpuBuffer,
      0,
      this.masterBuffer.buffer,
      this.offset,
      totalSize
    );
    if (this.engine.initGPU) {
      this.engine.initGPU(HypercubeGPUContext.device, this.gpuBuffer, this.stride, this.mapSize);
    }
  }
  async compute() {
    if (!this.engine) return;
    await this.engine.compute(this.faces, this.mapSize, this.x, this.y);
  }
  /** Helper pour vider une face spécifique */
  clearFace(faceIndex) {
    this.faces[faceIndex].fill(0);
  }
  /**
   * Rapatrie les données de la VRAM vers la RAM (Zero-Copy Host Buffer).
   * Nécessaire pour la visualisation ou la validation CPU des résultats GPU.
   */
  async syncToHost() {
    if (!this.gpuBuffer) return;
    const totalSize = this.faces.length * this.stride;
    const device = HypercubeGPUContext.device;
    const stagingBuffer = device.createBuffer({
      size: totalSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.gpuBuffer, 0, stagingBuffer, 0, totalSize);
    device.queue.submit([commandEncoder.finish()]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange();
    new Uint8Array(this.masterBuffer.buffer, this.offset, totalSize).set(new Uint8Array(arrayBuffer));
    stagingBuffer.unmap();
    stagingBuffer.destroy();
  }
  /**
   * Envoie les données de la RAM (MasterBuffer) vers la VRAM (GPUBuffer).
   * Indispensable pour l'interactivité ou l'initialisation complexe.
   */
  syncFromHost() {
    if (!this.gpuBuffer) return;
    const totalSize = this.faces.length * this.stride;
    HypercubeGPUContext.device.queue.writeBuffer(
      this.gpuBuffer,
      0,
      this.masterBuffer.buffer,
      this.offset,
      totalSize
    );
  }
};

// src/engines/HeatmapEngine.ts
var HeatmapEngine = class {
  get name() {
    return "Heatmap (O1 Spatial Convolution)";
  }
  getRequiredFaces() {
    return 6;
  }
  radius;
  weight;
  pipelineHorizontal = null;
  pipelineVertical = null;
  pipelineDiffusion = null;
  bindGroup = null;
  /**
   * @param radius Rayon d'influence en cellules
   * @param weight Coefficient multiplicateur à l'arrivée
   */
  constructor(radius = 10, weight = 1) {
    this.radius = radius;
    this.weight = weight;
  }
  /**
   * Initialisation spécifique au GPU. Compile les shaders et prépare le layout.
   */
  initGPU(device, cubeBuffer, stride, mapSize) {
    const shaderModule = device.createShaderModule({ code: this.wgslSource });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" }
        }
      ]
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    this.pipelineHorizontal = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "compute_sat_horizontal" }
    });
    this.pipelineVertical = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "compute_sat_vertical" }
    });
    this.pipelineDiffusion = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "compute_diffusion" }
    });
    const uniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const strideFloats = stride / 4;
    const uniformData = new ArrayBuffer(32);
    new Uint32Array(uniformData, 0)[0] = mapSize;
    new Int32Array(uniformData, 4)[0] = this.radius;
    new Float32Array(uniformData, 8)[0] = this.weight;
    new Uint32Array(uniformData, 12)[0] = strideFloats;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: cubeBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } }
      ]
    });
  }
  /**
   * Dispatch GPU des différents Compute Shaders via des passes distinctes.
   */
  computeGPU(device, commandEncoder, mapSize) {
    if (!this.bindGroup) return;
    if (this.pipelineHorizontal) {
      const pass1 = commandEncoder.beginComputePass();
      pass1.setBindGroup(0, this.bindGroup);
      pass1.setPipeline(this.pipelineHorizontal);
      pass1.dispatchWorkgroups(Math.ceil(mapSize / 256), mapSize);
      pass1.end();
    }
    if (this.pipelineVertical) {
      const pass2 = commandEncoder.beginComputePass();
      pass2.setBindGroup(0, this.bindGroup);
      pass2.setPipeline(this.pipelineVertical);
      pass2.dispatchWorkgroups(mapSize, Math.ceil(mapSize / 256));
      pass2.end();
    }
    if (this.pipelineDiffusion) {
      const pass3 = commandEncoder.beginComputePass();
      pass3.setBindGroup(0, this.bindGroup);
      pass3.setPipeline(this.pipelineDiffusion);
      pass3.dispatchWorkgroups(Math.ceil(mapSize / 16), Math.ceil(mapSize / 16));
      pass3.end();
    }
  }
  /**
   * Exécute le Summed Area Table Algorithm (Face 5) suivi 
   * d'un Box Filter O(1) vers la Synthèse (Face 3).
   */
  compute(faces, mapSize) {
    const face2 = faces[1];
    const face3 = faces[2];
    const face5 = faces[4];
    for (let y = 0; y < mapSize; y++) {
      for (let x = 0; x < mapSize; x++) {
        const idx = y * mapSize + x;
        const val = face2[idx];
        const top = y > 0 ? face5[(y - 1) * mapSize + x] : 0;
        const left = x > 0 ? face5[y * mapSize + (x - 1)] : 0;
        const topLeft = y > 0 && x > 0 ? face5[(y - 1) * mapSize + (x - 1)] : 0;
        face5[idx] = val + top + left - topLeft;
      }
    }
    for (let y = 0; y < mapSize; y++) {
      for (let x = 0; x < mapSize; x++) {
        const minX = Math.max(0, x - this.radius);
        const minY = Math.max(0, y - this.radius);
        const maxX = Math.min(mapSize - 1, x + this.radius);
        const maxY = Math.min(mapSize - 1, y + this.radius);
        const A = minX > 0 && minY > 0 ? face5[(minY - 1) * mapSize + (minX - 1)] : 0;
        const B = minY > 0 ? face5[(minY - 1) * mapSize + maxX] : 0;
        const C = minX > 0 ? face5[maxY * mapSize + (minX - 1)] : 0;
        const D = face5[maxY * mapSize + maxX];
        const sum = D - B - C + A;
        face3[y * mapSize + x] = sum * this.weight;
      }
    }
  }
  /**
   * @WebGPU
   * Code WGSL statique pour décharger le Box Filter SAT O(N) sur le GPU.
   * Binding 0: Face 2 (Input Binary Map)
   * Binding 1: Face 5 (SAT Buffer Intermédiaire)
   * Binding 2: Face 3 (Output Diffusion)
   * Binding 3: Config Uniforms (mapSize, radius, weight)
   */
  get wgslSource() {
    return `
            struct Uniforms {
                mapSize: u32,
                radius: i32,
                weight: f32,
                stride: u32,
            };

            @group(0) @binding(0) var<storage, read_write> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;

            // Face 2: Input, Face 5: SAT, Face 3: Output
            const FACE_IN = 1u;
            const FACE_SAT = 4u;
            const FACE_OUT = 2u;

            // --- PASS 1: Prefix Sum Horizontal (Lignes) ---
            @compute @workgroup_size(256)
            fn compute_sat_horizontal(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let y = global_id.y;
                if (y >= config.mapSize) { return; }

                var current_sum: f32 = 0.0;
                for (var x: u32 = 0u; x < config.mapSize; x++) {
                    let idx = y * config.mapSize + x;
                    current_sum += cube[FACE_IN * config.stride + idx];
                    cube[FACE_SAT * config.stride + idx] = current_sum;
                }
            }

            // --- PASS 2: Prefix Sum Vertical (Colonnes) ---
            @compute @workgroup_size(256)
            fn compute_sat_vertical(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                if (x >= config.mapSize) { return; }

                var current_sum: f32 = 0.0;
                for (var y: u32 = 0u; y < config.mapSize; y++) {
                    let idx = y * config.mapSize + x;
                    current_sum += cube[FACE_SAT * config.stride + idx];
                    cube[FACE_SAT * config.stride + idx] = current_sum;
                }
            }

            // --- PASS 3: Extraction Box Filter (Diffusion) ---
            @compute @workgroup_size(16, 16)
            fn compute_diffusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = i32(global_id.x);
                let y = i32(global_id.y);
                let mapSize = i32(config.mapSize);
                let stride = config.stride;

                if (x >= mapSize || y >= mapSize) { return; }

                let min_x = max(0, x - config.radius);
                let min_y = max(0, y - config.radius);
                let max_x = min(mapSize - 1, x + config.radius);
                let max_y = min(mapSize - 1, y + config.radius);

                var A: f32 = 0.0;
                var B: f32 = 0.0;
                var C: f32 = 0.0;
                
                let base = FACE_SAT * stride;
                if (min_x > 0 && min_y > 0) { A = cube[base + u32((min_y - 1) * mapSize + (min_x - 1))]; }
                if (min_y > 0) { B = cube[base + u32((min_y - 1) * mapSize + max_x)]; }
                if (min_x > 0) { C = cube[base + u32(max_y * mapSize + (min_x - 1))]; }
                
                let D: f32 = cube[base + u32(max_y * mapSize + max_x)];

                let sum = D - B - C + A;
                cube[FACE_OUT * stride + u32(y * mapSize + x)] = sum * config.weight;
            }
        `;
  }
};

// src/engines/FlowFieldEngine.ts
var FlowFieldEngine = class {
  get name() {
    return "FlowFieldEngine-V12";
  }
  getRequiredFaces() {
    return 6;
  }
  targetX = 256;
  targetY = 256;
  constructor(gpuPassCount = 30) {
  }
  async compute(faces, mapSize, chunkX = 0, chunkY = 0) {
    const face3_Integration = faces[2];
    const face4_ForceX = faces[3];
    const face5_ForceY = faces[4];
    const offsetX = chunkX * mapSize;
    const offsetY = chunkY * mapSize;
    for (let y = 0; y < mapSize; y++) {
      const rowOffset = y * mapSize;
      const globalY = offsetY + y;
      const dy = globalY - this.targetY;
      const dySq = dy * dy;
      for (let x = 0; x < mapSize; x++) {
        const idx = rowOffset + x;
        const globalX = offsetX + x;
        const dx = globalX - this.targetX;
        const distSq = dx * dx + dySq;
        const dist = Math.sqrt(distSq);
        face3_Integration[idx] = dist;
        if (dist > 0.1) {
          face4_ForceX[idx] = -dx / dist;
          face5_ForceY[idx] = -dy / dist;
        } else {
          face4_ForceX[idx] = 0;
          face5_ForceY[idx] = 0;
        }
      }
    }
  }
};

// src/engines/FluidEngine.ts
var FluidEngine = class {
  /**
   * @param dt Delta Time virtuel. Contrôle la vitesse apparente de l'écoulement.
   * @param buoyancy Force de flottabilité (Lévitation liée à la température).
   * @param dissipation Taux de disparition du fluide par frame (ex: 0.99 = la fumée s'estompe lentement).
   */
  constructor(dt = 0.8, buoyancy = 0.3, dissipation = 0.995) {
    this.dt = dt;
    this.buoyancy = buoyancy;
    this.dissipation = dissipation;
  }
  get name() {
    return "Fluid Engine (O1 Tensor Nav-Stokes)";
  }
  getRequiredFaces() {
    return 6;
  }
  // CPU: Buffers temporaires ("Ping-Pong") pour stocker l'état précédent lors de l'advection
  prevDensity = null;
  prevHeat = null;
  prevVelX = null;
  prevVelY = null;
  // WebGPU: Pipelines d'Advection (Fluides)
  pipelineForce = null;
  pipelineAdvection = null;
  bindGroup = null;
  /**
   * Interpole (Bilinear Sampling) une valeur sur une grille 2D
   */
  bilerp(x, y, buffer, mapSize) {
    const x0 = Math.max(0, Math.min(Math.floor(x), mapSize - 1));
    const x1 = Math.max(0, Math.min(x0 + 1, mapSize - 1));
    const y0 = Math.max(0, Math.min(Math.floor(y), mapSize - 1));
    const y1 = Math.max(0, Math.min(y0 + 1, mapSize - 1));
    const tx = x - x0;
    const ty = y - y0;
    const v00 = buffer[y0 * mapSize + x0];
    const v10 = buffer[y0 * mapSize + x1];
    const v01 = buffer[y1 * mapSize + x0];
    const v11 = buffer[y1 * mapSize + x1];
    const lerpX1 = v00 * (1 - tx) + v10 * tx;
    const lerpX2 = v01 * (1 - tx) + v11 * tx;
    return lerpX1 * (1 - ty) + lerpX2 * ty;
  }
  /**
   * Calcule la dynamique de fluid (Ajout de forces -> Advection)
   * Version CPU.
   */
  compute(faces, mapSize) {
    const face1_Density = faces[0];
    const face2_Heat = faces[1];
    const face3_VelX = faces[2];
    const face4_VelY = faces[3];
    const totalCells = mapSize * mapSize;
    if (!this.prevDensity || this.prevDensity.length !== totalCells) {
      this.prevDensity = new Float32Array(totalCells);
      this.prevHeat = new Float32Array(totalCells);
      this.prevVelX = new Float32Array(totalCells);
      this.prevVelY = new Float32Array(totalCells);
    }
    this.prevDensity.set(face1_Density);
    this.prevHeat.set(face2_Heat);
    this.prevVelX.set(face3_VelX);
    this.prevVelY.set(face4_VelY);
    for (let i = 0; i < totalCells; i++) {
      const heat = this.prevHeat[i];
      if (heat > 0) {
        face4_VelY[i] -= heat * this.buoyancy * this.dt;
      }
    }
    for (let y = 0; y < mapSize; y++) {
      for (let x = 0; x < mapSize; x++) {
        const idx = y * mapSize + x;
        const vx = face3_VelX[idx];
        const vy = face4_VelY[idx];
        const sourceX = x - vx * this.dt;
        const sourceY = y - vy * this.dt;
        face1_Density[idx] = this.bilerp(sourceX, sourceY, this.prevDensity, mapSize) * this.dissipation;
        face2_Heat[idx] = this.bilerp(sourceX, sourceY, this.prevHeat, mapSize) * this.dissipation;
        face3_VelX[idx] = this.bilerp(sourceX, sourceY, this.prevVelX, mapSize) * 0.99;
        face4_VelY[idx] = this.bilerp(sourceX, sourceY, this.prevVelY, mapSize) * 0.99;
      }
    }
  }
};

// src/engines/AerodynamicsEngine.ts
var AerodynamicsEngine = class {
  dragScore = 0;
  initialized = false;
  // WebGPU Attributes
  pipelineLBM = null;
  pipelineVorticity = null;
  bindGroup = null;
  uniformBuffer = null;
  get name() {
    return "Lattice Boltzmann D2Q9 (O(1))";
  }
  getRequiredFaces() {
    return 22;
  }
  /**
   * Initialisation WebGPU : Prépare les pipelines et les bindings.
   */
  initGPU(device, cubeBuffer, stride, mapSize) {
    const shaderModule = device.createShaderModule({ code: this.wgslSource });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" }
        }
      ]
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    this.pipelineLBM = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "compute_lbm" }
    });
    this.pipelineVorticity = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "compute_vorticity" }
    });
    this.uniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const strideFloats = stride / 4;
    const uniformData = new ArrayBuffer(32);
    new Uint32Array(uniformData, 0, 1)[0] = mapSize;
    new Float32Array(uniformData, 4, 1)[0] = 0.12;
    new Float32Array(uniformData, 8, 1)[0] = 1.95;
    new Uint32Array(uniformData, 12, 1)[0] = strideFloats;
    device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: cubeBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffer } }
      ]
    });
  }
  /**
   * Dispatch GPU via deux passes distinctes.
   */
  computeGPU(device, commandEncoder, mapSize) {
    if (!this.bindGroup || !this.pipelineLBM || !this.pipelineVorticity) return;
    const workgroupSize = 16;
    const workgroupCount = Math.ceil(mapSize / workgroupSize);
    const pass1 = commandEncoder.beginComputePass();
    pass1.setBindGroup(0, this.bindGroup);
    pass1.setPipeline(this.pipelineLBM);
    pass1.dispatchWorkgroups(workgroupCount, workgroupCount);
    pass1.end();
    const pass2 = commandEncoder.beginComputePass();
    pass2.setBindGroup(0, this.bindGroup);
    pass2.setPipeline(this.pipelineVorticity);
    pass2.dispatchWorkgroups(workgroupCount, workgroupCount);
    pass2.end();
  }
  compute(faces, mapSize) {
    const N = mapSize;
    const obstacles = faces[18];
    const ux_out = faces[19];
    const uy_out = faces[20];
    const curl_out = faces[21];
    const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
    const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
    const opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];
    const u0 = 0.12;
    const omega = 1.95;
    if (!this.initialized) {
      for (let idx = 0; idx < N * N; idx++) {
        const rho = 1;
        const ux = u0;
        const uy = 0;
        const u_sq = ux * ux + uy * uy;
        for (let i = 0; i < 9; i++) {
          const cu = cx[i] * ux + cy[i] * uy;
          const feq = w[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u_sq);
          faces[i][idx] = feq;
          faces[i + 9][idx] = feq;
        }
      }
      this.initialized = true;
    }
    let frameDrag = 0;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const idx = y * N + x;
        if (obstacles[idx] > 0) {
          ux_out[idx] = 0;
          uy_out[idx] = 0;
          continue;
        }
        let rho = 0, ux = 0, uy = 0;
        for (let i = 0; i < 9; i++) {
          const f_val = faces[i][idx];
          rho += f_val;
          ux += cx[i] * f_val;
          uy += cy[i] * f_val;
        }
        if (x === 0) {
          ux = u0;
          uy = 0;
          rho = 1;
        }
        if (rho > 0) {
          ux /= rho;
          uy /= rho;
        }
        ux_out[idx] = ux;
        uy_out[idx] = uy;
        const u_sq = ux * ux + uy * uy;
        for (let i = 0; i < 9; i++) {
          const cu = cx[i] * ux + cy[i] * uy;
          const feq = w[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u_sq);
          const f_post = faces[i][idx] * (1 - omega) + feq * omega;
          let nx = x + cx[i], ny = y + cy[i];
          if (ny < 0) ny = N - 1;
          else if (ny >= N) ny = 0;
          if (nx < 0 || nx >= N) continue;
          const nIdx = ny * N + nx;
          if (obstacles[nIdx] > 0) {
            faces[opp[i] + 9][idx] = f_post;
            frameDrag += f_post * cx[i];
          } else {
            faces[i + 9][nIdx] = f_post;
          }
        }
      }
    }
    for (let i = 0; i < 9; i++) faces[i].set(faces[i + 9]);
    this.dragScore = this.dragScore * 0.95 + frameDrag * 100 * 0.05;
    for (let y = 0; y < N; y++) {
      const row = y * N;
      const yM = y > 0 ? y - 1 : 0;
      const yP = y < N - 1 ? y + 1 : N - 1;
      for (let x = 0; x < N; x++) {
        const xM = x > 0 ? x - 1 : 0;
        const xP = x < N - 1 ? x + 1 : N - 1;
        const dUy_dx = uy_out[row + xP] - uy_out[row + xM];
        const dUx_dy = ux_out[yP * N + x] - ux_out[yM * N + x];
        curl_out[row + x] = dUy_dx - dUx_dy;
      }
    }
  }
  get wgslSource() {
    return `
            struct Config {
                mapSize: u32,
                u0: f32,
                omega: f32,
                stride: u32,
            };

            @group(0) @binding(0) var<storage, read_write> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Config;

            const cx: array<f32, 9> = array<f32, 9>(0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0);
            const cy: array<f32, 9> = array<f32, 9>(0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0);
            const w: array<f32, 9> = array<f32, 9>(0.444444, 0.111111, 0.111111, 0.111111, 0.111111, 0.027777, 0.027777, 0.027777, 0.027777);
            const opp: array<u32, 9> = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);

            fn get_face(f: u32, id: u32) -> f32 {
                return cube[f * config.stride + id];
            }

            fn set_face(f: u32, id: u32, val: f32) {
                cube[f * config.stride + id] = val;
            }

            @compute @workgroup_size(16, 16)
            fn compute_lbm(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = id.x;
                let y = id.y;
                let N = config.mapSize;
                if (x >= N || y >= N) { return; }
                let idx = y * N + x;

                let obs = get_face(18u, idx);
                if (obs > 0.5) { 
                    set_face(19u, idx, 0.0);
                    set_face(20u, idx, 0.0);
                    return; 
                }

                var rho: f32 = 0.0;
                var ux: f32 = 0.0;
                var uy: f32 = 0.0;

                for(var i: u32 = 0u; i < 9u; i = i + 1u) {
                    let f_val = get_face(i, idx);
                    rho = rho + f_val;
                    ux = ux + cx[i] * f_val;
                    uy = uy + cy[i] * f_val;
                }

                if (x == 0u) { ux = config.u0; uy = 0.0; rho = 1.0; }
                if (rho > 0.0) { ux = ux / rho; uy = uy / rho; }
                set_face(19u, idx, ux);
                set_face(20u, idx, uy);

                let u_sq = ux * ux + uy * uy;
                for(var i: u32 = 0u; i < 9u; i = i + 1u) {
                    let cu = cx[i] * ux + cy[i] * uy;
                    let feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                    let f_post = get_face(i, idx) * (1.0 - config.omega) + feq * config.omega;

                    var nx: i32 = i32(x) + i32(cx[i]);
                    var ny: i32 = i32(y) + i32(cy[i]);
                    if (ny < 0) { ny = i32(N) - 1; } else if (ny >= i32(N)) { ny = 0; }
                    if (nx < 0 || nx >= i32(N)) { continue; }

                    let n_idx = u32(ny) * N + u32(nx);
                    if (get_face(18u, n_idx) > 0.5) {
                        set_face(opp[i] + 9u, idx, f_post);
                    } else {
                        set_face(i + 9u, n_idx, f_post);
                    }
                }
            }

            @compute @workgroup_size(16, 16)
            fn compute_vorticity(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = id.x;
                let y = id.y;
                let N = config.mapSize;
                if (x >= N || y >= N) { return; }
                let idx = y * N + x;

                for(var i: u32 = 0u; i < 9u; i = i + 1u) {
                    set_face(i, idx, get_face(i + 9u, idx));
                }

                let xM = max(x, 1u) - 1u;
                let xP = min(x + 1u, N - 1u);
                let yM = max(y, 1u) - 1u;
                let yP = min(y + 1u, N - 1u);

                let dUy_dx = get_face(20u, y * N + xP) - get_face(20u, y * N + xM);
                let dUx_dy = get_face(19u, yP * N + x) - get_face(19u, yM * N + x);
                set_face(21u, idx, dUy_dx - dUx_dy);
            }
        `;
  }
};

// src/engines/OceanEngine.ts
var OceanEngine = class {
  get name() {
    return "OceanEngine";
  }
  getRequiredFaces() {
    return 23;
  }
  // Re-use lab-perfect constants
  w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
  cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
  cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
  opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];
  // Caches to avoid per-frame allocations
  feq_cache = new Float32Array(9);
  pulled_f = new Float32Array(9);
  params = {
    tau_0: 0.8,
    smagorinsky: 0.2,
    cflLimit: 0.38,
    bioDiffusion: 0.05,
    bioGrowth: 5e-4,
    vortexRadius: 28,
    vortexStrength: 0.02
  };
  stats = {
    maxU: 0,
    avgTau: 0
  };
  // UI Input simulation (will be fed by the high-level framework/addon)
  interaction = {
    mouseX: 0,
    mouseY: 0,
    active: false
  };
  constructor() {
  }
  /**
   * Entry point: Orchestrates LBM and Bio steps
   */
  compute(faces, size) {
    this.stepLBM(faces, size);
    this.stepBio(faces, size);
  }
  stepLBM(m, size) {
    const rho = m[20], ux = m[18], uy = m[19], obst = m[22];
    let maxU = 0;
    let sumTau = 0;
    let activeCells = 0;
    for (let k = 0; k < 9; k++) m[k + 9].fill(0);
    const mx = this.interaction.mouseX;
    const my = this.interaction.mouseY;
    const isForcing = this.interaction.active;
    const vr2 = this.params.vortexRadius * this.params.vortexRadius;
    for (let y = 1; y < size - 1; y++) {
      for (let x = 1; x < size - 1; x++) {
        const i = y * size + x;
        if (obst[i] > 0.5) {
          for (let k = 0; k < 9; k++) m[k + 9][i] = this.w[k];
          continue;
        }
        let r = 0, vx = 0, vy = 0;
        for (let k = 0; k < 9; k++) {
          const nx = x - this.cx[k];
          const ny = y - this.cy[k];
          const ni = ny * size + nx;
          if (obst[ni] > 0.5) {
            this.pulled_f[k] = m[this.opp[k]][i];
          } else {
            this.pulled_f[k] = m[k][ni];
          }
          r += this.pulled_f[k];
          vx += this.pulled_f[k] * this.cx[k];
          vy += this.pulled_f[k] * this.cy[k];
        }
        let isShockwave = false;
        if (r < 0.8) {
          r = 0.8;
          isShockwave = true;
        } else if (r > 1.2) {
          r = 1.2;
          isShockwave = true;
        } else if (r < 1e-4) r = 1e-4;
        vx /= r;
        vy /= r;
        if (isForcing) {
          const dx = x - mx;
          const dy = y - my;
          const dist2 = dx * dx + dy * dy;
          if (dist2 < vr2) {
            const forceScale = this.params.vortexStrength * 5e-3 * (1 - Math.sqrt(dist2) / this.params.vortexRadius);
            vx += -dy * forceScale;
            vy += dx * forceScale;
          }
        }
        const v2 = vx * vx + vy * vy;
        const speed = Math.sqrt(v2);
        if (speed > maxU) maxU = speed;
        let u2_clamped = v2;
        if (speed > this.params.cflLimit) {
          const scale = this.params.cflLimit / speed;
          vx *= scale;
          vy *= scale;
          u2_clamped = vx * vx + vy * vy;
          isShockwave = true;
        }
        rho[i] = r;
        ux[i] = vx;
        uy[i] = vy;
        if (isShockwave) {
          for (let k = 0; k < 9; k++) {
            const cu = 3 * (this.cx[k] * vx + this.cy[k] * vy);
            m[k + 9][i] = this.w[k] * r * (1 + cu + 0.5 * cu * cu - 1.5 * u2_clamped);
          }
        } else {
          let Pxx = 0, Pyy = 0, Pxy = 0;
          for (let k = 0; k < 9; k++) {
            const cu = 3 * (this.cx[k] * vx + this.cy[k] * vy);
            this.feq_cache[k] = this.w[k] * r * (1 + cu + 0.5 * cu * cu - 1.5 * u2_clamped);
            const fneq = this.pulled_f[k] - this.feq_cache[k];
            Pxx += fneq * this.cx[k] * this.cx[k];
            Pyy += fneq * this.cy[k] * this.cy[k];
            Pxy += fneq * this.cx[k] * this.cy[k];
          }
          const S_norm = Math.sqrt(2 * (Pxx * Pxx + Pyy * Pyy + 2 * Pxy * Pxy));
          let tau_eff = this.params.tau_0 + this.params.smagorinsky * S_norm;
          if (isNaN(tau_eff) || tau_eff < 0.505) tau_eff = 0.505;
          sumTau += tau_eff;
          activeCells++;
          for (let k = 0; k < 9; k++) {
            m[k + 9][i] = this.pulled_f[k] - (this.pulled_f[k] - this.feq_cache[k]) / tau_eff;
          }
        }
      }
    }
    if (activeCells > 0) this.stats.avgTau = sumTau / activeCells;
    this.stats.maxU = maxU;
    for (let k = 0; k < 9; k++) {
      const tmp = m[k];
      m[k] = m[k + 9];
      m[k + 9] = tmp;
    }
  }
  stepBio(m, size) {
    const bio = m[21];
    const bio_next = m[17];
    const area = size * size;
    for (let y = 1; y < size - 1; y++) {
      for (let x = 1; x < size - 1; x++) {
        const i = y * size + x;
        const lap = bio[i - 1] + bio[i + 1] + bio[i - size] + bio[i + size] - 4 * bio[i];
        let next = bio[i] + this.params.bioDiffusion * lap + this.params.bioGrowth * bio[i] * (1 - bio[i]);
        if (next < 0) next = 0;
        if (next > 1) next = 1;
        bio_next[i] = next;
      }
    }
    for (let i = 0; i < area; i++) bio[i] = bio_next[i];
  }
};

// src/core/cpu/cpu.worker.ts
var WorkerMasterBufferDummy = class {
  buffer;
  _offset = 0;
  _stride = 0;
  constructor(sharedBuf) {
    this.buffer = sharedBuf;
  }
  // Ne fait rien, car le cube est déjà alloué par le Main Thread
  allocateCube(mapSize, numFaces = 6) {
    return { offset: this._offset, stride: this._stride };
  }
  setMockLocation(offset, stride) {
    this._offset = offset;
    this._stride = stride;
  }
};
self.onmessage = (e) => {
  const data = e.data;
  if (data.type === "COMPUTE") {
    const { engineName, engineConfig, sharedBuffer, cubeOffset, stride, numFaces, mapSize, chunkX, chunkY } = data;
    if (!sharedBuffer) {
      console.error("[Worker] Pas de SharedArrayBuffer re\xE7u.");
      postMessage({ type: "DONE", success: false });
      return;
    }
    let engine = null;
    if (engineName === "Heatmap (O1 Spatial Convolution)") {
      engine = new HeatmapEngine(engineConfig?.radius, engineConfig?.weight);
    } else if (engineName === "FlowFieldEngine-V12") {
      engine = new FlowFieldEngine();
      if (engineConfig && "targetX" in engineConfig) {
        engine.targetX = engineConfig.targetX;
        engine.targetY = engineConfig.targetY;
      }
    } else if (engineName === "Simplified Fluid Dynamics") {
      engine = new FluidEngine(engineConfig?.dt, engineConfig?.buoyancy, engineConfig?.dissipation);
    } else if (engineName === "Lattice Boltzmann D2Q9 (O(1))") {
      engine = new AerodynamicsEngine();
    } else if (engineName === "OceanEngine") {
      engine = new OceanEngine();
      if (engineConfig && Object.keys(engineConfig).length > 0) {
        engine.params = engineConfig;
      }
    } else {
      console.error(`[Worker] Moteur non reconnu ou non support\xE9 par les Web Workers: ${engineName}`);
      postMessage({ type: "DONE", success: false });
      return;
    }
    if (!engine) {
      console.error(`[Worker CPU] Erreur fatale: engine est null.`);
      postMessage({ type: "DONE", success: false });
      return;
    }
    const dummyBuffer = new WorkerMasterBufferDummy(sharedBuffer);
    dummyBuffer.setMockLocation(cubeOffset, stride);
    const cube = new HypercubeChunk(chunkX || 0, chunkY || 0, mapSize, dummyBuffer, numFaces || 6);
    cube.setEngine(engine);
    try {
      Promise.resolve(cube.compute()).then(() => {
        postMessage({ type: "DONE", success: true });
      }).catch((err) => {
        console.error(`[Worker CPU] Crash asynchrone pendant l'ex\xE9cution du moteur ${engineName}:`, err);
        postMessage({ type: "DONE", success: false, error: err?.message });
      });
    } catch (error) {
      console.error(`[Worker CPU] Crash synchrone pendant l'ex\xE9cution du moteur ${engineName}:`, error);
      postMessage({ type: "DONE", success: false, error: error?.message });
    }
  }
};
