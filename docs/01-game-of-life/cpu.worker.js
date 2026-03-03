"use strict";

// src/core/gpu/HypercubeGPUContext.ts
var HypercubeGPUContext = class {
  static _device = null;
  static _adapter = null;
  static _isSupported = null;
  static _preferredFormat = "bgra8unorm";
  // fallback
  static get isSupported() {
    if (this._isSupported === null) {
      this._isSupported = typeof navigator !== "undefined" && "gpu" in navigator;
      if (this.isSupported) {
        this._preferredFormat = navigator.gpu.getPreferredCanvasFormat?.() ?? "bgra8unorm";
      }
    }
    return this._isSupported;
  }
  static get device() {
    if (!this._device) {
      throw new Error("[Hypercube GPU] GPUDevice non initialis\xE9. Appelez init() d'abord.");
    }
    return this._device;
  }
  static get preferredFormat() {
    return this._preferredFormat;
  }
  /**
   * Initialise WebGPU (Adapter + Device).
   * @param options Options pour requestAdapter / requestDevice
   */
  static async init(options = {}) {
    if (!this.isSupported) {
      console.warn("[Hypercube GPU] WebGPU non support\xE9.");
      return false;
    }
    if (this._device) return true;
    try {
      this._adapter = await navigator.gpu.requestAdapter(options);
      if (!this._adapter) {
        console.warn("[Hypercube GPU] Aucun adaptateur trouv\xE9.");
        return false;
      }
      const requiredFeatures = [];
      const requiredLimits = {
        maxComputeInvocationsPerWorkgroup: this._adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: this._adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: this._adapter.limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: this._adapter.limits.maxComputeWorkgroupSizeZ
      };
      this._device = await this._adapter.requestDevice({
        requiredFeatures,
        requiredLimits
      });
      this._device.lost.then((info) => {
        console.error(`[Hypercube GPU] Device perdu: ${info.message} (${info.reason})`);
        this._device = null;
        this._adapter = null;
      });
      return true;
    } catch (err) {
      console.error("[Hypercube GPU] Init \xE9chou\xE9e:", err);
      return false;
    }
  }
  /**
   * Crée un storage buffer à partir de Float32Array.
   * @param data Données initiales
   * @param mappedAtCreation Utiliser mappedAtCreation (true) ou writeBuffer (false)
   */
  static createStorageBuffer(data, mappedAtCreation = true) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation
    });
    if (mappedAtCreation) {
      new Float32Array(buffer.getMappedRange()).set(data);
      buffer.unmap();
    } else {
      this.device.queue.writeBuffer(buffer, 0, data);
    }
    return buffer;
  }
  /**
   * Crée un uniform buffer dynamique.
   * @param data ArrayBuffer ou une vue typée
   */
  static createUniformBuffer(data) {
    const buffer = this.device.createBuffer({
      size: Math.ceil(data.byteLength / 16) * 16,
      // align 16 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const actualData = "buffer" in data ? data : data;
    this.device.queue.writeBuffer(buffer, 0, actualData);
    return buffer;
  }
  /**
   * Compile un WGSL en Compute Pipeline.
   */
  static createComputePipeline(wgslCode, entryPoint) {
    const module2 = this.device.createShaderModule({ code: wgslCode });
    return this.device.createComputePipeline({
      layout: "auto",
      compute: { module: module2, entryPoint }
    });
  }
  /**
   * Nettoyage complet (pour hot-reload ou tab close).
   */
  static destroy() {
    if (this._device) {
      this._device.destroy();
      this._device = null;
    }
    this._adapter = null;
  }
};

// src/core/HypercubeChunk.ts
var HypercubeChunk = class {
  nx;
  ny;
  nz;
  faces = [];
  gpuBuffer = null;
  // Un seul buffer contigu pour le GPU (V3 GodMode)
  offset;
  stride;
  // Exposé pour la WorkerPool
  engine = null;
  x;
  y;
  z;
  masterBuffer;
  constructor(x, y, nx, ny, nz = 1, masterBuffer, numFaces = 6, z = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.masterBuffer = masterBuffer;
    this.nx = nx;
    this.ny = ny;
    this.nz = nz;
    const allocation = masterBuffer.allocateCube(nx, ny, nz, numFaces);
    this.offset = allocation.offset;
    this.stride = allocation.stride;
    const floatCount = nx * ny * nz;
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
  /**
   * Retourne l'index linéaire pour une position (x, y, z) locale au chunk.
   */
  getIndex(lx, ly, lz = 0) {
    return lz * this.ny * this.nx + ly * this.nx + lx;
  }
  /**
   * Extrait une tranche 2D (Slice Z) d'une face spécifique.
   * @returns Un Float32Array (copie) représentant la couche demandée.
   */
  getSlice(faceIndex, lz) {
    const sliceSize = this.nx * this.ny;
    const offset = lz * sliceSize;
    return this.faces[faceIndex].slice(offset, offset + sliceSize);
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
      this.engine.initGPU(HypercubeGPUContext.device, this.gpuBuffer, this.stride, this.nx, this.ny, this.nz);
    }
  }
  async compute() {
    if (!this.engine) return;
    await this.engine.compute(this.faces, this.nx, this.ny, this.nz, this.x, this.y, this.z);
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
    return 5;
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
  /**
   * Initialisation spécifique au GPU. Compile les shaders et prépare le layout.
   */
  initGPU(device, cubeBuffer, stride, nx, ny, nz) {
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
    const u32 = new Uint32Array(uniformData);
    const i32 = new Int32Array(uniformData);
    const f32 = new Float32Array(uniformData);
    u32[0] = nx;
    u32[1] = ny;
    u32[2] = nz;
    i32[3] = this.radius;
    f32[4] = this.weight;
    u32[5] = strideFloats;
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
  computeGPU(device, commandEncoder, nx, ny, nz) {
    if (!this.bindGroup) return;
    let passEncoder;
    if (this.pipelineHorizontal) {
      passEncoder = commandEncoder.beginComputePass();
      passEncoder.setBindGroup(0, this.bindGroup);
      passEncoder.setPipeline(this.pipelineHorizontal);
      passEncoder.dispatchWorkgroups(1, ny, nz);
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      commandEncoder = device.createCommandEncoder();
    }
    if (this.pipelineVertical) {
      passEncoder = commandEncoder.beginComputePass();
      passEncoder.setBindGroup(0, this.bindGroup);
      passEncoder.setPipeline(this.pipelineVertical);
      passEncoder.dispatchWorkgroups(nx, 1, nz);
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      commandEncoder = device.createCommandEncoder();
    }
    if (this.pipelineDiffusion) {
      passEncoder = commandEncoder.beginComputePass();
      passEncoder.setBindGroup(0, this.bindGroup);
      passEncoder.setPipeline(this.pipelineDiffusion);
      passEncoder.dispatchWorkgroups(Math.ceil(nx / 16), Math.ceil(ny / 16), nz);
      passEncoder.end();
    }
  }
  compute(faces, nx, ny, nz) {
    const face1 = faces[0];
    const face3 = faces[2];
    const face5 = faces[4];
    if (face5) face5.fill(0);
    for (let lz = 0; lz < nz; lz++) {
      const zOff = lz * ny * nx;
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const idx = zOff + y * nx + x;
          const val = face1[idx];
          const top = y > 0 ? face5[zOff + (y - 1) * nx + x] : 0;
          const left = x > 0 ? face5[zOff + y * nx + (x - 1)] : 0;
          const topLeft = y > 0 && x > 0 ? face5[zOff + (y - 1) * nx + (x - 1)] : 0;
          face5[idx] = val + top + left - topLeft;
        }
      }
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const minX = Math.max(0, x - this.radius);
          const minY = Math.max(0, y - this.radius);
          const maxX = Math.min(nx - 1, x + this.radius);
          const maxY = Math.min(ny - 1, y + this.radius);
          const A = minX > 0 && minY > 0 ? face5[zOff + (minY - 1) * nx + (minX - 1)] : 0;
          const B = minY > 0 ? face5[zOff + (minY - 1) * nx + maxX] : 0;
          const C = minX > 0 ? face5[zOff + maxY * nx + (minX - 1)] : 0;
          const D = face5[zOff + maxY * nx + maxX];
          const sum = D - B - C + A;
          face3[zOff + y * nx + x] = sum * this.weight;
        }
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

            // Face 0: Input, Face 4: SAT, Face 2: Output
            const FACE_IN = 0u;
            const FACE_SAT = 4u;
            const FACE_OUT = 2u;

            // --- PASS 1: Prefix Sum Horizontal (par ligne, parall\xE8le avec shared mem) ---
            @compute @workgroup_size(256)
            fn compute_sat_horizontal(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
                let y = global_id.y;
                if (y >= config.mapSize) { return; }

                let base_in = FACE_IN * config.stride;
                let base_sat = FACE_SAT * config.stride;
                let mapSize = config.mapSize;

                var shared: array<f32, 256>;  // Shared memory pour workgroup
                let lid = local_id.x;

                // Charge initiale (un thread par colonne dans la ligne)
                let x = global_id.x;
                if (x < mapSize) {
                    shared[lid] = cube[base_in + y * mapSize + x];
                } else {
                    shared[lid] = 0.0;
                }
                workgroupBarrier();

                // Hillis-Steele scan parall\xE8le (O(log N) steps)
                var offset: u32 = 1u;
                while (offset < 256u) {
                    if (lid >= offset) {
                        shared[lid] += shared[lid - offset];
                    }
                    workgroupBarrier();
                    offset *= 2u;
                }

                // \xC9criture finale dans SAT (seulement si x < mapSize)
                if (x < mapSize) {
                    cube[base_sat + y * mapSize + x] = shared[lid];
                }
            }

            // --- PASS 2: Prefix Sum Vertical (par colonne, parall\xE8le avec shared mem) ---
            @compute @workgroup_size(256)
            fn compute_sat_vertical(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
                let x = global_id.x;
                if (x >= config.mapSize) { return; }

                let base_sat = FACE_SAT * config.stride;
                let mapSize = config.mapSize;

                var shared: array<f32, 256>;
                let lid = local_id.y;  // Note : on swap sur y pour colonnes

                let y = global_id.y;
                if (y < mapSize) {
                    shared[lid] = cube[base_sat + y * mapSize + x];
                } else {
                    shared[lid] = 0.0;
                }
                workgroupBarrier();

                // Hillis-Steele scan parall\xE8le
                var offset: u32 = 1u;
                while (offset < 256u) {
                    if (lid >= offset) {
                        shared[lid] += shared[lid - offset];
                    }
                    workgroupBarrier();
                    offset *= 2u;
                }

                if (y < mapSize) {
                    cube[base_sat + y * mapSize + x] = shared[lid];
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
  setTarget(x, y) {
    this.targetX = x;
    this.targetY = y;
  }
  async compute(faces, nx, ny, nz, chunkX = 0, chunkY = 0, chunkZ = 0) {
    const face0_Distance = faces[0];
    const face4_ForceX = faces[3];
    const face5_ForceY = faces[4];
    const offsetX = chunkX * nx;
    const offsetY = chunkY * ny;
    for (let lz = 0; lz < nz; lz++) {
      const zOff = lz * ny * nx;
      for (let y = 0; y < ny; y++) {
        const globalY = offsetY + y;
        const dy = globalY - this.targetY;
        const dySq = dy * dy;
        for (let x = 0; x < nx; x++) {
          const idx = zOff + y * nx + x;
          const globalX = offsetX + x;
          const dx = globalX - this.targetX;
          const distSq = dx * dx + dySq;
          const dist = Math.sqrt(distSq);
          face0_Distance[idx] = dist;
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
  }
};

// src/engines/FluidEngine.ts
var FluidEngine = class {
  constructor(dt = 0.8, buoyancy = 0.3, dissipation = 0.995, velocityDissipation = 0.99, boundary = "clamp", useProjection = false) {
    this.dt = dt;
    this.buoyancy = buoyancy;
    this.dissipation = dissipation;
    this.velocityDissipation = velocityDissipation;
    this.boundary = boundary;
    this.useProjection = useProjection;
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
  bilerp(x, y, buffer, nx, ny, zOff) {
    if (this.boundary === "periodic") {
      x = (x % nx + nx) % nx;
      y = (y % ny + ny) % ny;
    } else {
      x = Math.max(0, Math.min(x, nx - 1));
      y = Math.max(0, Math.min(y, ny - 1));
    }
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    let x1, y1;
    if (this.boundary === "periodic") {
      x1 = (x0 + 1) % nx;
      y1 = (y0 + 1) % ny;
    } else {
      x1 = Math.min(x0 + 1, nx - 1);
      y1 = Math.min(y0 + 1, ny - 1);
    }
    const tx = x - x0;
    const ty = y - y0;
    const v00 = buffer[zOff + y0 * nx + x0];
    const v10 = buffer[zOff + y0 * nx + x1];
    const v01 = buffer[zOff + y1 * nx + x0];
    const v11 = buffer[zOff + y1 * nx + x1];
    const lerpX1 = v00 * (1 - tx) + v10 * tx;
    const lerpX2 = v01 * (1 - tx) + v11 * tx;
    return lerpX1 * (1 - ty) + lerpX2 * ty;
  }
  /**
   * Calcule la dynamique de fluid (Ajout de forces -> Advection)
   * Version CPU.
   */
  compute(faces, nx, ny, nz) {
    const face1_Density = faces[0];
    const face2_Heat = faces[1];
    const face3_VelX = faces[2];
    const face4_VelY = faces[3];
    const face5_Curl = faces[4];
    const totalCells = nx * ny * nz;
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
    for (let lz = 0; lz < nz; lz++) {
      const zOff = lz * ny * nx;
      for (let i = 0; i < nx * ny; i++) {
        const idx = zOff + i;
        const heat = this.prevHeat[idx];
        if (heat > 0) {
          face4_VelY[idx] -= heat * this.buoyancy * this.dt;
        }
      }
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const idx = zOff + y * nx + x;
          const vx = face3_VelX[idx];
          const vy = face4_VelY[idx];
          const sourceX = x - vx * this.dt;
          const sourceY = y - vy * this.dt;
          face1_Density[idx] = this.bilerp(sourceX, sourceY, this.prevDensity, nx, ny, zOff) * this.dissipation;
          face2_Heat[idx] = this.bilerp(sourceX, sourceY, this.prevHeat, nx, ny, zOff) * this.dissipation;
          face3_VelX[idx] = this.bilerp(sourceX, sourceY, this.prevVelX, nx, ny, zOff) * this.velocityDissipation;
          face4_VelY[idx] = this.bilerp(sourceX, sourceY, this.prevVelY, nx, ny, zOff) * this.velocityDissipation;
        }
      }
      if (face5_Curl) {
        for (let y = 1; y < ny - 1; y++) {
          for (let x = 1; x < nx - 1; x++) {
            const idx = zOff + y * nx + x;
            const dVx_dy = (this.prevVelX[zOff + (y + 1) * nx + x] - this.prevVelX[zOff + (y - 1) * nx + x]) * 0.5;
            const dVy_dx = (this.prevVelY[zOff + y * nx + x + 1] - this.prevVelY[zOff + y * nx + x - 1]) * 0.5;
            face5_Curl[idx] = dVy_dx - dVx_dy;
          }
        }
      }
    }
    if (this.useProjection) {
      for (let lz = 0; lz < nz; lz++) {
        const zOff = lz * ny * nx;
        const vX = faces[2].subarray(zOff, zOff + nx * ny);
        const vY = faces[3].subarray(zOff, zOff + nx * ny);
        this.project(vX, vY, new Float32Array(nx * ny), new Float32Array(nx * ny), nx, ny);
      }
    }
  }
  /**
   * Méthode de projection (Incompressibilité de Poisson).
   * Stubbé pour l'instant car très lourd en CPU.
   */
  project(velX, velY, p, div, nx, ny, iter = 20) {
  }
  /**
   * Ajoute un "splat" (source) de densité, chaleur et vélocité. Idéal pour les inputs utilisateur (souris, clavier).
   */
  addSplat(faces, nx, ny, nz, lz, cx, cy, vx, vy, radius = 20, densityAmt = 1, heatAmt = 5) {
    const face1_Density = faces[0];
    const face2_Heat = faces[1];
    const face3_VelX = faces[2];
    const face4_VelY = faces[3];
    const zOff = lz * ny * nx;
    const r2 = radius * radius;
    for (let y = 0; y < ny; y++) {
      const dy = y - cy;
      const dy2 = dy * dy;
      if (dy2 > r2) continue;
      for (let x = 0; x < nx; x++) {
        const dx = x - cx;
        if (dx * dx + dy2 <= r2) {
          const idx = zOff + y * nx + x;
          const falloff = 1 - (dx * dx + dy2) / r2;
          const f = Math.max(0, falloff);
          face1_Density[idx] += densityAmt * f;
          face2_Heat[idx] += heatAmt * f;
          face3_VelX[idx] += vx * f;
          face4_VelY[idx] += vy * f;
        }
      }
    }
  }
  /**
   * Calcule la masse totale du fluide.
   */
  getTotalDensity(faces) {
    let total = 0;
    const density = faces[0];
    for (let i = 0; i < density.length; i++) {
      total += density[i];
    }
    return total;
  }
  /**
   * Réinitialise toutes les faces du fluide.
   */
  reset(faces) {
    for (const face of faces) {
      if (face) face.fill(0);
    }
  }
};

// src/engines/AerodynamicsEngine.ts
var AerodynamicsEngine = class {
  dragScore = 0;
  initialized = false;
  isLeftBoundary = true;
  // Par défaut, injecte du vent à gauche
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
  getSyncFaces() {
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20];
  }
  /**
   * Initialisation WebGPU : Prépare les pipelines et les bindings.
   */
  /**
   * Initialisation spécifique au GPU. Prépare les pipelines et les bindings.
   */
  initGPU(device, cubeBuffer, stride, nx, ny, nz) {
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
      // 5 * 4 bytes = 20, aligned to 32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const strideFloats = stride / 4;
    const uniformData = new ArrayBuffer(32);
    const u32 = new Uint32Array(uniformData);
    const f32 = new Float32Array(uniformData);
    u32[0] = nx;
    f32[1] = 0.12;
    f32[2] = 1.95;
    u32[3] = strideFloats;
    u32[4] = this.isLeftBoundary ? 1 : 0;
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
  computeGPU(device, commandEncoder, nx, ny, nz) {
    if (!this.bindGroup || !this.pipelineLBM || !this.pipelineVorticity) return;
    const wgSize = 16;
    const wgX = Math.ceil(nx / wgSize);
    const wgY = Math.ceil(ny / wgSize);
    const pass1 = commandEncoder.beginComputePass();
    pass1.setBindGroup(0, this.bindGroup);
    pass1.setPipeline(this.pipelineLBM);
    pass1.dispatchWorkgroups(wgX, wgY, nz);
    pass1.end();
    const pass2 = commandEncoder.beginComputePass();
    pass2.setBindGroup(0, this.bindGroup);
    pass2.setPipeline(this.pipelineVorticity);
    pass2.dispatchWorkgroups(wgX, wgY, nz);
    pass2.end();
  }
  compute(faces, nx, ny, nz) {
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
      for (let idx = 0; idx < nx * ny * nz; idx++) {
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
    for (let lz = 0; lz < nz; lz++) {
      const zOff = lz * ny * nx;
      for (let y = 1; y < ny - 1; y++) {
        for (let x = 1; x < nx - 1; x++) {
          const idx = zOff + y * nx + x;
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
          if (x === 1 && this.isLeftBoundary) {
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
            let nnx = x + cx[i], nny = y + cy[i];
            if (nny < 0) nny = ny - 1;
            else if (nny >= ny) nny = 0;
            if (nnx < 0 || nnx >= nx) continue;
            const nIdx = zOff + nny * nx + nnx;
            if (obstacles[nIdx] > 0) {
              faces[opp[i] + 9][idx] = f_post;
              frameDrag += f_post * cx[i];
            } else {
              faces[i + 9][nIdx] = f_post;
            }
          }
        }
      }
      for (let i = 0; i < 9; i++) {
        const f_in = faces[i];
        const f_out = faces[i + 9];
        for (let y = 0; y < ny; y++) {
          for (let x = 0; x < nx; x++) {
            const idx = zOff + y * nx + x;
            f_in[idx] = f_out[idx];
          }
        }
      }
      for (let y = 1; y < ny - 1; y++) {
        const yM = y - 1;
        const yP = y + 1;
        for (let x = 1; x < nx - 1; x++) {
          const xM = x > 1 ? x - 1 : 1;
          const xP = x < nx - 2 ? x + 1 : nx - 2;
          const dxDist = x === 1 || x === nx - 2 ? 1 : 2;
          const loc_yM = y > 1 ? y - 1 : 1;
          const loc_yP = y < ny - 2 ? y + 1 : ny - 2;
          const loc_dyDist = y === 1 || y === ny - 2 ? 1 : 2;
          const dUy_dx = (uy_out[zOff + y * nx + xP] - uy_out[zOff + y * nx + xM]) / dxDist;
          const dUx_dy = (ux_out[zOff + loc_yP * nx + x] - ux_out[zOff + loc_yM * nx + x]) / loc_dyDist;
          curl_out[zOff + y * nx + x] = dUy_dx - dUx_dy;
        }
      }
      this.initialized = true;
    }
    this.dragScore = this.dragScore * 0.95 + frameDrag * 100 / nz * 0.05;
  }
  get wgslSource() {
    return `
            struct Config {
                mapSize: u32,
                u0: f32,
                omega: f32,
                stride: u32,
                isLeftBoundary: u32,
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
                if (x == 0u || x >= N - 1u || y == 0u || y >= N - 1u) { return; }
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

                if (x == 1u && config.isLeftBoundary > 0u) { ux = config.u0; uy = 0.0; rho = 1.0; }
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
                if (x == 0u || x >= N - 1u || y == 0u || y >= N - 1u) { return; }
                let idx = y * N + x;

                for(var i: u32 = 0u; i < 9u; i = i + 1u) {
                    set_face(i, idx, get_face(i + 9u, idx));
                }

                let xM = max(x, 2u) - 1u;      // clamp to 1 
                let xP = min(x + 1u, N - 2u);  // clamp to N-2
                let yM = max(y, 2u) - 1u;
                let yP = min(y + 1u, N - 2u);

                var dxDist: f32 = 2.0;
                if (x == 1u || x == N - 2u) { dxDist = 1.0; }

                var dyDist: f32 = 2.0;
                if (y == 1u || y == N - 2u) { dyDist = 1.0; }

                let dUy_dx = (get_face(20u, y * N + xP) - get_face(20u, y * N + xM)) / dxDist;
                let dUx_dy = (get_face(19u, yP * N + x) - get_face(19u, yM * N + x)) / dyDist;
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
    return 25;
  }
  getSyncFaces() {
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20];
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
    vortexStrength: 0.02,
    closedBounds: false
  };
  stats = {
    maxU: 0,
    avgTau: 0,
    avgRho: 0
  };
  // UI Input simulation (will be fed by the high-level framework/addon)
  interaction = {
    mouseX: 0,
    mouseY: 0,
    active: false
  };
  constructor() {
  }
  addGlobalCurrent(faces, targetUx, targetUy) {
    const nx = 256;
    const ny = 256;
    const ux = faces[19];
    const uy = faces[20];
    for (let i = 0; i < nx * ny; i++) {
      ux[i] += targetUx;
      uy[i] += targetUy;
    }
  }
  addVortex(faces, mx, my, strength = 10) {
    this.interaction.mouseX = mx;
    this.interaction.mouseY = my;
    this.interaction.active = true;
    setTimeout(() => {
      this.interaction.active = false;
    }, 50);
  }
  /**
   * Entry point: Orchestrates LBM and Bio steps
   */
  compute(faces, nx, ny, nz) {
    for (let lz = 0; lz < nz; lz++) {
      this.stepLBM(faces, nx, ny, lz);
      this.stepBio(faces, nx, ny, lz);
    }
  }
  stepLBM(faces, nx, ny, lz) {
    const size = nx;
    const rho = faces[22], ux = faces[19], uy = faces[20], obst = faces[18];
    const zOff = lz * ny * nx;
    let maxU = 0;
    let sumTau = 0;
    let sumRho = 0;
    let activeCells = 0;
    for (let k = 0; k < 9; k++) {
      for (let i = 0; i < nx * ny; i++) faces[k + 9][zOff + i] = 0;
    }
    const mx = this.interaction.mouseX;
    const my = this.interaction.mouseY;
    const isForcing = this.interaction.active;
    const vr2 = this.params.vortexRadius * this.params.vortexRadius;
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const i = zOff + y * nx + x;
        if (obst[i] > 0.5) {
          for (let k = 0; k < 9; k++) faces[k + 9][i] = this.w[k];
          continue;
        }
        let r = 0, vx = 0, vy = 0;
        for (let k = 0; k < 9; k++) {
          const local_nx = x - this.cx[k];
          const local_ny = y - this.cy[k];
          if (this.params.closedBounds && (local_nx <= 0 || local_nx >= nx - 1 || local_ny <= 0 || local_ny >= ny - 1)) {
            this.pulled_f[k] = faces[this.opp[k]][i];
          } else {
            const ni = zOff + local_ny * nx + local_nx;
            if (obst[ni] > 0.5) {
              this.pulled_f[k] = faces[this.opp[k]][i];
            } else {
              this.pulled_f[k] = faces[k][ni];
            }
          }
          r += this.pulled_f[k];
          vx += this.pulled_f[k] * this.cx[k];
          vy += this.pulled_f[k] * this.cy[k];
        }
        let isShockwave = false;
        if (r < 0.8 || r > 1.2 || r < 1e-4) {
          const targetRho = Math.max(0.8, Math.min(1.2, r < 1e-4 ? 1 : r));
          const scale = targetRho / r;
          for (let k = 0; k < 9; k++) this.pulled_f[k] *= scale;
          r = targetRho;
          isShockwave = true;
        }
        vx /= r;
        vy /= r;
        let Fx = 0;
        let Fy = 0;
        if (isForcing) {
          const dx = x - mx;
          const dy = y - my;
          const dist2 = dx * dx + dy * dy;
          if (dist2 < vr2) {
            const forceScale = this.params.vortexStrength * 5e-3 * (1 - Math.sqrt(dist2) / this.params.vortexRadius);
            Fx = -dy * forceScale;
            Fy = dx * forceScale;
            vx += Fx / r;
            vy += Fy / r;
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
            faces[k + 9][i] = this.w[k] * r * (1 + cu + 0.5 * cu * cu - 1.5 * u2_clamped);
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
          let S_norm = Math.sqrt(2 * (Pxx * Pxx + Pyy * Pyy + 2 * Pxy * Pxy));
          if (S_norm > 10 || isNaN(S_norm)) S_norm = 10;
          let tau_eff = this.params.tau_0 + this.params.smagorinsky * S_norm;
          if (isNaN(tau_eff) || tau_eff < 0.505) tau_eff = 0.505;
          else if (tau_eff > 2) tau_eff = 2;
          sumTau += tau_eff;
          sumRho += r;
          activeCells++;
          for (let k = 0; k < 9; k++) {
            faces[k + 9][i] = this.pulled_f[k] - (this.pulled_f[k] - this.feq_cache[k]) / tau_eff;
          }
        }
      }
    }
    const curl_out = faces[21];
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const i = zOff + y * nx + x;
        const xM = x > 1 ? x - 1 : 1;
        const xP = x < nx - 2 ? x + 1 : nx - 2;
        const dxDist = x === 1 || x === nx - 2 ? 1 : 2;
        const yM_idx = y > 1 ? y - 1 : 1;
        const yP_idx = y < ny - 2 ? y + 1 : ny - 2;
        const dyDist = y === 1 || y === ny - 2 ? 1 : 2;
        const dUy_dx = (uy[zOff + y * nx + xP] - uy[zOff + y * nx + xM]) / dxDist;
        const dUx_dy = (ux[zOff + yP_idx * nx + x] - ux[zOff + yM_idx * nx + x]) / dyDist;
        curl_out[i] = dUy_dx - dUx_dy;
      }
    }
    if (activeCells > 0) {
      this.stats.avgTau = sumTau / activeCells;
      this.stats.avgRho = sumRho / activeCells;
    }
    this.stats.maxU = maxU;
    for (let k = 0; k < 9; k++) {
      for (let i = 0; i < nx * ny; i++) {
        const idx = zOff + i;
        const tmp = faces[k][idx];
        faces[k][idx] = faces[k + 9][idx];
        faces[k + 9][idx] = tmp;
      }
    }
  }
  stepBio(faces, nx, ny, lz) {
    const bio = faces[23];
    const bio_next = faces[24];
    const zOff = lz * ny * nx;
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const i = zOff + y * nx + x;
        const lap = bio[i - 1] + bio[i + 1] + bio[i - nx] + bio[i + nx] - 4 * bio[i];
        let next = bio[i] + this.params.bioDiffusion * lap + this.params.bioGrowth * bio[i] * (1 - bio[i]);
        const ux = faces[18][i];
        const uy = faces[19][i];
        const ax = Math.max(1, Math.min(nx - 2, x - ux * 0.8));
        const ay = Math.max(1, Math.min(ny - 2, y - uy * 0.8));
        const ix = Math.floor(ax);
        const iy = Math.floor(ay);
        const fx = ax - ix;
        const fy = ay - iy;
        const v00 = bio[zOff + iy * nx + ix];
        const v10 = bio[zOff + iy * nx + Math.min(ix + 1, nx - 2)];
        const v01 = bio[zOff + Math.min(iy + 1, ny - 2) * nx + ix];
        const v11 = bio[zOff + Math.min(iy + 1, ny - 2) * nx + Math.min(ix + 1, nx - 2)];
        const advected = (1 - fy) * ((1 - fx) * v00 + fx * v10) + fy * ((1 - fx) * v01 + fx * v11);
        next = advected + this.params.bioDiffusion * lap + this.params.bioGrowth * bio[i] * (1 - bio[i]);
        if (next < 0) next = 0;
        if (next > 1) next = 1;
        bio_next[i] = next;
      }
    }
    for (let i = 0; i < nx * ny; i++) bio[zOff + i] = bio_next[zOff + i];
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
  allocateCube(nx, ny, nz = 1, numFaces = 6) {
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
    const { engineName, engineConfig, sharedBuffer, cubeOffset, stride, numFaces, nx, ny, nz, chunkX, chunkY } = data;
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
    const cube = new HypercubeChunk(chunkX || 0, chunkY || 0, nx, ny, nz || 1, dummyBuffer, numFaces || 6);
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
