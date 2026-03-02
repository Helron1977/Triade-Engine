import type { IHypercubeEngine } from './IHypercubeEngine';

/**
 * Engine de diffusion spatiale O(1) basé sur Summed Area Table (Integral Image).
 * Transforme un filtre box de rayon fixe en opération constante par pixel.
 * 
 * @example
 * const engine = new HeatmapEngine(15, 0.1);
 * // Face 1 = input (ex: 1.0 aux positions sources)
 * // Après compute() -> Face 2 = heatmap lissée
 */
export class HeatmapEngine implements IHypercubeEngine {
    public get name(): string {
        return "Heatmap (O1 Spatial Convolution)";
    }

    public getRequiredFaces(): number {
        return 5; // Utilise les faces: 1 (input), 2 (output) et 4 (temp SAT). Faces 0, 3 sont libres.
    }
    public radius: number;
    public weight: number;

    private pipelineHorizontal: GPUComputePipeline | null = null;
    private pipelineVertical: GPUComputePipeline | null = null;
    private pipelineDiffusion: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;

    /**
     * @param radius Rayon d'influence en cellules
     * @param weight Coefficient multiplicateur à l'arrivée
     */
    constructor(radius: number = 10, weight: number = 1.0) {
        this.radius = radius;
        this.weight = weight;
    }

    /**
     * Initialisation spécifique au GPU. Compile les shaders et prépare le layout.
     */
    public initGPU(device: GPUDevice, cubeBuffer: GPUBuffer, stride: number, mapSize: number): void {
        const shaderModule = device.createShaderModule({ code: this.wgslSource });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                }
            ]
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        this.pipelineHorizontal = device.createComputePipeline({
            layout: pipelineLayout, compute: { module: shaderModule, entryPoint: 'compute_sat_horizontal' }
        });
        this.pipelineVertical = device.createComputePipeline({
            layout: pipelineLayout, compute: { module: shaderModule, entryPoint: 'compute_sat_vertical' }
        });
        this.pipelineDiffusion = device.createComputePipeline({
            layout: pipelineLayout, compute: { module: shaderModule, entryPoint: 'compute_diffusion' }
        });

        const uniformBuffer = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Alignement strict WebGPU 16 bytes: vec4<u32> (mapSize), i32 (radius), f32 (weight), u32 (stride)
        const strideFloats = stride / 4;
        const uniformData = new ArrayBuffer(16);
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
    public computeGPU(device: GPUDevice, commandEncoder: GPUCommandEncoder, mapSize: number): void {
        if (!this.bindGroup) return;

        let passEncoder;

        // --- PASS 1: Prefix Sum Horizontal ---
        if (this.pipelineHorizontal) {
            passEncoder = commandEncoder.beginComputePass();
            passEncoder.setBindGroup(0, this.bindGroup);
            passEncoder.setPipeline(this.pipelineHorizontal);
            // Dispatch : 1 workgroup (size 256) gère 1 ligne. On dispatch mapSize (Y) workgroups.
            passEncoder.dispatchWorkgroups(1, mapSize);
            passEncoder.end();
            device.queue.submit([commandEncoder.finish()]);
            commandEncoder = device.createCommandEncoder(); // Nouveau encoder par Safety
        }

        // --- PASS 2: Prefix Sum Vertical ---
        if (this.pipelineVertical) {
            passEncoder = commandEncoder.beginComputePass();
            passEncoder.setBindGroup(0, this.bindGroup);
            passEncoder.setPipeline(this.pipelineVertical);
            // Dispatch : 1 workgroup (size 256) gère 1 colonne. On dispatch mapSize (X) workgroups.
            passEncoder.dispatchWorkgroups(mapSize, 1);
            passEncoder.end();
            device.queue.submit([commandEncoder.finish()]);
            commandEncoder = device.createCommandEncoder();
        }

        // --- PASS 3: Extraction Box Filter (Diffusion 2D) ---
        if (this.pipelineDiffusion) {
            passEncoder = commandEncoder.beginComputePass();
            passEncoder.setBindGroup(0, this.bindGroup);
            passEncoder.setPipeline(this.pipelineDiffusion);
            passEncoder.dispatchWorkgroups(Math.ceil(mapSize / 16), Math.ceil(mapSize / 16));
            passEncoder.end();
            // Le dernier encoder.finish() sera submit par le MasterGrid.
        }
    }

    /**
     * Exécute le Summed Area Table Algorithm (Face 5) suivi 
     * d'un Box Filter O(1) vers la Synthèse (Face 3).
     */
    compute(faces: Float32Array[], mapSize: number): void {
        const face2 = faces[1]; // Contexte Binaire d'entrée
        const face3 = faces[2]; // Synthèse de Diffusion
        const face5 = faces[4]; // Cheat-code O(1) SAT

        // Clear pass CPU
        if (face5) face5.fill(0);

        // 1. O(N) : Génération Cristallisée (Integral Image)
        for (let y = 0; y < mapSize; y++) {
            for (let x = 0; x < mapSize; x++) {
                const idx = y * mapSize + x;
                const val = face2[idx];
                const top = y > 0 ? face5[(y - 1) * mapSize + x] : 0;
                const left = x > 0 ? face5[y * mapSize + (x - 1)] : 0;
                const topLeft = (y > 0 && x > 0) ? face5[(y - 1) * mapSize + (x - 1)] : 0;

                face5[idx] = val + top + left - topLeft;
            }
        }

        // 2. O(N) : Extraction d'Influence Indépendante du Rayon
        for (let y = 0; y < mapSize; y++) {
            for (let x = 0; x < mapSize; x++) {
                // Clamping (Borne Map) rapide
                const minX = Math.max(0, x - this.radius);
                const minY = Math.max(0, y - this.radius);
                const maxX = Math.min(mapSize - 1, x + this.radius);
                const maxY = Math.min(mapSize - 1, y + this.radius);

                // Récupération O(1) des Opcodes d'angles
                const A = (minX > 0 && minY > 0) ? face5[(minY - 1) * mapSize + (minX - 1)] : 0;
                const B = (minY > 0) ? face5[(minY - 1) * mapSize + maxX] : 0;
                const C = (minX > 0) ? face5[maxY * mapSize + (minX - 1)] : 0;
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
    get wgslSource(): string {
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

            // --- PASS 1: Prefix Sum Horizontal (par ligne, parallèle avec shared mem) ---
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

                // Hillis-Steele scan parallèle (O(log N) steps)
                var offset: u32 = 1u;
                while (offset < 256u) {
                    if (lid >= offset) {
                        shared[lid] += shared[lid - offset];
                    }
                    workgroupBarrier();
                    offset *= 2u;
                }

                // Écriture finale dans SAT (seulement si x < mapSize)
                if (x < mapSize) {
                    cube[base_sat + y * mapSize + x] = shared[lid];
                }
            }

            // --- PASS 2: Prefix Sum Vertical (par colonne, parallèle avec shared mem) ---
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

                // Hillis-Steele scan parallèle
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
}




































