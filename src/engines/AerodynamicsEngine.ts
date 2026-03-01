import type { IHypercubeEngine } from "./IHypercubeEngine";

/**
 * AerodynamicsEngine - Lattice Boltzmann D2Q9
 * Implémentation robuste avec Bounce-Back intégré au Streaming.
 */
export class AerodynamicsEngine implements IHypercubeEngine {
    public dragScore: number = 0;
    private initialized: boolean = false;

    // WebGPU Attributes
    private pipelineLBM: GPUComputePipeline | null = null;
    private pipelineVorticity: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;

    public get name(): string {
        return "Lattice Boltzmann D2Q9 (O(1))";
    }

    public getRequiredFaces(): number {
        return 22; // 9(fi) + 9(f_post) + 1(obstacles) + 3(ux, uy, curl)
    }

    /**
     * Initialisation WebGPU : Prépare les pipelines et les bindings.
     * Utilise le mode "Packed Buffer" (Binding unique pour tout le cube).
     */
    public initGPU(device: GPUDevice, cubeBuffer: GPUBuffer, stride: number, mapSize: number): void {
        const shaderModule = device.createShaderModule({ code: this.wgslSource });

        // Création d'un Layout explicite pour garantir la compatibilité entre Pipelines et BindGroups
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

        this.pipelineLBM = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_lbm' }
        });

        this.pipelineVorticity = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_vorticity' }
        });

        // Uniforms (mapSize, u0, omega, strideFloats)
        this.uniformBuffer = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const strideFloats = stride / 4;
        const uniformData = new ArrayBuffer(32);
        new Uint32Array(uniformData, 0, 1)[0] = mapSize;
        new Float32Array(uniformData, 4, 1)[0] = 0.12; // u0
        new Float32Array(uniformData, 8, 1)[0] = 1.95; // omega
        new Uint32Array(uniformData, 12, 1)[0] = strideFloats;

        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        // Bind Group : 1 Storage Buffer (Cube) + 1 Uniform Buffer
        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: cubeBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });
    }

    /**
     * Dispatch GPU des shaders LBM et Vorticité.
     */
    public computeGPU(passEncoder: GPUComputePassEncoder, mapSize: number): void {
        if (!this.bindGroup || !this.pipelineLBM || !this.pipelineVorticity) return;

        const workgroupSize = 16;
        const workgroupCount = Math.ceil(mapSize / workgroupSize);

        passEncoder.setBindGroup(0, this.bindGroup);

        // Pass 1: LBM Core (Collision + Streaming)
        passEncoder.setPipeline(this.pipelineLBM);
        passEncoder.dispatchWorkgroups(workgroupCount, workgroupCount);

        // Pass 2: Vorticité
        passEncoder.setPipeline(this.pipelineVorticity);
        passEncoder.dispatchWorkgroups(workgroupCount, workgroupCount);
    }

    public compute(faces: Float32Array[], mapSize: number): void {
        const N = mapSize;
        const obstacles = faces[18];
        const ux_out = faces[19];
        const uy_out = faces[20];
        const curl_out = faces[21];

        // Vecteurs du modèle D2Q9
        const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
        const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
        const w = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];
        const opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];

        const u0 = 0.12;
        const omega = 1.95;

        // 0. INITIALISATION (F_eq)
        if (!this.initialized) {
            for (let idx = 0; idx < N * N; idx++) {
                const rho = 1.0;
                const ux = u0; const uy = 0.0;
                const u_sq = ux * ux + uy * uy;
                for (let i = 0; i < 9; i++) {
                    const cu = cx[i] * ux + cy[i] * uy;
                    const feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                    faces[i][idx] = feq;
                    faces[i + 9][idx] = feq;
                }
            }
            this.initialized = true;
        }

        let frameDrag = 0;

        // 1. LBM CORE (Loop unique : Macro -> Collision -> Streaming/Bounce-Back)
        for (let y = 0; y < N; y++) {
            for (let x = 0; x < N; x++) {
                const idx = y * N + x;

                // On saute les calculs si obstacle (le bounce-back est géré par les voisins fluides)
                if (obstacles[idx] > 0) {
                    ux_out[idx] = 0;
                    uy_out[idx] = 0;
                    continue;
                }

                // A. Calcul Macroscopique
                let rho = 0;
                let ux = 0;
                let uy = 0;
                for (let i = 0; i < 9; i++) {
                    const f_val = faces[i][idx];
                    rho += f_val;
                    ux += cx[i] * f_val;
                    uy += cy[i] * f_val;
                }

                // INLET : Vent forcé à gauche (Tunnel)
                if (x === 0) {
                    ux = u0;
                    uy = 0.0;
                    rho = 1.0;
                }

                if (rho > 0) {
                    ux /= rho;
                    uy /= rho;
                }
                ux_out[idx] = ux;
                uy_out[idx] = uy;

                // B. Collision (BGK)
                const u_sq = ux * ux + uy * uy;
                for (let i = 0; i < 9; i++) {
                    const cu = cx[i] * ux + cy[i] * uy;
                    const feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                    const f_post = faces[i][idx] * (1.0 - omega) + feq * omega;

                    // C. Streaming & Bounce-Back Combiné
                    let nx = x + cx[i];
                    let ny = y + cy[i];

                    // Torique Y (Wrap top/bottom)
                    if (ny < 0) ny = N - 1;
                    else if (ny >= N) ny = 0;

                    // Bordure X : Outlet (Sortie Libre)
                    if (nx < 0 || nx >= N) {
                        // En sortie (nx >= N), on laisse la population s'échapper (perte de masse)
                        // Évite l'accumulation de pression signalée (Steps 4 de la roadmap)
                        continue;
                    }

                    const nIdx = ny * N + nx;

                    if (obstacles[nIdx] > 0) {
                        // BOUNCE-BACK : On renvoie la particule vers soi-même avec la direction opposée
                        faces[opp[i] + 9][idx] = f_post;

                        // Mesure de traînée (impact sur CX+)
                        if (i === 1) frameDrag += f_post;
                    } else {
                        // STREAMING NORMAL
                        faces[i + 9][nIdx] = f_post;
                    }
                }
            }
        }

        // 2. SWAP BUFFERS
        for (let i = 0; i < 9; i++) {
            faces[i].set(faces[i + 9]);
        }

        // Stats UI
        this.dragScore = this.dragScore * 0.95 + (frameDrag * 100) * 0.05;

        // 3. VORTICITY (Curl) - Couverture totale N x N pour éviter les bords noirs
        for (let y = 0; y < N; y++) {
            const row = y * N;
            const yM = y > 0 ? y - 1 : 0;
            const yP = y < N - 1 ? y + 1 : N - 1;
            for (let x = 0; x < N; x++) {
                const idx = row + x;
                const xM = x > 0 ? x - 1 : 0;
                const xP = x < N - 1 ? x + 1 : N - 1;

                const dUy_dx = uy_out[row + xP] - uy_out[row + xM];
                const dUx_dy = ux_out[yP * N + x] - ux_out[yM * N + x];
                curl_out[idx] = dUy_dx - dUx_dy;
            }
        }
    }

    /**
     * Source WGSL pour le moteur Aerodynamics.
     * Gère les 22 bindings de faces.
     */
    public get wgslSource(): string {
        return `
            struct Config {
                mapSize: u32,
                u0: f32,
                omega: f32,
                stride: u32,
            };

            @group(0) @binding(0) var<storage, read_write> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Config;

            // Helper pour accéder à une face spécifique
            fn get_face_val(face_idx: u32, cell_idx: u32) -> f32 {
                return cube[face_idx * config.stride + cell_idx];
            }

            fn set_face_val(face_idx: u32, cell_idx: u32, val: f32) {
                cube[face_idx * config.stride + cell_idx] = val;
            }

            @compute @workgroup_size(16, 16)
            fn compute_lbm(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = id.x;
                let y = id.y;
                let N = config.mapSize;
                if (x >= N || y >= N) { return; }
                let idx = y * N + x;

                // 1. Lire les 9 populations (f0-f8)
                // 2. Calculer rho, ux, uy
                // 3. Collision + Streaming vers f9-f17
                // (Squelette : simple passthrough pour validation)
                let val = get_face_val(0u, idx);
                set_face_val(9u, idx, val);
            }

            @compute @workgroup_size(16, 16)
            fn compute_vorticity(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = id.x;
                let y = id.y;
                let N = config.mapSize;
                if (x >= N || y >= N) { return; }
                // TODO: Calcul du Curl en WGSL
            }
        `;
    }
}




































