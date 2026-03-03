import type { IHypercubeEngine, FlatTensorView } from "./IHypercubeEngine";

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
     */
    /**
     * Initialisation spécifique au GPU. Prépare les pipelines et les bindings.
     */
    public initGPU(device: GPUDevice, cubeBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void {
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

        this.pipelineLBM = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_lbm' }
        });

        this.pipelineVorticity = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_vorticity' }
        });

        this.uniformBuffer = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const strideFloats = stride / 4;
        const uniformData = new ArrayBuffer(32);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx;
        u32[1] = ny;
        u32[2] = nz;
        f32[3] = 0.12; // u0
        f32[4] = 1.95; // omega
        u32[5] = strideFloats;

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
    public computeGPU(device: GPUDevice, commandEncoder: GPUCommandEncoder, nx: number, ny: number, nz: number): void {
        if (!this.bindGroup || !this.pipelineLBM || !this.pipelineVorticity) return;

        const wgSize = 16;
        const wgX = Math.ceil(nx / wgSize);
        const wgY = Math.ceil(ny / wgSize);

        // --- PASS 1: LBM Core (Collision + Streaming) ---
        const pass1 = commandEncoder.beginComputePass();
        pass1.setBindGroup(0, this.bindGroup);
        pass1.setPipeline(this.pipelineLBM);
        pass1.dispatchWorkgroups(wgX, wgY, nz);
        pass1.end();

        // --- PASS 2: Vorticité (Dépend de Pass 1) ---
        const pass2 = commandEncoder.beginComputePass();
        pass2.setBindGroup(0, this.bindGroup);
        pass2.setPipeline(this.pipelineVorticity);
        pass2.dispatchWorkgroups(wgX, wgY, nz);
        pass2.end();
    }

    public compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        const obstacles = faces[18];
        const ux_out = faces[19];
        const uy_out = faces[20];
        const curl_out = faces[21];

        const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
        const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
        const w = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];
        const opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];

        const u0 = 0.12;
        const omega = 1.95;

        // 0. INITIALISATION
        if (!this.initialized) {
            for (let idx = 0; idx < nx * ny * nz; idx++) {
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

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;

            // 1. LBM CORE
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
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

                    if (x === 0) { ux = u0; uy = 0.0; rho = 1.0; }

                    if (rho > 0) { ux /= rho; uy /= rho; }
                    ux_out[idx] = ux;
                    uy_out[idx] = uy;

                    const u_sq = ux * ux + uy * uy;
                    for (let i = 0; i < 9; i++) {
                        const cu = cx[i] * ux + cy[i] * uy;
                        const feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                        const f_post = faces[i][idx] * (1.0 - omega) + feq * omega;

                        let nnx = x + cx[i], nny = y + cy[i];
                        if (nny < 0) nny = ny - 1; else if (nny >= ny) nny = 0;
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

            // 2. SWAP par couche
            for (let i = 0; i < 9; i++) {
                for (let j = 0; j < nx * ny; j++) {
                    const idx = zOff + j;
                    faces[i][idx] = faces[i + 9][idx];
                }
            }

            // 3. VORTICITY
            for (let y = 0; y < ny; y++) {
                const yM = y > 0 ? y - 1 : 0;
                const yP = y < ny - 1 ? y + 1 : ny - 1;
                const dyDist = (y === 0 || y === ny - 1) ? 1.0 : 2.0;

                for (let x = 0; x < nx; x++) {
                    const xM = x > 0 ? x - 1 : 0;
                    const xP = x < nx - 1 ? x + 1 : nx - 1;
                    const dxDist = (x === 0 || x === nx - 1) ? 1.0 : 2.0;

                    const dUy_dx = (uy_out[zOff + y * nx + xP] - uy_out[zOff + y * nx + xM]) / dxDist;
                    const dUx_dy = (ux_out[zOff + yP * nx + x] - ux_out[zOff + yM * nx + x]) / dyDist;
                    curl_out[zOff + y * nx + x] = dUy_dx - dUx_dy;
                }
            }
        }

        this.dragScore = this.dragScore * 0.95 + (frameDrag * 100 / nz) * 0.05;
    }

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

                var dxDist: f32 = 2.0;
                if (x == 0u || x == N - 1u) { dxDist = 1.0; }
                
                var dyDist: f32 = 2.0;
                if (y == 0u || y == N - 1u) { dyDist = 1.0; }

                let dUy_dx = (get_face(20u, y * N + xP) - get_face(20u, y * N + xM)) / dxDist;
                let dUx_dy = (get_face(19u, yP * N + x) - get_face(19u, yM * N + x)) / dyDist;
                set_face(21u, idx, dUy_dx - dUx_dy);
            }
        `;
    }
}
