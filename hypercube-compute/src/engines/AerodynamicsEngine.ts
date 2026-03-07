import type { IHypercubeEngine, FlatTensorView } from "./IHypercubeEngine";

/**
 * AerodynamicsEngine - Lattice Boltzmann D2Q9
 * Implémentation robuste avec Bounce-Back intégré au Streaming.
 */
export class AerodynamicsEngine implements IHypercubeEngine {
    public dragScore: number = 0;
    private lastNx: number = 256;
    private lastNy: number = 256;
    public boundaryConfig: any = null;
    private readonly cx = [0, 1, 0, -1, 0, 1, -1, -1, 1]; // E, N, W, S, NE, NW, SW, SE
    private readonly cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    private readonly w = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];

    // WebGPU Attributes
    private pipelineLBM: GPUComputePipeline | null = null;
    private pipelineVorticity: GPUComputePipeline | null = null;
    private pipelineSmoke: GPUComputePipeline | null = null;
    private bindGroup0: GPUBindGroup | null = null;
    private bindGroup1: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    public parity: number = 0;
    public gpuEnabled: boolean = false;

    public get name(): string { return "AerodynamicsEngine LBM D2Q9"; }
    public getTags(): string[] { return ['aerodynamics', '2d', 'arctic', 'lbm']; }

    /**
     * @description Définit les faces sémantiques de l'aérodynamique.
     */
    public getSchema() {
        return {
            faces: [
                { index: 18, label: 'Obstacles', isSynchronized: false, isReadOnly: true },
                { index: 19, label: 'Velocity_X', isSynchronized: true },
                { index: 20, label: 'Velocity_Y', isSynchronized: true },
                { index: 21, label: 'Vorticity', isSynchronized: true },
                { index: 22, label: 'Smoke_P0', isSynchronized: true },
                { index: 23, label: 'Smoke_P1', isSynchronized: true }
            ]
        };
    }

    /**
     * @description Définit la composition visuelle par défaut (Arctic).
     */
    public getVisualProfile() {
        const p = this.parity ?? 0;
        return {
            styleId: 'arctic',
            layers: [
                { faceIndex: 22 + p, faceLabel: 'Smoke', role: 'primary' as const }
            ]
        };
    }

    public getRequiredFaces(): number {
        // (9 pops P0 + 9 pops P1) + (obs, ux, uy, curl, smoke P0, smoke P1)
        return 24;
    }

    public getConfig(): Record<string, any> {
        return {
            boundaryConfig: this.boundaryConfig,
            parity: (this as any).parity || 0
        };
    }

    public setBoundaryConfig(config: any): void {
        this.boundaryConfig = config;
    }

    public getSyncFaces(): number[] {
        const p = (this as any).parity ?? 0;
        const outOffset = (1 - p) * 9;
        const smokeOutIdx = 22 + (1 - p);

        return [
            outOffset + 0, outOffset + 1, outOffset + 2, outOffset + 3, outOffset + 4,
            outOffset + 5, outOffset + 6, outOffset + 7, outOffset + 8,
            18, 19, 20, 21, smokeOutIdx
        ];
    }

    public getEquilibrium(rho: number, ux: number, uy: number): Float32Array {
        const res = new Float32Array(9);
        const u2 = ux * ux + uy * uy;
        for (let k = 0; k < 9; k++) {
            const cu = 3 * (this.cx[k] * ux + this.cy[k] * uy);
            res[k] = this.w[k] * rho * (1 + cu + 0.5 * cu * cu - 1.5 * u2);
        }
        return res;
    }
    public init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker: boolean = false): void {
        if (isWorker) return;
        const u0 = 0.12;
        for (let idx = 0; idx < nx * ny * nz; idx++) {
            const rho = 1.0;
            const ux = u0; const uy = 0.0;
            const u_sq_15 = 1.5 * (ux * ux + uy * uy);
            for (let i = 0; i < 9; i++) {
                const cu = this.cx[i] * ux + this.cy[i] * uy;
                const feq = this.w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                faces[i][idx] = feq;
                faces[i + 9][idx] = feq;
            }
            faces[22][idx] = 0; // Smoke 0
            faces[23][idx] = 0; // Smoke 1
        }
    }

    public initGPU(device: GPUDevice, readBuffer: GPUBuffer, writeBuffer: GPUBuffer, uniformBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void {
        const shaderModule = device.createShaderModule({
            code: this.wgslSource,
            label: 'Aerodynamics Shader'
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
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

        this.pipelineSmoke = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_smoke' }
        });

        this.uniformBuffer = device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Initialize BindGroups for ping-pong
        this.bindGroup0 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: readBuffer } },
                { binding: 1, resource: { buffer: writeBuffer } },
                { binding: 2, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.bindGroup1 = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: writeBuffer } },
                { binding: 1, resource: { buffer: readBuffer } },
                { binding: 2, resource: { buffer: this.uniformBuffer } }
            ]
        });

        const strideFloats = stride / 4;
        const uniformData = new ArrayBuffer(64);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx;
        u32[1] = ny;
        f32[2] = 1.95; // omega
        u32[3] = strideFloats;
        u32[4] = this.parity;
        u32[5] = 0; // Flags

        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        this.gpuEnabled = true;
    }


    public computeGPU(device: GPUDevice, commandEncoder: GPUCommandEncoder, nx: number, ny: number, nz: number, readBuffer: GPUBuffer, writeBuffer: GPUBuffer): void {
        if (!this.pipelineLBM || !this.pipelineVorticity || !this.pipelineSmoke || !this.uniformBuffer || !this.bindGroup0 || !this.bindGroup1) return;

        // Sync parity and flags
        const u32 = new Uint32Array(2);
        u32[0] = this.parity;

        // Flags: Bit 0 = Left, 1 = Right, 2 = Top, 3 = Bottom
        let flags = 0;
        if (this.boundaryConfig) {
            if (this.boundaryConfig.isLeftBoundary) flags |= 1;
            if (this.boundaryConfig.isRightBoundary) flags |= 2;
            if (this.boundaryConfig.isTopBoundary) flags |= 4;
            if (this.boundaryConfig.isBottomBoundary) flags |= 8;
        }
        u32[1] = flags;
        device.queue.writeBuffer(this.uniformBuffer, 16, u32); // Offset 16 (u32[4] and u32[5])

        const bindGroup = (this.parity === 0) ? this.bindGroup0 : this.bindGroup1;

        const wgSize = 16;
        const wgX = Math.ceil(nx / wgSize);
        const wgY = Math.ceil(ny / wgSize);

        // 1. LBM Step (Ping-pong In -> Out)
        const pass1 = commandEncoder.beginComputePass({ label: 'Aero GPU LBM' });
        pass1.setPipeline(this.pipelineLBM);
        pass1.setBindGroup(0, bindGroup);
        pass1.dispatchWorkgroups(wgX, wgY, nz || 1);
        pass1.end();

        // 2. Vorticity Step (Uses just written Velocities in Out buffer)
        const pass2 = commandEncoder.beginComputePass({ label: 'Aero GPU Vorticity' });
        pass2.setPipeline(this.pipelineVorticity);
        pass2.setBindGroup(0, bindGroup);
        pass2.dispatchWorkgroups(wgX, wgY, nz || 1);
        pass2.end();

        // 3. Smoke Advection Step (Ping-pong In -> Out)
        const pass3 = commandEncoder.beginComputePass({ label: 'Aero GPU Smoke' });
        pass3.setPipeline(this.pipelineSmoke);
        pass3.setBindGroup(0, bindGroup);
        pass3.dispatchWorkgroups(wgX, wgY, nz || 1);
        pass3.end();
    }

    public compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        this.lastNx = nx;
        this.lastNy = ny;
        const obstacles = faces[18];
        const ux_out = faces[19];
        const uy_out = faces[20];
        const curl_out = faces[21];
        const smoke = faces[22];

        const cx_w = [
            4.0 / 9.0,         // c0 (0,0)
            1.0 / 9.0,         // c1 (1,0)
            1.0 / 9.0,         // c2 (0,1)
            1.0 / 9.0,         // c3 (-1,0)
            1.0 / 9.0,         // c4 (0,-1)
            1.0 / 36.0,        // c5 (1,1)
            1.0 / 36.0,        // c6 (-1,1)
            1.0 / 36.0,        // c7 (-1,-1)
            1.0 / 36.0         // c8 (1,-1)
        ];

        // const f0_arr = faces[0], f1_arr = faces[1], f2_arr = faces[2], f3_arr = faces[3], f4_arr = faces[4];
        // const f5_arr = faces[5], f6_arr = faces[6], f7_arr = faces[7], f8_arr = faces[8];
        // const out0 = faces[9], out1 = faces[10], out2 = faces[11], out3 = faces[12], out4 = faces[13];
        // const out5 = faces[14], out6 = faces[15], out7 = faces[16], out8 = faces[17];

        if (this.parity === undefined) (this as any).parity = 0;
        const parity = (this as any).parity;
        const nextParity = 1 - parity;

        const offsetIn = parity * 9;
        const offsetOut = nextParity * 9;

        const f_in = [
            faces[offsetIn + 0], faces[offsetIn + 1], faces[offsetIn + 2], faces[offsetIn + 3], faces[offsetIn + 4],
            faces[offsetIn + 5], faces[offsetIn + 6], faces[offsetIn + 7], faces[offsetIn + 8]
        ];
        const f_out = [
            faces[offsetOut + 0], faces[offsetOut + 1], faces[offsetOut + 2], faces[offsetOut + 3], faces[offsetOut + 4],
            faces[offsetOut + 5], faces[offsetOut + 6], faces[offsetOut + 7], faces[offsetOut + 8]
        ];
        const omega = 1.75;

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;

            // 1. PULL-STREAMING, MACROS & COLLISION
            for (let y = 1; y < ny - 1; y++) {
                for (let x = 1; x < nx - 1; x++) {
                    const i = zOff + y * nx + x;

                    if (obstacles[i] > 0.5) {
                        ux_out[i] = 0;
                        uy_out[i] = 0;
                        // Stationary equilibrium for obstacles
                        for (let k = 0; k < 9; k++) f_out[k][i] = cx_w[k];
                        continue;
                    }

                    // --- BOUNDARY CONDITIONS ---
                    const config = this.boundaryConfig;
                    if (config) {
                        if (config.isLeftBoundary && x === 1 && config.left === 'INFLOW') {
                            let scale = 1.0;
                            if (config.isTopBoundary && y < 16) scale = y / 16.0;
                            if (config.isBottomBoundary && y > ny - 17) scale = (ny - 1 - y) / 16.0;

                            const inUx = (config.inflowUx ?? 0.12) * scale;
                            const inUy = (config.inflowUy ?? 0.0) * scale;
                            const inRho = config.inflowDensity ?? 1.0;

                            ux_out[i] = inUx;
                            uy_out[i] = inUy;
                            const u_sq_15 = 1.5 * (inUx * inUx + inUy * inUy);

                            for (let k = 0; k < 9; k++) {
                                const cu = this.cx[k] * inUx + this.cy[k] * inUy;
                                f_out[k][i] = cx_w[k] * inRho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                            }
                            continue;
                        }

                        if (config.isRightBoundary && x === nx - 2 && config.right === 'OUTFLOW') {
                            const prev = i - 1;
                            const uH = ux_out[prev];
                            const vH = uy_out[prev];
                            const u2 = 1.5 * (uH * uH + vH * vH);
                            for (let k = 0; k < 9; k++) {
                                const cu = this.cx[k] * uH + this.cy[k] * vH;
                                f_out[k][i] = cx_w[k] * (1.0 + 3.0 * cu + 4.5 * cu * cu - u2);
                            }
                            ux_out[i] = uH;
                            uy_out[i] = vH;
                            continue;
                        }
                    }

                    const opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];
                    let pf0 = f_in[0][i];
                    let pf1, pf2, pf3, pf4, pf5, pf6, pf7, pf8;

                    // Pull Streaming from neighbors with proper boundaries
                    pf1 = (obstacles[i - 1] > 0.5) ? f_in[3][i] : f_in[1][i - 1];
                    pf2 = (obstacles[i - nx] > 0.5) ? f_in[4][i] : f_in[2][i - nx];
                    pf3 = (obstacles[i + 1] > 0.5) ? f_in[1][i] : f_in[3][i + 1];
                    pf4 = (obstacles[i + nx] > 0.5) ? f_in[2][i] : f_in[4][i + nx];
                    pf5 = (obstacles[i - nx - 1] > 0.5) ? f_in[7][i] : f_in[5][i - nx - 1];
                    pf6 = (obstacles[i - nx + 1] > 0.5) ? f_in[8][i] : f_in[6][i - nx + 1];
                    pf7 = (obstacles[i + nx + 1] > 0.5) ? f_in[5][i] : f_in[7][i + nx + 1];
                    pf8 = (obstacles[i + nx - 1] > 0.5) ? f_in[6][i] : f_in[8][i + nx - 1];

                    let rho = pf0 + pf1 + pf2 + pf3 + pf4 + pf5 + pf6 + pf7 + pf8;
                    let ux = (pf1 + pf5 + pf8) - (pf3 + pf6 + pf7);
                    let uy = (pf2 + pf5 + pf6) - (pf4 + pf7 + pf8);

                    ux /= rho; uy /= rho;
                    ux_out[i] = ux;
                    uy_out[i] = uy;

                    const u_sq_15 = 1.5 * (ux * ux + uy * uy);
                    const om_1 = 1.0 - omega;

                    f_out[0][i] = pf0 * om_1 + (cx_w[0] * rho * (1.0 - u_sq_15)) * omega;
                    f_out[1][i] = pf1 * om_1 + (cx_w[1] * rho * (1.0 + 3.0 * ux + 4.5 * ux * ux - u_sq_15)) * omega;
                    f_out[2][i] = pf2 * om_1 + (cx_w[2] * rho * (1.0 + 3.0 * uy + 4.5 * uy * uy - u_sq_15)) * omega;
                    f_out[3][i] = pf3 * om_1 + (cx_w[3] * rho * (1.0 - 3.0 * ux + 4.5 * ux * ux - u_sq_15)) * omega;
                    f_out[4][i] = pf4 * om_1 + (cx_w[4] * rho * (1.0 - 3.0 * uy + 4.5 * uy * uy - u_sq_15)) * omega;
                    f_out[5][i] = pf5 * om_1 + (cx_w[5] * rho * (1.0 + 3.0 * (ux + uy) + 4.5 * (ux + uy) * (ux + uy) - u_sq_15)) * omega;
                    f_out[6][i] = pf6 * om_1 + (cx_w[6] * rho * (1.0 + 3.0 * (-ux + uy) + 4.5 * (-ux + uy) * (-ux + uy) - u_sq_15)) * omega;
                    f_out[7][i] = pf7 * om_1 + (cx_w[7] * rho * (1.0 + 3.0 * (-ux - uy) + 4.5 * (-ux - uy) * (-ux - uy) - u_sq_15)) * omega;
                    f_out[8][i] = pf8 * om_1 + (cx_w[8] * rho * (1.0 + 3.0 * (ux - uy) + 4.5 * (ux - uy) * (ux - uy) - u_sq_15)) * omega;
                }
            }

            // No explicit swap needed due to ping-pong buffer strategy

            // 3. VORTICITY
            // 3. VORTICITY
            for (let y = 1; y < ny - 1; y++) {
                const yM = y - 1;
                const yP = y + 1;
                for (let x = 1; x < nx - 1; x++) {
                    const xM = x - 1;
                    const xP = x + 1;

                    // Central Difference across 2 units
                    const dUy_dx = (uy_out[zOff + y * nx + xP] - uy_out[zOff + y * nx + xM]) / 2.0;
                    const dUx_dy = (ux_out[zOff + yP * nx + x] - ux_out[zOff + yM * nx + x]) / 2.0;
                    curl_out[zOff + y * nx + x] = dUy_dx - dUx_dy;
                }
            }

            // 4. TRACER ADVECTION (Smoke)
            // Smoke uses a dedicated parity too for perfect smoothness across chunks
            const smoke_in = faces[22 + parity];
            const smoke_out = faces[22 + nextParity];

            for (let y = 1; y < ny - 1; y++) {
                for (let x = 1; x < nx - 1; x++) {
                    const idx = zOff + y * nx + x;
                    if (obstacles[idx] > 0) {
                        smoke_out[idx] = 0;
                        continue;
                    }
                    const vx = ux_out[idx];
                    const vy = uy_out[idx];
                    const sx = x - vx;
                    const sy = y - vy;

                    // BILINEAR INTERPOLATION for "SMOKE" look
                    // Instead of Math.floor, we blend 4 neighbor pixels
                    const x0 = Math.floor(sx), y0 = Math.floor(sy);
                    const x1 = x0 + 1, y1 = y0 + 1;
                    const fx = sx - x0, fy = sy - y0;

                    if (x0 >= 0 && x1 < nx && y0 >= 0 && y1 < ny) {
                        const val00 = smoke_in[y0 * nx + x0];
                        const val10 = smoke_in[y0 * nx + x1];
                        const val01 = smoke_in[y1 * nx + x0];
                        const val11 = smoke_in[y1 * nx + x1];
                        const raw = (val00 * (1 - fx) + val10 * fx) * (1 - fy) + (val01 * (1 - fx) + val11 * fx) * fy;

                        const neighborAvg = (smoke_in[idx - 1] + smoke_in[idx + 1] + smoke_in[idx - nx] + smoke_in[idx + nx]) * 0.25;
                        smoke_out[idx] = (raw * 0.995 + neighborAvg * 0.005) * 0.9999;
                    }

                    if (this.boundaryConfig?.isLeftBoundary && x === 1) {
                        const pitch = Math.floor(ny / 20);
                        if ((y + 2) % pitch <= 2) {
                            smoke_out[idx] = 1.0;
                        }
                    }
                }
            }
        }
    }

    public get wgslSource(): string {
        return `
            struct Config {
                nx: u32,
                ny: u32,
                omega: f32,
                stride: u32,
                parity: u32,
                flags: u32,
            };

            @group(0) @binding(0) var<storage, read_write> cube_in: array<f32>;
            @group(0) @binding(1) var<storage, read_write> cube_out: array<f32>;
            @group(0) @binding(2) var<uniform> config: Config;

            const cx: array<f32, 9> = array<f32, 9>(0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0);
            const cy: array<f32, 9> = array<f32, 9>(0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0);
            const w: array<f32, 9> = array<f32, 9>(0.444444, 0.111111, 0.111111, 0.111111, 0.111111, 0.027777, 0.027777, 0.027777, 0.027777);
            const opp: array<u32, 9> = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);

            fn get_in(f: u32, id: u32) -> f32 {
                return cube_in[f * config.stride + id];
            }

            fn get_out(f: u32, id: u32) -> f32 {
                return cube_out[f * config.stride + id];
            }

            fn set_out(f: u32, id: u32, val: f32) {
                cube_out[f * config.stride + id] = val;
            }

            @compute @workgroup_size(16, 16)
            fn compute_lbm(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = id.x;
                let y = id.y;
                let NX = config.nx;
                let NY = config.ny;
                if (x == 0u || x >= NX - 1u || y == 0u || y >= NY - 1u) { return; }
                let idx = y * NX + x;

                // Ping-pong parity logic
                let offsetIn = config.parity * 9u;
                let offsetOut = (1u - config.parity) * 9u;

                // --- OBSTACLE PERSISTENCE ---
                // Copy obstacle from In to Out to ensure it's available after swap
                set_out(18u, idx, get_in(18u, idx));

                let obs = get_in(18u, idx);
                if (obs > 0.5) { 
                    set_out(19u, idx, 0.0);
                    set_out(20u, idx, 0.0);
                    // Stationary equil (respect parity)
                    for(var i: u32 = 0u; i < 9u; i = i + 1u) {
                        set_out(offsetOut + i, idx, w[i]);
                    }
                    return; 
                }

                var rho: f32 = 0.0;
                var ux: f32 = 0.0;
                var uy: f32 = 0.0;

                // Pull streaming
                var f_pull: array<f32, 9>;
                for(var i: u32 = 0u; i < 9u; i = i + 1u) {
                   let nx = i32(x) - i32(cx[i]);
                   let ny = i32(y) - i32(cy[i]);
                   let n_idx = u32(ny) * NX + u32(nx);
                   
                   if (get_in(18u, n_idx) > 0.5) {
                       f_pull[i] = get_in(offsetIn + opp[i], idx);
                   } else {
                       f_pull[i] = get_in(offsetIn + i, n_idx);
                   }
                   rho = rho + f_pull[i];
                   ux = ux + cx[i] * f_pull[i];
                   uy = uy + cy[i] * f_pull[i];
                }


                if (rho > 0.0) { ux = ux / rho; uy = uy / rho; }
                set_out(19u, idx, ux);
                set_out(20u, idx, uy);

                let u_sq_15 = 1.5 * (ux * ux + uy * uy);
                let om_1 = 1.0 - config.omega;
                for(var i: u32 = 0u; i < 9u; i = i + 1u) {
                    let cu = cx[i] * ux + cy[i] * uy;
                    let feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                    set_out(offsetOut + i, idx, f_pull[i] * om_1 + feq * config.omega);
                }
            }

            @compute @workgroup_size(16, 16)
            fn compute_vorticity(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = id.x;
                let y = id.y;
                let NX = config.nx;
                let NY = config.ny;
                if (x == 0u || x >= NX - 1u || y == 0u || y >= NY - 1u) { return; }
                let idx = y * NX + x;

                // Read velocities from Out buffer (just written by LBM)
                let dUy_dx = (get_out(20u, idx + 1u) - get_out(20u, idx - 1u)) * 0.5;
                let dUx_dy = (get_out(19u, idx + NX) - get_out(19u, idx - NX)) * 0.5;
                set_out(21u, idx, dUy_dx - dUx_dy);
            }

            @compute @workgroup_size(16, 16)
            fn compute_smoke(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = id.x;
                let y = id.y;
                let NX = config.nx;
                let NY = config.ny;
                if (x == 0u || x >= NX - 1u || y == 0u || y >= NY - 1u) { return; }
                let idx = y * NX + x;

                let smokeInIdx = 22u + config.parity;
                let smokeOutIdx = 22u + (1u - config.parity);

                if (get_in(18u, idx) > 0.5) {
                    set_out(smokeOutIdx, idx, 0.0);
                    return;
                }

                // --- SMOKE SOURCE INJECTION ---
                // If this is global left boundary, inject smoke
                if ((config.flags & 1u) != 0u && x == 1u) {
                    let pitch = NY / 20u;
                    if ((y + 2u) % pitch <= 2u) {
                        set_out(smokeOutIdx, idx, 1.0);
                        return;
                    }
                }

                let vx = get_out(19u, idx);
                let vy = get_out(20u, idx);
                let sx = f32(x) - vx;
                let sy = f32(y) - vy;

                let x0 = i32(floor(sx));
                let y0 = i32(floor(sy));
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = sx - f32(x0);
                let fy = sy - f32(y0);

                if (x0 >= 0 && x1 < i32(NX) && y0 >= 0 && y1 < i32(NY)) {
                    let val00 = get_in(smokeInIdx, u32(y0) * NX + u32(x0));
                    let val10 = get_in(smokeInIdx, u32(y0) * NX + u32(x1));
                    let val01 = get_in(smokeInIdx, u32(y1) * NX + u32(x0));
                    let val11 = get_in(smokeInIdx, u32(y1) * NX + u32(x1));
                    
                    let raw = mix(mix(val00, val10, fx), mix(val01, val11, fx), fy);
                    // Slight diffusion/leak to match CPU behavior
                    set_out(smokeOutIdx, idx, raw * 0.998);
                } else {
                    set_out(smokeOutIdx, idx, get_in(smokeInIdx, idx) * 0.99);
                }
            }
        `;
    }
}
