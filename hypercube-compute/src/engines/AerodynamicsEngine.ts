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
    private readonly cx = [0, 1, 0, -1, 0, -1, -1, 1, 1]; // Corrected order if needed, but matched with Ocean for consistency
    private readonly cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    private readonly w = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];

    // WebGPU Attributes
    private pipelineLBM: GPUComputePipeline | null = null;
    private pipelineVorticity: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;

    public get name(): string {
        return "Aerodynamics LBM D2Q9";
    }

    public getRequiredFaces(): number {
        return 23; // 9(fi) + 9(f_post) + 1(obstacles) + 3(ux, uy, curl) + 1(smoke)
    }

    public getConfig(): Record<string, any> {
        return {
            boundaryConfig: this.boundaryConfig
        };
    }

    public setBoundaryConfig(config: any): void {
        this.boundaryConfig = config;
    }

    public getSyncFaces(): number[] {
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 22]; // Sync LBM pops + macros + smoke
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
        if (isWorker) return; // Do not overwrite SAB data initialized by HypercubeGrid

        const u0 = 0.12;
        for (let idx = 0; idx < nx * ny * nz; idx++) {
            const rho = 1.0;
            const ux = u0; const uy = 0.0;
            const u_sq = ux * ux + uy * uy;
            for (let i = 0; i < 9; i++) {
                const cu = this.cx[i] * ux + this.cy[i] * uy;
                const feq = this.w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                faces[i][idx] = feq;
                faces[i + 9][idx] = feq;
            }
        }
    }

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
        f32[1] = 0.12; // u0
        f32[2] = 1.95; // omega
        u32[3] = strideFloats;

        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: cubeBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });
    }



    public computeGPU(device: GPUDevice, commandEncoder: GPUCommandEncoder, nx: number, ny: number, nz: number): void {
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

        const f0_arr = faces[0], f1_arr = faces[1], f2_arr = faces[2], f3_arr = faces[3], f4_arr = faces[4];
        const f5_arr = faces[5], f6_arr = faces[6], f7_arr = faces[7], f8_arr = faces[8];
        const out0 = faces[9], out1 = faces[10], out2 = faces[11], out3 = faces[12], out4 = faces[13];
        const out5 = faces[14], out6 = faces[15], out7 = faces[16], out8 = faces[17];

        const u0 = 0.12;
        const omega = 1.95;

        let frameDrag = 0;

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;

            // 1. PULL-STREAMING, MACROS & COLLISION (O1 Optimized)
            for (let y = 1; y < ny - 1; y++) {
                for (let x = 1; x < nx - 1; x++) {
                    const i = zOff + y * nx + x;

                    if (obstacles[i] > 0) {
                        ux_out[i] = 0;
                        uy_out[i] = 0;
                        continue;
                    }
                    // --- BOUNDARY CONDITIONS (Config Driven) ---
                    const config = this.boundaryConfig;
                    if (config) {
                        // Left Inflow
                        if (config.isLeftBoundary && x === 1 && config.left === 'INFLOW') {
                            const inUx = config.inflowUx ?? 0.12;
                            const inUy = config.inflowUy ?? 0.0;
                            const inRho = config.inflowDensity ?? 1.0;

                            ux_out[i] = inUx;
                            uy_out[i] = inUy;
                            const u_sq = inUx * inUx + inUy * inUy;
                            const u_sq_15 = 1.5 * u_sq;

                            out0[i] = cx_w[0] * inRho * (1.0 - u_sq_15);
                            let cu;
                            cu = inUx; out1[i] = cx_w[1] * inRho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                            cu = inUy; out2[i] = cx_w[2] * inRho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                            cu = -inUx; out3[i] = cx_w[3] * inRho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                            cu = -inUy; out4[i] = cx_w[4] * inRho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                            cu = inUx + inUy; out5[i] = cx_w[5] * inRho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                            cu = -inUx + inUy; out6[i] = cx_w[6] * inRho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                            cu = -inUx - inUy; out7[i] = cx_w[7] * inRho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                            cu = inUx - inUy; out8[i] = cx_w[8] * inRho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                            continue;
                        }

                        // Right Outflow (Simple Extrapolation)
                        if (config.isRightBoundary && x === nx - 2 && config.right === 'OUTFLOW') {
                            out0[i] = out0[i - 1]; out1[i] = out1[i - 1]; out2[i] = out2[i - 1];
                            out3[i] = out3[i - 1]; out4[i] = out4[i - 1]; out5[i] = out5[i - 1];
                            out6[i] = out6[i - 1]; out7[i] = out7[i - 1]; out8[i] = out8[i - 1];
                            ux_out[i] = ux_out[i - 1];
                            uy_out[i] = uy_out[i - 1];
                            continue;
                        }
                    }

                    // --- PULL STREAMING UNROLLED ---
                    // pfX is the population arriving FROM direction X.
                    let pf0 = f0_arr[i];
                    let pf1, pf2, pf3, pf4, pf5, pf6, pf7, pf8;

                    // Dir 1 (cx:1, cy:0) opposite is 3
                    let ni1 = zOff + y * nx + (x - 1);
                    pf1 = (obstacles[ni1] > 0) ? f3_arr[i] : f1_arr[ni1];
                    // Dir 2 (cx:0, cy:1) opposite is 4
                    let ni2 = zOff + (y - 1) * nx + x;
                    pf2 = (obstacles[ni2] > 0) ? f4_arr[i] : f2_arr[ni2];
                    // Dir 3 (cx:-1, cy:0) opposite is 1
                    let ni3 = zOff + y * nx + (x + 1);
                    pf3 = (obstacles[ni3] > 0) ? f1_arr[i] : f3_arr[ni3];
                    // Dir 4 (cx:0, cy:-1) opposite is 2
                    let ni4 = zOff + (y + 1) * nx + x;
                    pf4 = (obstacles[ni4] > 0) ? f2_arr[i] : f4_arr[ni4];

                    // Dir 5 (cx:1, cy:1) opp 7
                    let ni5 = zOff + (y - 1) * nx + (x - 1);
                    pf5 = (obstacles[ni5] > 0) ? f7_arr[i] : f5_arr[ni5];
                    // Dir 6 (cx:-1, cy:1) opp 8
                    let ni6 = zOff + (y - 1) * nx + (x + 1);
                    pf6 = (obstacles[ni6] > 0) ? f8_arr[i] : f6_arr[ni6];
                    // Dir 7 (cx:-1, cy:-1) opp 5
                    let ni7 = zOff + (y + 1) * nx + (x + 1);
                    pf7 = (obstacles[ni7] > 0) ? f5_arr[i] : f7_arr[ni7];
                    // Dir 8 (cx:1, cy:-1) opp 6
                    let ni8 = zOff + (y + 1) * nx + (x - 1);
                    pf8 = (obstacles[ni8] > 0) ? f6_arr[i] : f8_arr[ni8];

                    // MACROS CALCULATION
                    const rho = pf0 + pf1 + pf2 + pf3 + pf4 + pf5 + pf6 + pf7 + pf8;
                    let ux = (pf1 + pf5 + pf8) - (pf3 + pf6 + pf7);
                    let uy = (pf2 + pf5 + pf6) - (pf4 + pf7 + pf8);

                    if (rho > 0) { ux /= rho; uy /= rho; }
                    ux_out[i] = ux;
                    uy_out[i] = uy;

                    const u_sq = ux * ux + uy * uy;
                    const u_sq_15 = 1.5 * u_sq;
                    const one_minus_omega = 1.0 - omega;

                    // EQUILIBRIUM & COLLISION
                    let feq = cx_w[0] * rho * (1.0 - u_sq_15);
                    out0[i] = pf0 * one_minus_omega + feq * omega;

                    let cu = ux;
                    feq = cx_w[1] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                    out1[i] = pf1 * one_minus_omega + feq * omega;

                    cu = uy;
                    feq = cx_w[2] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                    out2[i] = pf2 * one_minus_omega + feq * omega;

                    cu = -ux;
                    feq = cx_w[3] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                    out3[i] = pf3 * one_minus_omega + feq * omega;

                    cu = -uy;
                    feq = cx_w[4] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                    out4[i] = pf4 * one_minus_omega + feq * omega;

                    cu = ux + uy;
                    feq = cx_w[5] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                    out5[i] = pf5 * one_minus_omega + feq * omega;

                    cu = -ux + uy;
                    feq = cx_w[6] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                    out6[i] = pf6 * one_minus_omega + feq * omega;

                    cu = -ux - uy;
                    feq = cx_w[7] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                    out7[i] = pf7 * one_minus_omega + feq * omega;

                    cu = ux - uy;
                    feq = cx_w[8] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - u_sq_15);
                    out8[i] = pf8 * one_minus_omega + feq * omega;
                }
            }

            // 2. SWAP par couche
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

            // 3. VORTICITY
            for (let y = 1; y < ny - 1; y++) {
                const yM = y - 1;
                const yP = y + 1;

                for (let x = 1; x < nx - 1; x++) {
                    const xM = x > 1 ? x - 1 : 1;
                    const xP = x < nx - 2 ? x + 1 : nx - 2;
                    const dxDist = (x === 1 || x === nx - 2) ? 1.0 : 2.0;

                    const loc_yM = y > 1 ? y - 1 : 1;
                    const loc_yP = y < ny - 2 ? y + 1 : ny - 2;
                    const loc_dyDist = (y === 1 || y === ny - 2) ? 1.0 : 2.0;

                    const dUy_dx = (uy_out[zOff + y * nx + xP] - uy_out[zOff + y * nx + xM]) / dxDist;
                    const dUx_dy = (ux_out[zOff + loc_yP * nx + x] - ux_out[zOff + loc_yM * nx + x]) / loc_dyDist;
                    curl_out[zOff + y * nx + x] = dUy_dx - dUx_dy;
                }
            }

            // 4. TRACER ADVECTION (Smoke)
            for (let y = 1; y < ny - 1; y++) {
                for (let x = 1; x < nx - 1; x++) {
                    const idx = zOff + y * nx + x;
                    if (obstacles[idx] > 0) {
                        smoke[idx] = 0;
                        continue;
                    }
                    const vx = ux_out[idx];
                    const vy = uy_out[idx];
                    const sx = x - vx;
                    const sy = y - vy;

                    const ix = Math.floor(sx), iy = Math.floor(sy);
                    if (ix >= 1 && ix < nx - 1 && iy >= 1 && iy < ny - 1) {
                        smoke[idx] = smoke[iy * nx + ix] * 0.995;
                    }

                    if (x === 1 && y > ny / 4 && y < 3 * ny / 4) {
                        smoke[idx] = 1.0;
                    }
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
}
