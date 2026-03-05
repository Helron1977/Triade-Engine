import type { IHypercubeEngine } from './IHypercubeEngine';

export interface OceanEngineParams {
    tau_0: number;
    smagorinsky: number;
    cflLimit: number;
    bioDiffusion: number;
    bioGrowth: number;
    closedBounds: boolean;
}

/**
 * OceanEngine – Shallow Water + Plankton Dynamics (D2Q9 LBM)
 * Simulation océanique simplifiée : courants, tourbillons, forcing interactif, et bio-diffusion.
 * 
 * @faces
 * - 0–8   : f (populations LBM)
 * - 9–17  : f_post (post-collision temp buffers)
 * - 18    : obst (murs/îles statiques > 0.5)
 * - 19    : ux (vitesse X vectorielle)
 * - 20    : uy (vitesse Y vectorielle)
 * - 21    : curl (vorticité pour rendu)
 * - 22    : rho (densité de masse locale)
 * - 23    : bio (plancton / concentration passive)
 * - 24    : bio_next (temp buffer pour bio)
 * 
 * Note globale : La propriété `interaction` doit être mise à jour chaque frame 
 * par l'environnement ou un `EventListener` de type "mousemove & mousedown".
 */
export class OceanEngine implements IHypercubeEngine {
    public get name(): string {
        return "OceanEngine";
    }

    public getRequiredFaces(): number {
        return 25; // Suite faces 0-17 + 18-24
    }

    public getSyncFaces(): number[] {
        // En Ping-Pong, on synchronise les faces qui viennent d'être ÉCRITES.
        // Si parity == 1, cela signifie qu'on vient de faire un pass 0 -> 9-17.
        // Si parity == 0, cela signifie qu'on vient de faire un pass 1 -> 0-8.
        const pops = this.parity === 1 ? [9, 10, 11, 12, 13, 14, 15, 16, 17] : [0, 1, 2, 3, 4, 5, 6, 7, 8];
        return [...pops, 18, 19, 20, 22]; // LBM pop + macros (ux, uy, rho)
    }

    public getParity(): number {
        return this.parity;
    }

    // Re-use lab-perfect constants
    private readonly w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
    private readonly cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
    private readonly cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    private readonly opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];

    // Caches to avoid per-frame allocations
    private feq_cache = new Float32Array(9);
    private pulled_f = new Float32Array(9);

    public params: OceanEngineParams = {
        tau_0: 0.8,
        smagorinsky: 0.2,
        cflLimit: 0.38,
        bioDiffusion: 0.05,
        bioGrowth: 0.0005,
        closedBounds: false
    };

    public stats = {
        maxU: 0,
        avgTau: 0,
        avgRho: 0
    };

    constructor() { }

    public getConfig(): Record<string, any> {
        return {
            ...this.params
        };
    }

    private pipelineLBM: GPUComputePipeline | null = null;
    private pipelineBio: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private lastStride: number = 0;
    private parity: number = 0;
    public gpuEnabled: boolean = false;

    public initGPU(device: GPUDevice, cubeBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void {
        this.lastStride = stride / 4; // convert byte stride to float index stride
        const shaderModule = device.createShaderModule({ code: this.getWgslSource() });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        this.pipelineLBM = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_lbm' }
        });

        this.pipelineBio = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_bio' }
        });

        const uniformSize = 16 * 4; // 16 floats/uints
        const uniformData = new ArrayBuffer(uniformSize);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = this.lastStride; // strideFace
        f32[4] = this.params.tau_0; f32[5] = this.params.smagorinsky;
        f32[6] = this.params.cflLimit; f32[7] = this.params.closedBounds ? 1.0 : 0.0;
        f32[8] = this.params.bioDiffusion; f32[9] = this.params.bioGrowth;
        u32[10] = this.parity;

        this.uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: cubeBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.gpuEnabled = true;
    }

    public computeGPU(device: GPUDevice, commandEncoder: GPUCommandEncoder, nx: number, ny: number, nz: number): void {
        if (!this.bindGroup || !this.pipelineLBM || !this.pipelineBio || !this.uniformBuffer) return;

        const uniformSize = 16 * 4;
        const uniformData = new ArrayBuffer(uniformSize);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = this.lastStride;
        f32[4] = this.params.tau_0; f32[5] = this.params.smagorinsky;
        f32[6] = this.params.cflLimit; f32[7] = this.params.closedBounds ? 1.0 : 0.0;
        f32[8] = this.params.bioDiffusion; f32[9] = this.params.bioGrowth;
        u32[10] = this.parity;
        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        this.parity = 1 - this.parity; // Flip for next call (though typically controlled by Grid)

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setBindGroup(0, this.bindGroup);

        passEncoder.setPipeline(this.pipelineLBM);
        passEncoder.dispatchWorkgroups(Math.ceil(nx / 16), Math.ceil(ny / 16), nz || 1);

        passEncoder.setPipeline(this.pipelineBio);
        passEncoder.dispatchWorkgroups(Math.ceil(nx / 16), Math.ceil(ny / 16), nz || 1);

        passEncoder.end();
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
        if (isWorker) return; // Main thread already initialized SAB

        const u0 = 0.0;
        const v0 = 0.0;
        const rho0 = 1.0;
        const u2 = u0 * u0 + v0 * v0;

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;
            for (let i = 0; i < nx * ny; i++) {
                const idx = zOff + i;
                faces[22][idx] = rho0;
                faces[19][idx] = u0;
                faces[20][idx] = v0;
                faces[23][idx] = 0.01; // Initial small plankton amount

                for (let k = 0; k < 9; k++) {
                    const cu = 3 * (this.cx[k] * u0 + this.cy[k] * v0);
                    const feq = this.w[k] * rho0 * (1 + cu + 0.5 * cu * cu - 1.5 * u2);
                    faces[k][idx] = feq;
                    faces[k + 9][idx] = feq;
                }
            }
        }
    }



    /**
     * Entry point: Orchestrates LBM and Bio steps
     */
    compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        for (let lz = 0; lz < nz; lz++) {
            this.stepLBM(faces, nx, ny, lz);
            this.stepBio(faces, nx, ny, lz);
        }
    }

    private stepLBM(faces: Float32Array[], nx: number, ny: number, lz: number): void {
        const size = nx;
        const rho = faces[22], ux = faces[19], uy = faces[20], obst = faces[18];
        const zOff = lz * ny * nx;

        let maxU = 0;
        let sumTau = 0;
        let sumRho = 0;
        let activeCells = 0;

        // Ping-Pong CPU Selection
        const f_in_indices = this.parity === 0 ? [0, 1, 2, 3, 4, 5, 6, 7, 8] : [9, 10, 11, 12, 13, 14, 15, 16, 17];
        const f_out_indices = this.parity === 0 ? [9, 10, 11, 12, 13, 14, 15, 16, 17] : [0, 1, 2, 3, 4, 5, 6, 7, 8];

        const in0 = faces[f_in_indices[0]], in1 = faces[f_in_indices[1]], in2 = faces[f_in_indices[2]], in3 = faces[f_in_indices[3]], in4 = faces[f_in_indices[4]], in5 = faces[f_in_indices[5]], in6 = faces[f_in_indices[6]], in7 = faces[f_in_indices[7]], in8 = faces[f_in_indices[8]];
        const out0 = faces[f_out_indices[0]], out1 = faces[f_out_indices[1]], out2 = faces[f_out_indices[2]], out3 = faces[f_out_indices[3]], out4 = faces[f_out_indices[4]], out5 = faces[f_out_indices[5]], out6 = faces[f_out_indices[6]], out7 = faces[f_out_indices[7]], out8 = faces[f_out_indices[8]];

        const cx_w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
        const isClosed = this.params.closedBounds;

        // 1. PULL-STREAMING, MACROS & COLLISION (O1 Optimized)
        for (let y = 1; y < ny - 1; y++) {
            for (let x = 1; x < nx - 1; x++) {
                const i = zOff + y * nx + x;

                if (obst[i] > 0.5) {
                    out0[i] = cx_w[0]; out1[i] = cx_w[1]; out2[i] = cx_w[2];
                    out3[i] = cx_w[3]; out4[i] = cx_w[4]; out5[i] = cx_w[5];
                    out6[i] = cx_w[6]; out7[i] = cx_w[7]; out8[i] = cx_w[8];
                    continue;
                }

                // --- PULL STREAMING UNROLLED ---
                let pf0 = in0[i];
                let pf1: number, pf2: number, pf3: number, pf4: number, pf5: number, pf6: number, pf7: number, pf8: number;

                // Dir 1 (cx:1, cy:0) opp:3
                let nx1 = x - 1, ny1 = y;
                if (isClosed && nx1 <= 0) pf1 = in3[i]; else { let ni = zOff + ny1 * nx + nx1; pf1 = obst[ni] > 0.5 ? in3[i] : in1[ni]; }
                // Dir 2 (cx:0, cy:1) opp:4
                let nx2 = x, ny2 = y - 1;
                if (isClosed && ny2 <= 0) pf2 = in4[i]; else { let ni = zOff + ny2 * nx + nx2; pf2 = obst[ni] > 0.5 ? in4[i] : in2[ni]; }
                // Dir 3 (cx:-1, cy:0) opp:1
                let nx3 = x + 1, ny3 = y;
                if (isClosed && nx3 >= nx - 1) pf3 = in1[i]; else { let ni = zOff + ny3 * nx + nx3; pf3 = obst[ni] > 0.5 ? in1[i] : in3[ni]; }
                // Dir 4 (cx:0, cy:-1) opp:2
                let nx4 = x, ny4 = y + 1;
                if (isClosed && ny4 >= ny - 1) pf4 = in2[i]; else { let ni = zOff + ny4 * nx + nx4; pf4 = obst[ni] > 0.5 ? in2[i] : in4[ni]; }
                // Dir 5 (cx:1, cy:1) opp:7
                let nx5 = x - 1, ny5 = y - 1;
                if (isClosed && (nx5 <= 0 || ny5 <= 0)) pf5 = in7[i]; else { let ni = zOff + ny5 * nx + nx5; pf5 = obst[ni] > 0.5 ? in7[i] : in5[ni]; }
                // Dir 6 (cx:-1, cy:1) opp:8
                let nx6 = x + 1, ny6 = y - 1;
                if (isClosed && (nx6 >= nx - 1 || ny6 <= 0)) pf6 = in8[i]; else { let ni = zOff + ny6 * nx + nx6; pf6 = obst[ni] > 0.5 ? in8[i] : in6[ni]; }
                // Dir 7 (cx:-1, cy:-1) opp:5
                let nx7 = x + 1, ny7 = y + 1;
                if (isClosed && (nx7 >= nx - 1 || ny7 >= ny - 1)) pf7 = in5[i]; else { let ni = zOff + ny7 * nx + nx7; pf7 = obst[ni] > 0.5 ? in5[i] : in7[ni]; }
                // Dir 8 (cx:1, cy:-1) opp:6
                let nx8 = x - 1, ny8 = y + 1;
                if (isClosed && (nx8 <= 0 || ny8 >= ny - 1)) pf8 = in6[i]; else { let ni = zOff + ny8 * nx + nx8; pf8 = obst[ni] > 0.5 ? in6[i] : in8[ni]; }

                let r = pf0 + pf1 + pf2 + pf3 + pf4 + pf5 + pf6 + pf7 + pf8;
                let vx = (pf1 + pf5 + pf8) - (pf3 + pf6 + pf7);
                let vy = (pf2 + pf5 + pf6) - (pf4 + pf7 + pf8);

                // Stability Clamping
                let isShockwave = false;
                if (r < 0.8 || r > 1.2 || r < 0.0001) {
                    const targetRho = Math.max(0.8, Math.min(1.2, r < 0.0001 ? 1.0 : r));
                    const scale = targetRho / r;
                    pf0 *= scale; pf1 *= scale; pf2 *= scale; pf3 *= scale; pf4 *= scale;
                    pf5 *= scale; pf6 *= scale; pf7 *= scale; pf8 *= scale;
                    r = targetRho;
                    isShockwave = true;
                }

                vx /= r; vy /= r;

                const v2 = vx * vx + vy * vy;
                const speed = Math.sqrt(v2);
                if (speed > maxU) maxU = speed;

                let u2_clamped = v2;
                if (speed > this.params.cflLimit) {
                    const scale = this.params.cflLimit / speed;
                    vx *= scale; vy *= scale;
                    u2_clamped = vx * vx + vy * vy;
                    isShockwave = true;
                }

                rho[i] = r; ux[i] = vx; uy[i] = vy;

                const u2_15 = 1.5 * u2_clamped;

                let feq0 = cx_w[0] * r * (1.0 - u2_15);
                let feq1 = cx_w[1] * r * (1.0 + 3.0 * vx + 4.5 * vx * vx - u2_15);
                let feq2 = cx_w[2] * r * (1.0 + 3.0 * vy + 4.5 * vy * vy - u2_15);
                let feq3 = cx_w[3] * r * (1.0 - 3.0 * vx + 4.5 * vx * vx - u2_15);
                let feq4 = cx_w[4] * r * (1.0 - 3.0 * vy + 4.5 * vy * vy - u2_15);

                let cu5 = vx + vy; let feq5 = cx_w[5] * r * (1.0 + 3.0 * cu5 + 4.5 * cu5 * cu5 - u2_15);
                let cu6 = -vx + vy; let feq6 = cx_w[6] * r * (1.0 + 3.0 * cu6 + 4.5 * cu6 * cu6 - u2_15);
                let cu7 = -vx - vy; let feq7 = cx_w[7] * r * (1.0 + 3.0 * cu7 + 4.5 * cu7 * cu7 - u2_15);
                let cu8 = vx - vy; let feq8 = cx_w[8] * r * (1.0 + 3.0 * cu8 + 4.5 * cu8 * cu8 - u2_15);

                if (isShockwave) {
                    out0[i] = feq0; out1[i] = feq1; out2[i] = feq2; out3[i] = feq3; out4[i] = feq4;
                    out5[i] = feq5; out6[i] = feq6; out7[i] = feq7; out8[i] = feq8;
                } else {
                    let fneq1 = pf1 - feq1; let fneq2 = pf2 - feq2;
                    let fneq3 = pf3 - feq3; let fneq4 = pf4 - feq4;
                    let fneq5 = pf5 - feq5; let fneq6 = pf6 - feq6;
                    let fneq7 = pf7 - feq7; let fneq8 = pf8 - feq8;

                    let Pxx = fneq1 + fneq3 + fneq5 + fneq6 + fneq7 + fneq8;
                    let Pyy = fneq2 + fneq4 + fneq5 + fneq6 + fneq7 + fneq8;
                    let Pxy = fneq5 - fneq6 + fneq7 - fneq8;

                    let S_norm = Math.sqrt(2 * (Pxx * Pxx + Pyy * Pyy + 2 * Pxy * Pxy));
                    if (S_norm > 10.0 || isNaN(S_norm)) S_norm = 10.0;
                    let tau_eff = this.params.tau_0 + this.params.smagorinsky * S_norm;
                    if (isNaN(tau_eff) || tau_eff < 0.505) tau_eff = 0.505;
                    else if (tau_eff > 2.0) tau_eff = 2.0;

                    sumTau += tau_eff;
                    sumRho += r;
                    activeCells++;

                    let inv_tau = 1.0 / tau_eff;
                    out0[i] = pf0 - (pf0 - feq0) * inv_tau;
                    out1[i] = pf1 - fneq1 * inv_tau;
                    out2[i] = pf2 - fneq2 * inv_tau;
                    out3[i] = pf3 - fneq3 * inv_tau;
                    out4[i] = pf4 - fneq4 * inv_tau;
                    out5[i] = pf5 - fneq5 * inv_tau;
                    out6[i] = pf6 - fneq6 * inv_tau;
                    out7[i] = pf7 - fneq7 * inv_tau;
                    out8[i] = pf8 - fneq8 * inv_tau;
                }
            }
        }

        // 2. VORTICITY / CURL Calculation (Face 21 - needed for visualization)
        const curl_out = faces[21];
        for (let y = 1; y < ny - 1; y++) {
            for (let x = 1; x < nx - 1; x++) {
                const i = zOff + y * nx + x;
                const xM = x > 1 ? x - 1 : 1;
                const xP = x < nx - 2 ? x + 1 : nx - 2;
                const dxDist = (x === 1 || x === nx - 2) ? 1.0 : 2.0;

                const yM_idx = y > 1 ? y - 1 : 1;
                const yP_idx = y < ny - 2 ? y + 1 : ny - 2;
                const dyDist = (y === 1 || y === ny - 2) ? 1.0 : 2.0;

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
    }

    private stepBio(faces: Float32Array[], nx: number, ny: number, lz: number): void {
        const b_in = faces[23 + this.parity];
        const b_out = faces[23 + (1 - this.parity)];

        const ux = faces[19];
        const uy = faces[20];

        const diff = this.params.bioDiffusion;
        const growth = this.params.bioGrowth;

        const zOff = lz * ny * nx;

        for (let y = 1; y < ny - 1; y++) {
            for (let x = 1; x < nx - 1; x++) {
                const i = zOff + y * nx + x;
                const bo = b_in[i];

                // Laplacian
                const lap = b_in[i - 1] + b_in[i + 1] + b_in[i - nx] + b_in[i + nx] - 4 * bo;

                // Advection semi-lagrangienne
                const ax = Math.max(1, Math.min(nx - 2, x - ux[i] * 0.8));
                const ay = Math.max(1, Math.min(ny - 2, y - uy[i] * 0.8));
                const ix = Math.floor(ax); const iy = Math.floor(ay);
                const fx = ax - ix; const fy = ay - iy;

                const v00 = b_in[zOff + iy * nx + ix];
                const v10 = b_in[zOff + iy * nx + (ix + 1)];
                const v01 = b_in[zOff + (iy + 1) * nx + ix];
                const v11 = b_in[zOff + (iy + 1) * nx + (ix + 1)];

                const advected = (1 - fy) * ((1 - fx) * v00 + fx * v10) + fy * ((1 - fx) * v01 + fx * v11);

                let next = advected + diff * lap + growth * bo * (1 - bo);
                if (next < 0) next = 0;
                if (next > 1) next = 1;
                b_out[i] = next;
            }
        }
    }

    private getWgslSource(): string {
        return `
            struct Uniforms {
                nx: u32, ny: u32, nz: u32, strideFace: u32,
                tau_0: f32, smagorinsky: f32, cflLimit: f32, isClosed: f32,
                bioDiffusion: f32, bioGrowth: f32, parity: u32, pad2: f32, pad3: f32, pad4: f32, pad5: f32, pad6: f32
            };

            @group(0) @binding(0) var<storage, read_write> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;

            const w0 = 0.444444444; const w1 = 0.111111111; const w2 = 0.027777777;

            @compute @workgroup_size(16, 16, 1)
            fn compute_lbm(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                let y = global_id.y;
                let z = global_id.z;

                if (x >= config.nx || y >= config.ny || z >= config.nz) { return; }
                
                let idx = z * config.nx * config.ny + y * config.nx + x;
                let stride = config.strideFace;
                
                // Ping-Pong Buffers
                // parity 0: in=0-8, out=9-17
                // parity 1: in=9-17, out=0-8
                let f_in_base = config.parity * 9u * stride;
                let f_out_base = (1u - config.parity) * 9u * stride;

                if (x == 0u || x == config.nx - 1u || y == 0u || y == config.ny - 1u) { 
                    for (var k = 0u; k < 9u; k = k + 1u) {
                        cube[f_out_base + k * stride + idx] = cube[f_in_base + k * stride + idx];
                    }
                    return; 
                }

                let isClosed = config.isClosed > 0.5;

                // Output Buffers (f_post) => 9..17
                

                let obst = cube[18u * stride + idx];
                if (obst > 0.5) {
                    cube[f_out_base + 0u * stride + idx] = w0;
                    cube[f_out_base + 1u * stride + idx] = w1; cube[f_out_base + 2u * stride + idx] = w1;
                    cube[f_out_base + 3u * stride + idx] = w1; cube[f_out_base + 4u * stride + idx] = w1;
                    cube[f_out_base + 5u * stride + idx] = w2; cube[f_out_base + 6u * stride + idx] = w2;
                    cube[f_out_base + 7u * stride + idx] = w2; cube[f_out_base + 8u * stride + idx] = w2;
                    return;
                }

                // Internal Pull Streaming
                var pf0 = cube[f_in_base + 0u * stride + idx];
                
                var pf1: f32; var pf2: f32; var pf3: f32; var pf4: f32;
                var pf5: f32; var pf6: f32; var pf7: f32; var pf8: f32;

                // 1(+x), opp:3
                let i1 = idx - 1u;
                pf1 = select(cube[f_in_base + 1u * stride + i1], cube[f_in_base + 3u * stride + idx], (isClosed && (x <= 1u)) || cube[18u * stride + i1] > 0.5);
                // 2(+y), opp:4
                let i2 = idx - config.nx;
                pf2 = select(cube[f_in_base + 2u * stride + i2], cube[f_in_base + 4u * stride + idx], (isClosed && (y <= 1u)) || cube[18u * stride + i2] > 0.5);
                // 3(-x), opp:1
                let i3 = idx + 1u;
                pf3 = select(cube[f_in_base + 3u * stride + i3], cube[f_in_base + 1u * stride + idx], (isClosed && (x >= config.nx - 2u)) || cube[18u * stride + i3] > 0.5);
                // 4(-y), opp:2
                let i4 = idx + config.nx;
                pf4 = select(cube[f_in_base + 4u * stride + i4], cube[f_in_base + 2u * stride + idx], (isClosed && (y >= config.ny - 2u)) || cube[18u * stride + i4] > 0.5);
                // 5(+x+y), opp:7
                let i5 = idx - config.nx - 1u;
                pf5 = select(cube[f_in_base + 5u * stride + i5], cube[f_in_base + 7u * stride + idx], (isClosed && (x <= 1u || y <= 1u)) || cube[18u * stride + i5] > 0.5);
                // 6(-x+y), opp:8
                let i6 = idx - config.nx + 1u;
                pf6 = select(cube[f_in_base + 6u * stride + i6], cube[f_in_base + 8u * stride + idx], (isClosed && (x >= config.nx - 2u || y <= 1u)) || cube[18u * stride + i6] > 0.5);
                // 7(-x-y), opp:5
                let i7 = idx + config.nx + 1u;
                pf7 = select(cube[f_in_base + 7u * stride + i7], cube[f_in_base + 5u * stride + idx], (isClosed && (x >= config.nx - 2u || y >= config.ny - 2u)) || cube[18u * stride + i7] > 0.5);
                // 8(+x-y), opp:6
                let i8 = idx + config.nx - 1u;
                pf8 = select(cube[f_in_base + 8u * stride + i8], cube[f_in_base + 6u * stride + idx], (isClosed && (x <= 1u || y >= config.ny - 2u)) || cube[18u * stride + i8] > 0.5);

                var r = pf0 + pf1 + pf2 + pf3 + pf4 + pf5 + pf6 + pf7 + pf8;
                var vx = (pf1 + pf5 + pf8) - (pf3 + pf6 + pf7);
                var vy = (pf2 + pf5 + pf6) - (pf4 + pf7 + pf8);

                var isShockwave = false;
                if (r < 0.8 || r > 1.2 || r < 0.0001) {
                    var targetRho = clamp(r, 0.8, 1.2);
                    if (r < 0.0001) { targetRho = 1.0; }
                    let scale = targetRho / (r + 0.00001);
                    pf0 *= scale; pf1 *= scale; pf2 *= scale; pf3 *= scale; pf4 *= scale;
                    pf5 *= scale; pf6 *= scale; pf7 *= scale; pf8 *= scale;
                    r = targetRho;
                    isShockwave = true;
                }

                vx /= r; vy /= r;

                let v2 = vx * vx + vy * vy;
                let speed = sqrt(v2);
                var u2_clamped = v2;

                if (speed > config.cflLimit) {
                    let scale = config.cflLimit / speed;
                    vx *= scale; vy *= scale;
                    u2_clamped = vx * vx + vy * vy;
                    isShockwave = true;
                }

                cube[22u * stride + idx] = r;
                cube[19u * stride + idx] = vx;
                cube[20u * stride + idx] = vy;

                let u2_15 = 1.5 * u2_clamped;

                let feq0 = w0 * r * (1.0 - u2_15);
                let feq1 = w1 * r * (1.0 + 3.0 * vx + 4.5 * vx * vx - u2_15);
                let feq2 = w1 * r * (1.0 + 3.0 * vy + 4.5 * vy * vy - u2_15);
                let feq3 = w1 * r * (1.0 - 3.0 * vx + 4.5 * vx * vx - u2_15);
                let feq4 = w1 * r * (1.0 - 3.0 * vy + 4.5 * vy * vy - u2_15);

                let cu5 = vx + vy; let feq5 = w2 * r * (1.0 + 3.0 * cu5 + 4.5 * cu5 * cu5 - u2_15);
                let cu6 = -vx + vy; let feq6 = w2 * r * (1.0 + 3.0 * cu6 + 4.5 * cu6 * cu6 - u2_15);
                let cu7 = -vx - vy; let feq7 = w2 * r * (1.0 + 3.0 * cu7 + 4.5 * cu7 * cu7 - u2_15);
                let cu8 = vx - vy; let feq8 = w2 * r * (1.0 + 3.0 * cu8 + 4.5 * cu8 * cu8 - u2_15);

                if (isShockwave) {
                    cube[f_out_base + 0u * stride + idx] = feq0;
                    cube[f_out_base + 1u * stride + idx] = feq1; cube[f_out_base + 2u * stride + idx] = feq2;
                    cube[f_out_base + 3u * stride + idx] = feq3; cube[f_out_base + 4u * stride + idx] = feq4;
                    cube[f_out_base + 5u * stride + idx] = feq5; cube[f_out_base + 6u * stride + idx] = feq6;
                    cube[f_out_base + 7u * stride + idx] = feq7; cube[f_out_base + 8u * stride + idx] = feq8;
                } else {
                    let fneq1 = pf1 - feq1; let fneq2 = pf2 - feq2;
                    let fneq3 = pf3 - feq3; let fneq4 = pf4 - feq4;
                    let fneq5 = pf5 - feq5; let fneq6 = pf6 - feq6;
                    let fneq7 = pf7 - feq7; let fneq8 = pf8 - feq8;

                    let Pxx = fneq1 + fneq3 + fneq5 + fneq6 + fneq7 + fneq8;
                    let Pyy = fneq2 + fneq4 + fneq5 + fneq6 + fneq7 + fneq8;
                    let Pxy = fneq5 - fneq6 + fneq7 - fneq8;

                    var S_norm = sqrt(2.0 * (Pxx * Pxx + Pyy * Pyy + 2.0 * Pxy * Pxy));
                    if (S_norm > 10.0) { S_norm = 10.0; }

                    var tau_eff = config.tau_0 + config.smagorinsky * S_norm;
                    if (tau_eff < 0.505) { tau_eff = 0.505; }
                    if (tau_eff > 2.0) { tau_eff = 2.0; }

                    let inv_tau = 1.0 / tau_eff;
                    cube[f_out_base + 0u * stride + idx] = pf0 - (pf0 - feq0) * inv_tau;
                    cube[f_out_base + 1u * stride + idx] = pf1 - fneq1 * inv_tau;
                    cube[f_out_base + 2u * stride + idx] = pf2 - fneq2 * inv_tau;
                    cube[f_out_base + 3u * stride + idx] = pf3 - fneq3 * inv_tau;
                    cube[f_out_base + 4u * stride + idx] = pf4 - fneq4 * inv_tau;
                    cube[f_out_base + 5u * stride + idx] = pf5 - fneq5 * inv_tau;
                    cube[f_out_base + 6u * stride + idx] = pf6 - fneq6 * inv_tau;
                    cube[f_out_base + 7u * stride + idx] = pf7 - fneq7 * inv_tau;
                    cube[f_out_base + 8u * stride + idx] = pf8 - fneq8 * inv_tau;
                }

                // Curl calculation (vorticity) for rendering (Face 21)
                let xM = max(1u, x - 1u); let xP = min(config.nx - 2u, x + 1u);
                let dxDist = select(2.0, 1.0, x == 1u || x == config.nx - 2u);
                let yM_idx = max(1u, y - 1u); let yP_idx = min(config.ny - 2u, y + 1u);
                let dyDist = select(2.0, 1.0, y == 1u || y == config.ny - 2u);

                let dUy_dx = (cube[20u * stride + z * config.nx * config.ny + y * config.nx + xP] - 
                             cube[20u * stride + z * config.nx * config.ny + y * config.nx + xM]) / dxDist;
                
                let dUx_dy = (cube[19u * stride + z * config.nx * config.ny + yP_idx * config.nx + x] - 
                             cube[19u * stride + z * config.nx * config.ny + yM_idx * config.nx + x]) / dyDist;
                
                cube[21u * stride + idx] = dUy_dx - dUx_dy;
            }

            @compute @workgroup_size(16, 16, 1)
            fn compute_bio(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x; let y = global_id.y; let z = global_id.z;
                if (x >= config.nx || y >= config.ny || z >= config.nz) { return; }
                let idx = z * config.nx * config.ny + y * config.nx + x;
                let stride = config.strideFace;

                let b_in_base = (23u + config.parity) * stride;
                let b_out_base = (23u + (1u - config.parity)) * stride;

                if (x == 0u || x == config.nx - 1u || y == 0u || y == config.ny - 1u) { 
                    cube[b_out_base + idx] = cube[b_in_base + idx];
                    return;
                }

                let bo = cube[b_in_base + idx];
                
                // Laplacian
                let lap = cube[b_in_base + idx - 1u] + cube[b_in_base + idx + 1u] + 
                          cube[b_in_base + idx - config.nx] + cube[b_in_base + idx + config.nx] - 4.0 * bo;

                // Advection semi-lagrangian
                // Face 18 is static obstacle, 19 is ux, 20 is uy. In CPU code above, there is a bug using face 18 for UX. 
                // We fix it properly in WGSL: ux is 19, uy is 20.
                let ux = cube[19u * stride + idx];
                let uy = cube[20u * stride + idx];
                
                let ax = clamp(f32(x) - ux * 0.8, 1.0, f32(config.nx - 2u));
                let ay = clamp(f32(y) - uy * 0.8, 1.0, f32(config.ny - 2u));
                let ix = u32(ax); let iy = u32(ay);
                let fx = ax - f32(ix); let fy = ay - f32(iy);

                let v00 = cube[b_in_base + z * config.nx * config.ny + iy * config.nx + ix];
                let v10 = cube[b_in_base + z * config.nx * config.ny + iy * config.nx + min(ix + 1u, config.nx - 2u)];
                let v01 = cube[b_in_base + z * config.nx * config.ny + min(iy + 1u, config.ny - 2u) * config.nx + ix];
                let v11 = cube[b_in_base + z * config.nx * config.ny + min(iy + 1u, config.ny - 2u) * config.nx + min(ix + 1u, config.nx - 2u)];

                let advected = (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * ((1.0 - fx) * v01 + fx * v11);
                
                var next = advected + config.bioDiffusion * lap + config.bioGrowth * bo * (1.0 - bo);
                next = clamp(next, 0.0, 1.0);
                
                cube[b_out_base + idx] = next; // bio_next
            }

            // copy_faces removed in favor of ping-pong
        `;
    }
}


