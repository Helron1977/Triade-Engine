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
 * OceanEngine V5.4 – Version GPU-first avec double buffering correct
 */
export class OceanEngine implements IHypercubeEngine {
    public get name(): string { return "OceanEngine 2.5D"; }

    public getRequiredFaces(): number { return 25; }

    // ── GPU REFACTO V5.4 ── La Grid contrôle totalement la parité
    public parity: number = 0;

    public params: OceanEngineParams = {
        tau_0: 0.8,
        smagorinsky: 0.2,
        cflLimit: 0.38,
        bioDiffusion: 0.05,
        bioGrowth: 0.0005,
        closedBounds: false
    };

    public stats = { maxU: 0, avgTau: 0, avgRho: 0 };

    // GPU cached resources
    private pipelineLBM: GPUComputePipeline | null = null;
    private pipelineBio: GPUComputePipeline | null = null;
    private uniformBuffer: GPUBuffer | null = null;

    private bindGroup0: GPUBindGroup | null = null;  // Read 0  Write 1
    private bindGroup1: GPUBindGroup | null = null;  // Read 1  Write 0

    private lastNx = 0;
    private lastNy = 0;
    private stride: number = 0; // ── GPU REFACTO V5.4 ── Stocké pour computeGPU

    public getConfig(): Record<string, any> {
        return { ...this.params, parity: this.parity };
    }

    public applyConfig(config: any): void {
        Object.assign(this.params, config);
        if (typeof config.parity === 'number') this.parity = config.parity;
    }

    // ── GPU REFACTO V5.4 ──
    public initGPU(
        device: GPUDevice,
        readBuffer: GPUBuffer,
        writeBuffer: GPUBuffer,
        stride: number,
        nx: number,
        ny: number,
        nz: number
    ): void {
        this.lastNx = nx;
        this.lastNy = ny;
        this.stride = stride;

        const shaderModule = device.createShaderModule({
            code: this.getWgslSource(),
            label: 'OceanEngine Shader'
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        this.pipelineLBM = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: 'compute_lbm' }
        });

        this.pipelineBio = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: 'compute_bio' }
        });

        this.uniformBuffer = device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // BindGroups pré-créés une seule fois
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
    }

    // ── GPU REFACTO V5.4 ── Version finale propre
    public computeGPU(
        device: GPUDevice,
        commandEncoder: GPUCommandEncoder,
        nx: number,
        ny: number,
        nz: number,
        readBuffer: GPUBuffer,
        writeBuffer: GPUBuffer
    ): void {
        if (!this.pipelineLBM || !this.pipelineBio || !this.uniformBuffer) return;

        // ── GPU REFACTO V5.4 ── Mise à jour des uniformes (Respect strict de l'alignement WGSL)
        const u = new ArrayBuffer(64);
        const f32 = new Float32Array(u);
        const u32 = new Uint32Array(u);

        u32[0] = nx; u32[1] = ny; u32[2] = nz;
        u32[3] = this.stride / 4;           // strideFace
        f32[4] = this.params.tau_0; f32[5] = this.params.smagorinsky; f32[6] = this.params.cflLimit;
        f32[7] = this.params.closedBounds ? 1.0 : 0.0;
        f32[8] = this.params.bioDiffusion; f32[9] = this.params.bioGrowth;
        u32[10] = this.parity;                // ── CRITIQUE ── On écrit bien un u32 ici !

        device.queue.writeBuffer(this.uniformBuffer, 0, u, 0, 64);

        const bindGroup = (this.parity === 0) ? this.bindGroup0! : this.bindGroup1!;

        const wx = Math.ceil(nx / 16);
        const wy = Math.ceil(ny / 16);

        const pass = commandEncoder.beginComputePass({ label: 'OceanEngine LBM Pass' });
        pass.setPipeline(this.pipelineLBM);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(wx, wy, nz || 1);
        pass.end();

        const passBio = commandEncoder.beginComputePass({ label: 'OceanEngine Bio Pass' });
        passBio.setPipeline(this.pipelineBio);
        passBio.setBindGroup(0, bindGroup);
        passBio.dispatchWorkgroups(wx, wy, nz || 1);
        passBio.end();
    }

    // ── WGSL corrigé et simplifié ──
    private getWgslSource(): string {
        return `
            struct Uniforms {
    nx: u32, ny: u32, nz: u32, strideFace: u32,
        tau_0: f32, smagorinsky: f32, cflLimit: f32, isClosed: f32,
            bioDiffusion: f32, bioGrowth: f32, parity: u32
};

@group(0) @binding(0) var<storage, read > cube_in: array<f32>;
@group(0) @binding(1) var<storage, read_write > cube_out: array<f32>;
@group(0) @binding(2) var<uniform>u: Uniforms;

const cx = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const cy = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const w = array<f32, 9>(0.444444, 0.111111, 0.111111, 0.111111, 0.111111, 0.027778, 0.027778, 0.027778, 0.027778);
const opp = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);

@compute @workgroup_size(16, 16, 1)
            fn compute_lbm(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x); let y = i32(id.y); let z = i32(id.z);
    if (x < 1 || y < 1 || x >= i32(u.nx) - 1 || y >= i32(u.ny) - 1) { return; }

    let idx = u32(z) * u.nx * u.ny + u32(y) * u.nx + u32(x);
    let stride = u.strideFace;
    let f_in = u.parity * 9u * stride;
    let f_out = (1u - u.parity) * 9u * stride;

    let obst = cube_in[18u * stride + idx];
    cube_out[18u * stride + idx] = obst; // ── GPU REFACTO V5.4 ── On préserve l'obstacle dans le buffer de sortie

    if (obst > 0.5) {
        for (var k = 0u; k < 9u; k = k + 1u) {
            cube_out[f_out + k * stride + idx] = w[k];
        }
        return;
    }

    var pf: array<f32, 9>;
    pf[0] = cube_in[f_in + 0u * stride + idx];

    for (var k = 1u; k < 9u; k = k + 1u) {
        var sx = x - cx[k];
        var sy = y - cy[k];
        if (sx < 0 || sx >= i32(u.nx) || sy < 0 || sy >= i32(u.ny)) {
            pf[k] = cube_in[f_in + opp[k] * stride + idx];
        } else {
            let n_idx = u32(z) * u.nx * u.ny + u32(sy) * u.nx + u32(sx);
            if (cube_in[18u * stride + n_idx] > 0.5) {
                pf[k] = cube_in[f_in + opp[k] * stride + idx];
            } else {
                pf[k] = cube_in[f_in + k * stride + n_idx];
            }
        }
    }

    var r = 0.0;
    for (var k = 0u; k < 9u; k = k + 1u) { r += pf[k]; }
    if (r < 0.1 || r > 10.0) { r = 1.0; }

    var vx = (pf[1] + pf[5] + pf[8] - (pf[3] + pf[6] + pf[7])) / (r + 1e-6);
    var vy = (pf[2] + pf[5] + pf[6] - (pf[4] + pf[7] + pf[8])) / (r + 1e-6);

    let v_mag = sqrt(vx * vx + vy * vy);
    if (v_mag > u.cflLimit) {
        vx *= u.cflLimit / v_mag;
        vy *= u.cflLimit / v_mag;
    }

    cube_out[22u * stride + idx] = r;
    cube_out[19u * stride + idx] = vx;
    cube_out[20u * stride + idx] = vy;

    let u2_15 = 1.5 * (vx * vx + vy * vy);
    let inv_tau = 1.0 / (u.tau_0 + 1e-5);

    for (var k = 0u; k < 9u; k = k + 1u) {
        let cu = 3.0 * (f32(cx[k]) * vx + f32(cy[k]) * vy);
        let feq = w[k] * r * (1.0 + cu + 0.5 * cu * cu - u2_15);
        cube_out[f_out + k * stride + idx] = pf[k] - (pf[k] - feq) * inv_tau;
    }

    // Curl
    let xP = u32(x + 1); let xM = u32(x - 1);
    let yP = u32(y + 1); let yM = u32(y - 1);
    let dUy_dx = (cube_in[20u * stride + u32(z) * u.nx * u.ny + u32(y) * u.nx + xP]- cube_in[20u * stride + u32(z) * u.nx * u.ny + u32(y) * u.nx + xM]) * 0.5;
    let dUx_dy = (cube_in[19u * stride + u32(z) * u.nx * u.ny + yP * u.nx + u32(x)]- cube_in[19u * stride + u32(z) * u.nx * u.ny + yM * u.nx + u32(x)]) * 0.5;
    cube_out[21u * stride + idx] = dUy_dx - dUx_dy;
}

@compute @workgroup_size(16, 16, 1)
            fn compute_bio(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x); let y = i32(id.y); let z = i32(id.z);
    if (x < 1 || y < 1 || x >= i32(u.nx) - 1 || y >= i32(u.ny) - 1) { return; }

    let idx = u32(z) * u.nx * u.ny + u32(y) * u.nx + u32(x);
    let stride = u.strideFace;
    let b_in = (23u + u.parity) * stride;
    let b_out = (23u + (1u - u.parity)) * stride;

    let bo = cube_in[b_in + idx];
    let lap = cube_in[b_in + idx - 1u] + cube_in[b_in + idx + 1u]+ cube_in[b_in + idx - u.nx] + cube_in[b_in + idx + u.nx] - 4.0 * bo;

    let ux = cube_in[19u* stride + idx];
    let uy = cube_in[20u* stride + idx];

    let ax = clamp(f32(x) - ux * 0.8, 1.0, f32(u.nx) - 2.0);
    let ay = clamp(f32(y) - uy * 0.8, 1.0, f32(u.ny) - 2.0);
    let ix = u32(ax); let iy = u32(ay);
    let fx = ax - f32(ix); let fy = ay - f32(iy);

    let base = u32(z) * u.nx * u.ny;
    let v00 = cube_in[b_in + base + iy * u.nx + ix];
    let v10 = cube_in[b_in + base + iy * u.nx + ix + 1u];
    let v01 = cube_in[b_in + base + (iy + 1u)* u.nx + ix];
    let v11 = cube_in[b_in + base + (iy + 1u)* u.nx + ix + 1u];

    let adv = (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * ((1.0 - fx) * v01 + fx * v11);
    cube_out[b_out + idx] = clamp(adv + u.bioDiffusion * lap + u.bioGrowth * bo * (1.0 - bo), 0.0, 1.0);
}
`;
    }

    // Méthodes CPU restent inchangées 
    public init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker: boolean = false): void {
        const u0 = 0.0; const v0 = 0.0; const rho0 = 1.0;
        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
        const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
        const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;
            for (let i = 0; i < nx * ny; i++) {
                const idx = zOff + i;
                faces[22][idx] = rho0; faces[19][idx] = u0; faces[20][idx] = v0; faces[23][idx] = 0.01;
                for (let k = 0; k < 9; k++) {
                    const cu = 3 * (cx[k] * u0 + cy[k] * v0);
                    const feq = w[k] * rho0 * (1 + cu + 0.5 * cu * cu - 1.5 * (u0 * u0 + v0 * v0));
                    faces[k][idx] = feq;
                    faces[k + 9][idx] = feq;
                }
            }
        }
    }

    public compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        let maxU = 0;
        let totalRho = 0;
        const totalSize = nx * ny * nz;

        for (let lz = 0; lz < nz; lz++) {
            this.stepLBM(faces, nx, ny, lz);
            this.stepBio(faces, nx, ny, lz);
        }

        // ── GPU REFACTO V5.4 ── Mise à jour des stats pour les tests et le HUD
        const rho = faces[22];
        const ux = faces[19];
        const uy = faces[20];
        for (let i = 0; i < totalSize; i++) {
            totalRho += rho[i];
            const u2 = ux[i] * ux[i] + uy[i] * uy[i];
            if (u2 > maxU) maxU = u2;
        }
        this.stats.avgRho = totalRho / totalSize;
        this.stats.maxU = Math.sqrt(maxU);
    }

    private stepLBM(faces: Float32Array[], nx: number, ny: number, lz: number): void {
        const parity = this.parity;
        const f_in = parity === 0 ? [0, 1, 2, 3, 4, 5, 6, 7, 8] : [9, 10, 11, 12, 13, 14, 15, 16, 17];
        const f_out = parity === 0 ? [9, 10, 11, 12, 13, 14, 15, 16, 17] : [0, 1, 2, 3, 4, 5, 6, 7, 8];
        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
        const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
        const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
        const opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];

        const zOff = lz * ny * nx;
        const invTau = 1.0 / (this.params.tau_0 + 1e-6);

        // Pre-allocate for performance (avoid 16k allocations per frame)
        const pf = new Float32Array(9);

        for (let y = 1; y < ny - 1; y++) {
            for (let x = 1; x < nx - 1; x++) {
                const i = zOff + y * nx + x;
                if (faces[18][i] > 0.5) {
                    for (let k = 0; k < 9; k++) faces[f_out[k]][i] = w[k];
                    continue;
                }

                pf[0] = faces[f_in[0]][i];
                for (let k = 1; k < 9; k++) {
                    const sx = x - cx[k]; const sy = y - cy[k];
                    if (this.params.closedBounds && (sx < 0 || sx >= nx || sy < 0 || sy >= ny)) {
                        pf[k] = faces[f_in[opp[k]]][i];
                    } else {
                        const ni = zOff + ((sy + ny) % ny) * nx + ((sx + nx) % nx);
                        if (faces[18][ni] > 0.5) pf[k] = faces[f_in[opp[k]]][i];
                        else pf[k] = faces[f_in[k]][ni];
                    }
                }
                let r = 0; for (let k = 0; k < 9; k++) r += pf[k];
                if (isNaN(r) || r < 0.1 || r > 10.0) { r = 1.0; for (let k = 0; k < 9; k++) pf[k] = w[k]; }

                let vx = (pf[1] + pf[5] + pf[8] - (pf[3] + pf[6] + pf[7])) / r;
                let vy = (pf[2] + pf[5] + pf[6] - (pf[4] + pf[7] + pf[8])) / r;
                const vMag = Math.sqrt(vx * vx + vy * vy);
                if (vMag > this.params.cflLimit) { vx *= (this.params.cflLimit / vMag); vy *= (this.params.cflLimit / vMag); }

                faces[22][i] = r; faces[19][i] = vx; faces[20][i] = vy;
                const u2 = vx * vx + vy * vy;
                for (let k = 0; k < 9; k++) {
                    const cu = 3 * (cx[k] * vx + cy[k] * vy);
                    const feq = w[k] * r * (1 + cu + 0.5 * cu * cu - 1.5 * u2);
                    faces[f_out[k]][i] = pf[k] - (pf[k] - feq) * invTau;
                }
            }
        }
    }

    private stepBio(faces: Float32Array[], nx: number, ny: number, lz: number): void {
        const b_in = faces[this.parity === 0 ? 23 : 24];
        const b_out = faces[this.parity === 0 ? 24 : 23];
        const ux = faces[19], uy = faces[20];
        const zOff = lz * ny * nx;
        for (let y = 1; y < ny - 1; y++) {
            for (let x = 1; x < nx - 1; x++) {
                const i = zOff + y * nx + x;
                const lap = b_in[i - 1] + b_in[i + 1] + b_in[i - nx] + b_in[i + nx] - 4 * b_in[i];
                const ax = Math.max(1, Math.min(nx - 2, x - ux[i] * 0.8));
                const ay = Math.max(1, Math.min(ny - 2, y - uy[i] * 0.8));
                const ix = Math.floor(ax), iy = Math.floor(ay);
                const fx = ax - ix, fy = ay - iy;
                const v00 = b_in[zOff + iy * nx + ix], v10 = b_in[zOff + iy * nx + ix + 1];
                const v01 = b_in[zOff + (iy + 1) * nx + ix], v11 = b_in[zOff + (iy + 1) * nx + ix + 1];
                const adv = (1 - fy) * ((1 - fx) * v00 + fx * v10) + fy * ((1 - fx) * v01 + fx * v11);
                b_out[i] = Math.max(0, Math.min(1, adv + this.params.bioDiffusion * lap + this.params.bioGrowth * b_in[i] * (1 - b_in[i])));
            }
        }
    }

    // ── GPU REFACTO V5.4 ── Requis pour applyEquilibrium
    public getEquilibrium(rho: number, ux: number, uy: number): Float32Array {
        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
        const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
        const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
        const f = new Float32Array(9);
        const u2 = ux * ux + uy * uy;
        for (let k = 0; k < 9; k++) {
            const cu = 3 * (cx[k] * ux + cy[k] * uy);
            f[k] = w[k] * rho * (1 + cu + 0.5 * cu * cu - 1.5 * u2);
        }
        return f;
    }

    public getSyncFaces(): number[] {
        const pops = this.parity === 0 ? [9, 10, 11, 12, 13, 14, 15, 16, 17] : [0, 1, 2, 3, 4, 5, 6, 7, 8];
        const bio = this.parity === 0 ? [24] : [23];
        return [...pops, ...bio, 19, 20, 22];
    }
}
