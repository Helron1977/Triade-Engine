import type { IHypercubeEngine } from './IHypercubeEngine';
import { HypercubeGPUContext } from '../core/gpu/HypercubeGPUContext';

/**
 * VolumeDiffusionEngine
 * Premier moteur nativement 3D.
 * Simule la diffusion thermique ou de concentration dans un volume [nx, ny, nz]
 * via un stencil à 7 points (Laplacien 3D).
 * 
 * Mapping des Faces :
 * Face 0: State t (Input)
 * Face 1: State t+1 (Output)
 */
export class VolumeDiffusionEngine implements IHypercubeEngine {
    public get name(): string {
        return "Volume Diffusion (3D Stencil)";
    }

    public getRequiredFaces(): number {
        return 2;
    }

    /**
     * @param diffusionRate Coefficient de diffusion (D). Plafonné à 1/6 pour la stabilité.
     * @param dissipation Taux de perte par frame (ex: 1.0).
     * @param boundaryMode 'periodic' ou 'clamped' (Neumann no-flux).
     */
    constructor(
        public diffusionRate: number = 0.1,
        public dissipation: number = 1.0,
        public boundaryMode: 'periodic' | 'clamped' = 'periodic'
    ) {
        // CFL Stability : D * Δt / Δx^2 < 1/6 pour 7-point stencil
        this.diffusionRate = Math.min(diffusionRate, 1 / 6);
    }

    public init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker: boolean = false): void {
        // Initial state is typically injected from outside via specific splats
    }

    /**
     * Simulation CPU du stencil 3D.
     */
    compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        const current = faces[0];
        const next = faces[1];

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;

            // Indices pour Z
            let lzPrev, lzNext;
            if (this.boundaryMode === 'periodic') {
                lzPrev = (lz - 1 + nz) % nz;
                lzNext = (lz + 1) % nz;
            } else {
                lzPrev = Math.max(0, lz - 1);
                lzNext = Math.min(nz - 1, lz + 1);
            }
            const zPrevOff = lzPrev * ny * nx;
            const zNextOff = lzNext * ny * nx;

            for (let ly = 0; ly < ny; ly++) {
                const yOff = ly * nx;

                // Indices pour Y
                let lyPrev, lyNext;
                if (this.boundaryMode === 'periodic') {
                    lyPrev = (ly - 1 + ny) % ny;
                    lyNext = (ly + 1) % ny;
                } else {
                    lyPrev = Math.max(0, ly - 1);
                    lyNext = Math.min(ny - 1, ly + 1);
                }
                const yPrevOff = lyPrev * nx;
                const yNextOff = lyNext * nx;

                for (let lx = 0; lx < nx; lx++) {
                    const idx = zOff + yOff + lx;
                    const val = current[idx];

                    // Indices pour X
                    let lxPrev, lxNext;
                    if (this.boundaryMode === 'periodic') {
                        lxPrev = (lx - 1 + nx) % nx;
                        lxNext = (lx + 1) % nx;
                    } else {
                        lxPrev = Math.max(0, lx - 1);
                        lxNext = Math.min(nx - 1, lx + 1);
                    }

                    // Voisins 6 directions (7-point stencil)
                    const L = current[zOff + yOff + lxPrev];
                    const R = current[zOff + yOff + lxNext];
                    const T = current[zOff + yPrevOff + lx];
                    const B = current[zOff + yNextOff + lx];
                    const F = current[zPrevOff + yOff + lx];
                    const Bk = current[zNextOff + yOff + lx];

                    // Laplacien 3D : Δu ≈ sum(voisins) - 6*u
                    const laplacian = (L + R + T + B + F + Bk) - (6 * val);

                    // u_next = (u + D * Δu) * dissipation
                    next[idx] = (val + this.diffusionRate * laplacian) * this.dissipation;
                }
            }
        }

        // Finalisation : on recopie next dans current pour le prochain tour
        current.set(next);
    }

    private pipelineDiffusion: GPUComputePipeline | null = null;
    private pipelineCopy: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private cubeBuffer: GPUBuffer | null = null;
    public gpuEnabled: boolean = false;

    // Dynamic workgroup sizes
    private wgSizeX: number = 8;
    private wgSizeY: number = 8;
    private wgSizeZ: number = 4;
    private wgSizeCopy: number = 256;

    public initGPU(device: GPUDevice, readBuffer: GPUBuffer, writeBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void {
        // Dynamically scale workgroup sizes to match hardware limits (default 256, often 512 or 1024)
        const maxInvoc = device.limits.maxComputeInvocationsPerWorkgroup || 256;
        if (maxInvoc >= 1024) {
            this.wgSizeZ = 16;
            this.wgSizeCopy = 1024;
        } else if (maxInvoc >= 512) {
            this.wgSizeZ = 8;
            this.wgSizeCopy = 512;
        } else {
            this.wgSizeZ = 4;
            this.wgSizeCopy = 256;
        }

        const shaderCode = this.getWgslSource();
        const shaderModule = device.createShaderModule({ code: shaderCode });

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

        this.pipelineDiffusion = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_diffusion' }
        });

        this.pipelineCopy = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'copy_face' }
        });

        // stride est en bytes -> strideFace en floats
        const strideFace = (nx * ny * nz);

        const uniformData = new ArrayBuffer(32);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx;
        u32[1] = ny;
        u32[2] = nz;
        f32[3] = this.diffusionRate;
        f32[4] = this.dissipation;
        u32[5] = strideFace;
        u32[6] = this.boundaryMode === 'periodic' ? 1 : 0;
        u32[7] = 0; // Padding

        this.uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        this.gpuEnabled = true;
    }

    public computeGPU(device: GPUDevice, commandEncoder: GPUCommandEncoder, nx: number, ny: number, nz: number, readBuffer: GPUBuffer, writeBuffer: GPUBuffer): void {
        if (!this.pipelineDiffusion || !this.pipelineCopy || !this.uniformBuffer) return;

        const uniformData = new ArrayBuffer(32);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        const strideFace = (nx * ny * nz);

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = strideFace;
        f32[3] = this.diffusionRate; f32[4] = this.dissipation;
        u32[5] = strideFace; u32[6] = this.boundaryMode === 'periodic' ? 1 : 0;
        u32[7] = 0;

        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const bindGroup = device.createBindGroup({
            layout: this.pipelineDiffusion.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: readBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        // Dispatch diffusion compute shader
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.setPipeline(this.pipelineDiffusion);

        passEncoder.dispatchWorkgroups(
            Math.ceil(nx / this.wgSizeX),
            Math.ceil(ny / this.wgSizeY),
            Math.ceil(nz / this.wgSizeZ)
        );

        // Copy Face 1 (Output) back to Face 0 (Input) natively in VRAM 
        const totalElements = nx * ny * nz;
        passEncoder.setPipeline(this.pipelineCopy);
        passEncoder.dispatchWorkgroups(Math.ceil(totalElements / this.wgSizeCopy));

        passEncoder.end();
    }

    private getWgslSource(): string {
        return `
            struct Uniforms {
                nx: u32,
                ny: u32,
                nz: u32,
                diffusionRate: f32,
                dissipation: f32,
                strideFace: u32,
                periodic: u32,
                padding: u32,
            };

            @group(0) @binding(0) var<storage, read_write> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;

            @compute @workgroup_size(${this.wgSizeX}, ${this.wgSizeY}, ${this.wgSizeZ})
            fn compute_diffusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                let y = global_id.y;
                let z = global_id.z;

                if (x >= config.nx || y >= config.ny || z >= config.nz) { return; }

                // Corrected index calculation for 3D array
                let idx = z * config.nx * config.ny + y * config.nx + x;

                let val = cube[idx];

                var lap = 0.0;

                // X
                let lx = select(x - 1u, config.nx - 1u, x == 0u && config.periodic == 1u);
                lap += cube[z * config.nx * config.ny + y * config.nx + lx];

                let rx = select(x + 1u, 0u, x == config.nx - 1u && config.periodic == 1u);
                lap += cube[z * config.nx * config.ny + y * config.nx + rx];

                // Y
                let ly = select(y - 1u, config.ny - 1u, y == 0u && config.periodic == 1u);
                lap += cube[z * config.nx * config.ny + ly * config.nx + x];

                let ry = select(y + 1u, 0u, y == config.ny - 1u && config.periodic == 1u);
                lap += cube[z * config.nx * config.ny + ry * config.nx + x];

                // Z
                let lz = select(z - 1u, config.nz - 1u, z == 0u && config.periodic == 1u);
                lap += cube[lz * config.nx * config.ny + y * config.nx + x];

                let rz = select(z + 1u, 0u, z == config.nz - 1u && config.periodic == 1u);
                lap += cube[rz * config.nx * config.ny + y * config.nx + x];

                lap -= 6.0 * val;

                let nval = (val + config.diffusionRate * lap) * config.dissipation;
                cube[config.strideFace + idx] = clamp(nval, 0.0, 1.0);
            }

            @compute @workgroup_size(${this.wgSizeCopy})
            fn copy_face(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let total_size = config.nx * config.ny * config.nz;
                
                if (idx >= total_size) { return; }
                
                // Copy Output (Face 1) back to Input (Face 0)
                cube[idx] = cube[config.strideFace + idx];
            }
        `;
    }
}
