import type { IHypercubeEngine } from './IHypercubeEngine';

/**
 * GrayScottEngine
 * Simule des motifs de Turing (Réaction-Diffusion) haute performance.
 * Équations :
 * dA/dt = Da * lap(A) - AB^2 + f(1-A)
 * dB/dt = Db * lap(B) + AB^2 - (f+k)B
 */
export class GrayScottEngine implements IHypercubeEngine {
    public get name(): string {
        return "GrayScottEngine";
    }

    public getSyncFaces(): number[] {
        return [0, 1]; // Sync Substance A and B across chunk boundaries
    }

    public getRequiredFaces(): number {
        return 4; // A, B, A_next, B_next
    }

    constructor(
        public Da: number = 0.2,
        public Db: number = 0.1,
        public feed: number = 0.035,
        public kill: number = 0.06
    ) { }

    public init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker: boolean = false): void {
        if (isWorker) return;
        // Initial state: Fill A with 1.0, B with 0.0
        faces[0].fill(1.0);
        faces[1].fill(0.0);
    }

    private pipelineGS: GPUComputePipeline | null = null;
    private pipelineCopy: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private lastStride: number = 0;

    public initGPU(device: GPUDevice, cubeBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void {
        this.lastStride = stride / 4;
        const wgSizeX = 16;
        const wgSizeY = 16;
        const shaderModule = device.createShaderModule({ code: this.getWgslSource() });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        this.pipelineGS = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_gs' }
        });

        this.pipelineCopy = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'copy_faces' }
        });

        const uniformData = new ArrayBuffer(32);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = this.lastStride;
        f32[4] = this.Da; f32[5] = this.Db; f32[6] = this.feed; f32[7] = this.kill;

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

    public gpuEnabled: boolean = false;

    public computeGPU(device: GPUDevice, commandEncoder: GPUCommandEncoder, nx: number, ny: number, nz: number): void {
        if (!this.bindGroup || !this.pipelineGS || !this.pipelineCopy || !this.uniformBuffer) return;

        const uniformData = new ArrayBuffer(32);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = this.lastStride; // strideFace
        f32[4] = this.Da; f32[5] = this.Db; f32[6] = this.feed; f32[7] = this.kill;
        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setBindGroup(0, this.bindGroup);

        passEncoder.setPipeline(this.pipelineGS);
        passEncoder.dispatchWorkgroups(Math.ceil(nx / 16), Math.ceil(ny / 16), nz || 1);

        passEncoder.setPipeline(this.pipelineCopy);
        passEncoder.dispatchWorkgroups(Math.ceil((nx * ny * nz) / 256));

        passEncoder.end();
    }

    public compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        const A = faces[0];
        const B = faces[1];
        const Anext = faces[2];
        const Bnext = faces[3];

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;
            for (let ly = 1; ly < ny - 1; ly++) {
                const yOff = ly * nx;
                for (let lx = 1; lx < nx - 1; lx++) {
                    const idx = zOff + yOff + lx;

                    const a = A[idx];
                    const b = B[idx];

                    // 5-point laplacian (2D slice focus for performance & patterns)
                    const lapA = (A[idx - 1] + A[idx + 1] + A[idx - nx] + A[idx + nx] - 4 * a);
                    const lapB = (B[idx - 1] + B[idx + 1] + B[idx - nx] + B[idx + nx] - 4 * b);

                    const react = a * b * b;

                    Anext[idx] = a + (this.Da * lapA - react + this.feed * (1 - a));
                    Bnext[idx] = b + (this.Db * lapB + react - (this.feed + this.kill) * b);
                }
            }
        }

        A.set(Anext);
        B.set(Bnext);
    }

    private getWgslSource(): string {
        return `
            struct Uniforms {
                nx: u32, ny: u32, nz: u32, strideFace: u32,
                Da: f32, Db: f32, feed: f32, kill: f32
            };

            @group(0) @binding(0) var<storage, read_write> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;

            @compute @workgroup_size(16, 16, 1)
            fn compute_gs(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                let y = global_id.y;
                let z = global_id.z;

                if (x >= config.nx || y >= config.ny || z >= config.nz) { return; }
                let idx = z * config.nx * config.ny + y * config.nx + x;

                let stride = config.strideFace;
                let face_A = 0u;
                let face_B = stride * 1u;
                let face_Anext = stride * 2u;
                let face_Bnext = stride * 3u;

                let a = cube[face_A + idx];
                let b = cube[face_B + idx];

                if (x == 0u || x == config.nx - 1u || y == 0u || y == config.ny - 1u) {
                    cube[face_Anext + idx] = a;
                    cube[face_Bnext + idx] = b;
                    return;
                }

                let left = idx - 1u;
                let right = idx + 1u;
                let top = idx - config.nx;
                let bottom = idx + config.nx;

                let lapA = (cube[face_A + left] + cube[face_A + right] + cube[face_A + top] + cube[face_A + bottom] - 4.0 * a);
                let lapB = (cube[face_B + left] + cube[face_B + right] + cube[face_B + top] + cube[face_B + bottom] - 4.0 * b);

                let react = a * b * b;

                cube[face_Anext + idx] = a + (config.Da * lapA - react + config.feed * (1.0 - a));
                cube[face_Bnext + idx] = b + (config.Db * lapB + react - (config.feed + config.kill) * b);
            }

            @compute @workgroup_size(256)
            fn copy_faces(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let total_size = config.nx * config.ny * config.nz;
                if (idx >= total_size) { return; }
                
                let stride = config.strideFace;
                // A = Anext
                cube[idx] = cube[stride * 2u + idx];
                // B = Bnext
                cube[stride * 1u + idx] = cube[stride * 3u + idx];
            }
        `;
    }
}
