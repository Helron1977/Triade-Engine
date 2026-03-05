import { IHypercubeEngine } from "./IHypercubeEngine";

export class HeatDiffusionEngine3D implements IHypercubeEngine {
    private alpha: number = 0.1; // Diffusion rate

    get name(): string {
        return "HeatDiffusionEngine3D";
    }

    getRequiredFaces(): number {
        return 2; // Face 0: Current temp, Face 1: Next temp
    }

    getConfig(): any {
        return { alpha: this.alpha };
    }

    init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker?: boolean): void {
        if (isWorker) return; // Skip clear if worker (SharedArrayBuffer already initialized by main)

        // Clear buffers
        for (const face of faces) {
            face.fill(0);
        }
    }

    applyConfig(config: any): void {
        if (config.alpha !== undefined) this.alpha = config.alpha;
    }

    private pipelineHD: GPUComputePipeline | null = null;
    private pipelineCopy: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private lastStride: number = 0;

    public initGPU(device: GPUDevice, cubeBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void {
        this.lastStride = stride / 4;
        const shaderModule = device.createShaderModule({ code: this.getWgslSource() });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        this.pipelineHD = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_heat' }
        });

        this.pipelineCopy = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'copy_faces' }
        });

        const uniformSize = 8 * 4; // 8 floats/uints
        const uniformData = new ArrayBuffer(uniformSize);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = nx * ny * nz; // strideFace
        f32[4] = this.alpha;

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
        if (!this.bindGroup || !this.pipelineHD || !this.pipelineCopy || !this.uniformBuffer) return;

        const uniformSize = 8 * 4;
        const uniformData = new ArrayBuffer(uniformSize);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = this.lastStride;
        f32[4] = this.alpha;
        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setBindGroup(0, this.bindGroup);

        passEncoder.setPipeline(this.pipelineHD);
        passEncoder.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 8));

        passEncoder.setPipeline(this.pipelineCopy);
        passEncoder.dispatchWorkgroups(Math.ceil((nx * ny * nz) / 256));

        passEncoder.end();
    }

    compute(
        faces: Float32Array[],
        nx: number,
        ny: number,
        nz: number,
        chunkX?: number,
        chunkY?: number,
        chunkZ?: number
    ): void {
        const temp_in = faces[0];
        const temp_out = faces[1];
        const obstacles = faces.length > 2 ? faces[2] : null;

        // Laplacian 3D
        for (let z = 1; z < nz - 1; z++) {
            const zOff = z * ny * nx;
            const zOffP = (z + 1) * ny * nx;
            const zOffM = (z - 1) * ny * nx;

            for (let y = 1; y < ny - 1; y++) {
                const yOff = y * nx;
                const yOffP = (y + 1) * nx;
                const yOffM = (y - 1) * nx;

                for (let x = 1; x < nx - 1; x++) {
                    const idx = zOff + yOff + x;

                    if (obstacles && obstacles[idx] > 0) {
                        temp_out[idx] = 0;
                        continue;
                    }

                    const val = temp_in[idx];
                    const laplacian = (
                        temp_in[idx - 1] + temp_in[idx + 1] + // Left / Right
                        temp_in[zOff + yOffM + x] + temp_in[zOff + yOffP + x] + // Top / Bottom
                        temp_in[zOffM + yOff + x] + temp_in[zOffP + yOff + x]   // Front / Back
                    ) - 6 * val;

                    temp_out[idx] = val + this.alpha * laplacian;
                }
            }
        }

        // DANGEROUS: face.set() overwrites boundaries (ghost cells).
        // Only copy back the "useful" part to avoid zeroing out sync data.
        for (let z = 1; z < nz - 1; z++) {
            const zOff = z * ny * nx;
            for (let y = 1; y < ny - 1; y++) {
                const yOff = zOff + y * nx;
                const start = yOff + 1;
                const end = yOff + nx - 1;
                temp_in.set(temp_out.subarray(start, end), start);
            }
        }
    }

    private getWgslSource(): string {
        return `
            struct Uniforms {
                nx: u32, ny: u32, nz: u32, strideFace: u32,
                alpha: f32, pad1: f32, pad2: f32, pad3: f32
            };

            @group(0) @binding(0) var<storage, read_write> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;

            @compute @workgroup_size(8, 8, 8)
            fn compute_heat(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                let y = global_id.y;
                let z = global_id.z;

                if (x >= config.nx || y >= config.ny || z >= config.nz) { return; }

                let idx = z * config.nx * config.ny + y * config.nx + x;
                let stride = config.strideFace;

                if (x == 0u || x == config.nx - 1u || y == 0u || y == config.ny - 1u || z == 0u || z == config.nz - 1u) {
                    cube[1u * stride + idx] = cube[0u * stride + idx];
                    return;
                }

                // Obstacles handling (optional, face 2)
                // In WGSL we might not know if obstacles are allocated, but we assume face 2 is 0.0 if not.
                let obst = cube[2u * stride + idx];
                if (obst > 0.5) {
                    cube[1u * stride + idx] = 0.0;
                    return;
                }

                let val = cube[0u * stride + idx];
                
                let nx = config.nx;
                let ny = config.ny;
                
                let idxM_X = idx - 1u;
                let idxP_X = idx + 1u;
                let idxM_Y = idx - nx;
                let idxP_Y = idx + nx;
                let idxM_Z = idx - (nx * ny);
                let idxP_Z = idx + (nx * ny);

                let laplacian = (
                    cube[0u * stride + idxM_X] + cube[0u * stride + idxP_X] +
                    cube[0u * stride + idxM_Y] + cube[0u * stride + idxP_Y] +
                    cube[0u * stride + idxM_Z] + cube[0u * stride + idxP_Z]
                ) - 6.0 * val;

                cube[1u * stride + idx] = val + config.alpha * laplacian;
            }

            @compute @workgroup_size(256)
            fn copy_faces(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx >= config.nx * config.ny * config.nz) { return; }
                let stride = config.strideFace;

                cube[0u * stride + idx] = cube[1u * stride + idx];
            }
        `;
    }
}
