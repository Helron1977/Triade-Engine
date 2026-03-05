import { IHypercubeEngine } from "./IHypercubeEngine";

export class HeatDiffusionEngine3D implements IHypercubeEngine {
    private alpha: number = 0.1; // Diffusion rate

    get name(): string {
        return "HeatDiffusionEngine3D";
    }

    getRequiredFaces(): number {
        return 2; // Face 0: Current temp, Face 1: Next temp
    }

    public getSyncFaces(): number[] {
        // Output face changes with parity
        return this.parity === 1 ? [1] : [0];
    }

    getConfig(): any {
        return {
            alpha: this.alpha,
            parity: this.parity
        };
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
        if (config.parity !== undefined) this.parity = config.parity;
    }

    private pipelineHD: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private lastStride: number = 0;
    private parity: number = 0;
    public gpuEnabled: boolean = false;

    public initGPU(device: GPUDevice, readBuffer: GPUBuffer, writeBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void {
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

        const uniformSize = 8 * 4;
        const uniformData = new ArrayBuffer(uniformSize);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = this.lastStride;
        f32[4] = this.alpha;
        u32[5] = this.parity;

        this.uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        this.gpuEnabled = true;
    }

    public computeGPU(device: GPUDevice, commandEncoder: GPUCommandEncoder, nx: number, ny: number, nz: number, readBuffer: GPUBuffer, writeBuffer: GPUBuffer): void {
        if (!this.pipelineHD || !this.uniformBuffer) return;

        const uniformSize = 8 * 4;
        const uniformData = new ArrayBuffer(uniformSize);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = this.lastStride;
        f32[4] = this.alpha;
        u32[5] = this.parity;
        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const bindGroup = device.createBindGroup({
            layout: this.pipelineHD.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: readBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.setPipeline(this.pipelineHD);
        passEncoder.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 8));
        passEncoder.end();

        this.parity = 1 - this.parity;
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
        const temp_in = faces[this.parity];
        const temp_out = faces[1 - this.parity];
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

        this.parity = 1 - this.parity;
    }
    private getWgslSource(): string {
        return `
            struct Uniforms {
                nx: u32, ny: u32, nz: u32, strideFace: u32,
                alpha: f32, parity: u32, pad2: f32, pad3: f32
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
                let face_in = config.parity * stride;
                let face_out = (1u - config.parity) * stride;

                if (x == 0u || x == config.nx - 1u || y == 0u || y == config.ny - 1u || z == 0u || z == config.nz - 1u) {
                    cube[face_out + idx] = cube[face_in + idx];
                    return;
                }

                // Obstacles handling (optional, face 2)
                let obst = cube[2u * stride + idx];
                if (obst > 0.5) {
                    cube[face_out + idx] = 0.0;
                    return;
                }

                let val = cube[face_in + idx];
                
                let nx = config.nx;
                let ny = config.ny;
                
                let idxM_X = idx - 1u;
                let idxP_X = idx + 1u;
                let idxM_Y = idx - nx;
                let idxP_Y = idx + nx;
                let idxM_Z = idx - (nx * ny);
                let idxP_Z = idx + (nx * ny);

                let laplacian = (
                    cube[face_in + idxM_X] + cube[face_in + idxP_X] +
                    cube[face_in + idxM_Y] + cube[face_in + idxP_Y] +
                    cube[face_in + idxM_Z] + cube[face_in + idxP_Z]
                ) - 6.0 * val;

                cube[face_out + idx] = val + config.alpha * laplacian;
            }

            // copy_faces removed
        `;
    }
}
