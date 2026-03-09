import { IDispatcher } from './IDispatcher';
import { IVirtualGrid, IMasterBuffer } from './GridAbstractions';
import { ParityManager } from './ParityManager';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';
import { DataContract } from './DataContract';
import { MasterBuffer } from './MasterBuffer';
import { GpuKernelRegistry } from './kernels/GpuKernelRegistry';

/**
 * Orchestrates the numerical dispatch on the GPU (WebGPU).
 * Maintains physical parity with CPU dispatchers but uses WGSL kernels.
 */
export class GpuDispatcher implements IDispatcher {
    private device: GPUDevice;
    private pipelines: Map<string, GPUComputePipeline> = new Map();
    private uniformBuffer?: GPUBuffer;
    private bindGroups: Map<string, GPUBindGroup> = new Map();

    constructor(
        private vGrid: IVirtualGrid,
        private mBuffer: IMasterBuffer,
        private parityManager: ParityManager
    ) {
        if (!HypercubeGPUContext.isInitialized) {
            throw new Error("GpuDispatcher: HypercubeGPUContext must be initialized before use.");
        }
        this.device = HypercubeGPUContext.device;

        // Create a uniform buffer for simulation parameters (aligned to 256 bytes)
        this.uniformBuffer = this.device.createBuffer({
            size: 512, // Increased from 256 to accommodate 8 objects (requires ~336 bytes)
            usage: (GPUBufferUsage as any).UNIFORM | (GPUBufferUsage as any).COPY_DST,
            label: 'Neo GpuDispatcher Uniforms'
        });
    }

    /**
     * Executes all rules defined in the engine descriptor on the GPU.
     */
    public async dispatch(t: number = 0): Promise<void> {
        const grid = this.vGrid as any;
        const dataContract = grid.dataContract as DataContract;
        const descriptor = dataContract.descriptor;
        const mBuf = this.mBuffer as MasterBuffer;
        const gpuBuffer = mBuf.gpuBuffer;

        if (!gpuBuffer) {
            throw new Error("GpuDispatcher: MasterBuffer does not contain a valid GPUBuffer.");
        }

        // 1. Prepare Aligned Uniforms for ALL chunks in this dispatch (Bumped to 512 for objects)
        const bytesPerChunkAligned = 512;
        const totalUniformSize = this.vGrid.chunks.length * bytesPerChunkAligned;

        if (!this.uniformBuffer || this.uniformBuffer.size < totalUniformSize) {
            if (this.uniformBuffer) this.uniformBuffer.destroy();
            this.uniformBuffer = this.device.createBuffer({
                size: totalUniformSize,
                usage: (GPUBufferUsage as any).UNIFORM | (GPUBufferUsage as any).COPY_DST,
                label: 'Neo GpuDispatcher Uniforms'
            });
            this.bindGroups.clear();
        }

        const u32Data = new Uint32Array(this.vGrid.chunks.length * (bytesPerChunkAligned / 4));

        const commandEncoder = this.device.createCommandEncoder();

        for (const scheme of descriptor.rules) {
            const pipeline = await this.getPipeline(scheme.type);

            // Fill all chunk parameters for this scheme
            for (let i = 0; i < this.vGrid.chunks.length; i++) {
                const vChunk = this.vGrid.chunks[i];
                const base = i * (bytesPerChunkAligned / 4);
                const f32 = new Float32Array(u32Data.buffer);

                // This offset calculation is for the MasterBuffer's data, not the uniform buffer.
                // It's placed here to match the MasterBuffer's internal calculation for consistency,
                // even if not directly used for uniform buffer population.
                const gid = vChunk.y * this.vGrid.chunkLayout.x + vChunk.x;
                const offset = gid * mBuf.strideFace * 23 * 4; // Changed 21 to 23 as per instruction

                u32Data[base + 0] = this.vGrid.dimensions.nx;
                u32Data[base + 1] = this.vGrid.dimensions.ny;
                u32Data[base + 2] = this.vGrid.chunkLayout.x;
                u32Data[base + 3] = this.vGrid.chunkLayout.y;
                f32[base + 4] = (scheme.params?.omega as number) || 1.75;
                f32[base + 5] = (scheme.params?.inflowUx as number) || 0.15;
                f32[base + 6] = t;
                u32Data[base + 7] = this.parityManager.currentTick;
                u32Data[base + 8] = vChunk.x;
                u32Data[base + 9] = vChunk.y;
                u32Data[base + 10] = mBuf.strideFace;
                u32Data[base + 11] = grid.config.objects?.length || 0;

                // Pack up to 8 objects (starting at base + 16, each taking 8 words/32 bytes)
                const objects = grid.config.objects || [];
                for (let j = 0; j < Math.min(objects.length, 8); j++) {
                    const objBase = base + 16 + j * 8;
                    const obj = objects[j];

                    f32[objBase + 0] = obj.position.x;
                    f32[objBase + 1] = obj.position.y;
                    f32[objBase + 2] = obj.dimensions.w;
                    f32[objBase + 3] = obj.dimensions.h;
                    f32[objBase + 4] = obj.properties.isObstacle || 0;
                    f32[objBase + 5] = obj.properties.isSmoke || 0;
                    u32Data[objBase + 6] = (obj.type === 'circle' ? 1 : (obj.type === 'rect' ? 2 : 0));
                    // u32Data[objBase + 7] is padding
                }
            }
            this.device.queue.writeBuffer(this.uniformBuffer!, 0, u32Data);

            for (let i = 0; i < this.vGrid.chunks.length; i++) {
                const vChunk = this.vGrid.chunks[i];
                const uniformOffset = i * bytesPerChunkAligned;

                // Calculate actual physical offset for this chunk in the MasterBuffer
                const gid = vChunk.y * this.vGrid.chunkLayout.x + vChunk.x;
                const chunkBufferOffset = gid * mBuf.totalSlotsPerChunk * mBuf.strideFace * 4;
                const chunkBufferSize = mBuf.totalSlotsPerChunk * mBuf.strideFace * 4;

                const bindGroup = this.getBindGroup(
                    pipeline,
                    gpuBuffer,
                    this.uniformBuffer!,
                    vChunk.id,
                    uniformOffset,
                    chunkBufferOffset,
                    chunkBufferSize
                );

                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setPipeline(pipeline);
                passEncoder.setBindGroup(0, bindGroup);

                const nx = this.vGrid.dimensions.nx;
                const ny = this.vGrid.dimensions.ny;

                passEncoder.dispatchWorkgroups(Math.ceil(nx / 16), Math.ceil(ny / 16));
                passEncoder.end();
            }
        }

        this.device.queue.submit([commandEncoder.finish()]);
    }

    public getChunkBufferParams(chunkIdx: number) {
        const strideFaceBytes = this.mBuffer.strideFace * 4;
        return {
            offset: chunkIdx * this.mBuffer.totalSlotsPerChunk * strideFaceBytes,
            size: this.mBuffer.totalSlotsPerChunk * strideFaceBytes
        };
    }

    private getBindGroup(
        pipeline: GPUComputePipeline,
        dataBuffer: GPUBuffer,
        uniformBuffer: GPUBuffer,
        chunkId: string,
        uniformOffset: number,
        dataOffset: number,
        dataSize: number
    ): GPUBindGroup {
        const key = `${chunkId}_${pipeline.label}_${uniformOffset}_${dataOffset}`;
        if (this.bindGroups.has(key)) return this.bindGroups.get(key)!;

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: dataBuffer, offset: dataOffset, size: dataSize } },
                { binding: 1, resource: { buffer: uniformBuffer, offset: uniformOffset, size: 512 } }
            ],
            label: `BindGroup_${key}`
        });

        this.bindGroups.set(key, bindGroup);
        return bindGroup;
    }

    private async getPipeline(type: string): Promise<GPUComputePipeline> {
        if (this.pipelines.has(type)) return this.pipelines.get(type)!;

        const wgslSource = GpuKernelRegistry.getSource(type);
        const pipeline = HypercubeGPUContext.createComputePipeline(wgslSource, `Neo_${type}`);
        this.pipelines.set(type, pipeline);
        return pipeline;
    }
}
