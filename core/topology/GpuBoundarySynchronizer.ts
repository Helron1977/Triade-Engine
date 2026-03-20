import { IVirtualGrid, IBoundarySynchronizer } from './GridAbstractions';
import { IBufferBridge } from '../memory/IBufferBridge';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { DataContract } from '../DataContract';

/**
 * GpuBoundarySynchronizer V2.0 - FULL 3D SUPPORT
 * Synchronizes boundaries on the GPU using Compute Shaders.
 * Correctly accounts for NZ > 1 in the sync tasks.
 */
export class GpuBoundarySynchronizer implements IBoundarySynchronizer {
    private device: GPUDevice;
    private pipeline: GPUComputePipeline;
    private batchBuffer?: GPUBuffer;

    constructor() {
        if (!HypercubeGPUContext.isInitialized) {
            throw new Error("GpuBoundarySynchronizer: HypercubeGPUContext must be initialized.");
        }
        this.device = HypercubeGPUContext.device;

        const wgsl = `
            struct SyncParams { srcOffset: u32, dstOffset: u32, count: u32, stride: u32 };
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @group(0) @binding(1) var<storage, read> batch: array<SyncParams>;

            @compute @workgroup_size(64)
            fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>) {
                let p = batch[wg_id.x];
                for (var i = local_id.x; i < p.count; i = i + 64u) {
                    data[p.dstOffset + i * p.stride] = data[p.srcOffset + i * p.stride];
                }
            }
        `;
        const module = this.device.createShaderModule({ code: wgsl });
        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' }
        });
    }

    public syncAll(vGrid: IVirtualGrid, bridge: IBufferBridge, parityManager: ParityManager, target: 'read' | 'write'): void {
        const grid = vGrid as any;
        const gpuBuffer = bridge.gpuBuffer;
        if (!gpuBuffer) return;

        const dataContract = grid.dataContract as DataContract;
        const descriptor = dataContract.descriptor;
        const padding = descriptor.requirements?.ghostCells ?? 0;
        if (padding === 0) return;

        const syncTasks: { srcOffset: number, dstOffset: number, count: number, stride: number }[] = [];
        const tick = parityManager.currentTick;
        const mode = target === 'read' ? tick % 2 : (1 - (tick % 2));

        const nx = Math.floor(grid.config.dimensions.nx / grid.config.chunks.x);
        const ny = Math.floor(grid.config.dimensions.ny / grid.config.chunks.y);
        const nz = grid.config.dimensions.nz || 1;

        const pNx = nx + 2 * padding;
        const pNy = ny + 2 * padding;
        const pNz = nz > 1 ? nz + 2 * padding : 1;
        
        const planeSize = pNx * pNy;

        // 1. Identify synchronized face offsets
        const faceMappings = dataContract.getFaceMappings();
        const syncFaceOffsets: number[] = [];
        let currentOffset = 0;
        for (let j = 0; j < faceMappings.length; j++) {
            const face = faceMappings[j];
            const isSync = descriptor.faces.find(f => f.name === face.name)?.isSynchronized;
            if (isSync) {
                syncFaceOffsets.push(face.isPingPong ? currentOffset + mode : currentOffset);
            }
            currentOffset += face.isPingPong ? 2 : 1;
        }

        const facesPerChunk = bridge.totalSlotsPerChunk;
        const strideFace = bridge.strideFace;

        // 2. Build Sync Tasks (3D Aware)
        for (let i = 0; i < vGrid.chunks.length; i++) {
            const chunk = vGrid.chunks[i];
            const myBase = i * facesPerChunk * strideFace;

            for (const joint of chunk.joints) {
                if (joint.role !== 'joint' || !joint.neighborId) continue;
                const neighborIdx = vGrid.chunks.findIndex(c => c.id === joint.neighborId);
                const theirBase = neighborIdx * facesPerChunk * strideFace;

                for (const faceOffset of syncFaceOffsets) {
                    const mF = myBase + faceOffset * strideFace;
                    const tF = theirBase + faceOffset * strideFace;

                    if (joint.face === 'left') {
                        // All Z-slices for Left edge
                        for(let z=0; z<pNz; z++) {
                            for(let p=0; p<padding; p++) {
                                syncTasks.push({ 
                                    srcOffset: tF + z * planeSize + (nx + p), 
                                    dstOffset: mF + z * planeSize + p, 
                                    count: pNy, stride: pNx 
                                });
                            }
                        }
                    } else if (joint.face === 'right') {
                        for(let z=0; z<pNz; z++) {
                            for(let p=0; p<padding; p++) {
                                syncTasks.push({ 
                                    srcOffset: tF + z * planeSize + (padding + p), 
                                    dstOffset: mF + z * planeSize + (nx + padding + p), 
                                    count: pNy, stride: pNx 
                                });
                            }
                        }
                    } else if (joint.face === 'top') {
                        for(let z=0; z<pNz; z++) {
                            for(let p=0; p<padding; p++) {
                                syncTasks.push({ 
                                    srcOffset: tF + z * planeSize + (ny + p) * pNx, 
                                    dstOffset: mF + z * planeSize + p * pNx, 
                                    count: pNx, stride: 1 
                                });
                            }
                        }
                    } else if (joint.face === 'bottom') {
                        for(let z=0; z<pNz; z++) {
                            for(let p=0; p<padding; p++) {
                                syncTasks.push({ 
                                    srcOffset: tF + z * planeSize + (padding + p) * pNx, 
                                    dstOffset: mF + z * planeSize + (ny + padding + p) * pNx, 
                                    count: pNx, stride: 1 
                                });
                            }
                        }
                    }
                }
            }
        }

        if (syncTasks.length > 0) this.dispatchBatch(gpuBuffer, syncTasks);
    }

    private dispatchBatch(dataBuffer: GPUBuffer, tasks: any[]) {
        const batchSize = tasks.length * 16;
        if (!this.batchBuffer || this.batchBuffer.size < batchSize) {
            if (this.batchBuffer) this.batchBuffer.destroy();
            this.batchBuffer = this.device.createBuffer({
                size: Math.ceil(batchSize / 256) * 256,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
        }

        const data = new Uint32Array(tasks.length * 4);
        for (let i = 0; i < tasks.length; i++) {
            data[i * 4 + 0] = tasks[i].srcOffset;
            data[i * 4 + 1] = tasks[i].dstOffset;
            data[i * 4 + 2] = tasks[i].count;
            data[i * 4 + 3] = tasks[i].stride;
        }
        this.device.queue.writeBuffer(this.batchBuffer!, 0, data);

        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: dataBuffer } },
                { binding: 1, resource: { buffer: this.batchBuffer! } }
            ]
        });
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.min(65535, tasks.length));
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }
}
