import { IVirtualGrid, IBoundarySynchronizer } from './GridAbstractions';
import { IBufferBridge } from '../memory/IBufferBridge';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { DataContract } from '../DataContract';

/**
 * Synchronizes boundaries on the GPU using Compute Shaders.
 * Direct VRAM-to-VRAM copy without CPU intervention.
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

        // Load sync kernel
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
        const padding = descriptor.requirements.ghostCells;
        if (padding === 0) return; // Nothing to sync

        const syncTasks: { srcOffset: number, dstOffset: number, count: number, stride: number }[] = [];
        const tick = parityManager.currentTick;
        const mode = target === 'read' ? tick % 2 : (1 - (tick % 2));

        const chunkXCount = grid.config.chunks.x;
        const nx = Math.floor(grid.config.dimensions.nx / chunkXCount);
        const ny = Math.floor(grid.config.dimensions.ny / grid.config.chunks.y);
        const pNx = nx + 2 * padding;
        const cellsPerFace = pNx * (ny + 2 * padding);

        // 1. Identify synchronized face offsets
        const faceMappings = dataContract.getFaceMappings();
        const syncFaceOffsets: number[] = [];
        let currentOffset = 0;
        for (let j = 0; j < faceMappings.length; j++) {
            const face = faceMappings[j];
            const isSync = descriptor.faces.find(f => f.name === face.name)?.isSynchronized;
            if (isSync) {
                // IMPORTANT: Only apply mode shifting if it's a ping-pong face!
                if (face.isPingPong) {
                    syncFaceOffsets.push(currentOffset + mode);
                } else {
                    syncFaceOffsets.push(currentOffset);
                }
            }
            currentOffset += face.isPingPong ? 2 : 1;
        }

        const facesPerChunk = bridge.totalSlotsPerChunk;
        const strideFace = bridge.strideFace;

        // 2. Build Sync Tasks
        for (let i = 0; i < vGrid.chunks.length; i++) {
            const chunk = vGrid.chunks[i];
            const myBase = i * facesPerChunk * strideFace;

            for (const joint of chunk.joints) {
                if (joint.role !== 'joint' || !joint.neighborId) continue;

                const neighborIdx = vGrid.chunks.findIndex(c => c.id === joint.neighborId);
                const theirBase = neighborIdx * facesPerChunk * strideFace;

                for (const faceOffset of syncFaceOffsets) {
                    const myFaceBase = myBase + faceOffset * strideFace;
                    const theirFaceBase = theirBase + faceOffset * strideFace;

                    if (joint.face === 'left') {
                        // My Left Ghost (x < padding) <- Their Right Real (x = nx to nx+padding-1)
                        for (let p = 0; p < padding; p++) {
                            syncTasks.push({ 
                                srcOffset: theirFaceBase + (nx + p), 
                                dstOffset: myFaceBase + p, 
                                count: ny + 2 * padding, 
                                stride: pNx 
                            });
                        }
                    } else if (joint.face === 'right') {
                        // My Right Ghost (x = nx+padding to nx+2*padding-1) <- Their Left Real (x = padding to 2*padding-1)
                        for (let p = 0; p < padding; p++) {
                            syncTasks.push({ 
                                srcOffset: theirFaceBase + (padding + p), 
                                dstOffset: myFaceBase + (nx + padding + p), 
                                count: ny + 2 * padding, 
                                stride: pNx 
                            });
                        }
                    } else if (joint.face === 'top') {
                        // My Top Ghost (y < padding) <- Their Bottom Real (y = ny to ny+padding-1)
                        for (let p = 0; p < padding; p++) {
                            syncTasks.push({ 
                                srcOffset: theirFaceBase + (ny + p) * pNx, 
                                dstOffset: myFaceBase + p * pNx, 
                                count: pNx, 
                                stride: 1 
                            });
                        }
                    } else if (joint.face === 'bottom') {
                        // My Bottom Ghost (y = ny+padding to ny+2*padding-1) <- Their Top Real (y = padding to 2*padding-1)
                        for (let p = 0; p < padding; p++) {
                            syncTasks.push({ 
                                srcOffset: theirFaceBase + (padding + p) * pNx, 
                                dstOffset: myFaceBase + (ny + padding + p) * pNx, 
                                count: pNx, 
                                stride: 1 
                            });
                        }
                    }
                }
            }

            // 3. DIAGONAL CORNERS (Essential for D2Q9 diagonal propagation)
            const cx = chunk.x;
            const cy = chunk.y;
            const findChunk = (tx: number, ty: number) => vGrid.chunks.find(c => c.x === tx && c.y === ty);

            const corners = [
                { dx: -1, dy: -1, name: 'TL' },
                { dx: 1, dy: -1, name: 'TR' },
                { dx: -1, dy: 1, name: 'BL' },
                { dx: 1, dy: 1, name: 'BR' }
            ];

            for (const corner of corners) {
                const neighbor = findChunk(cx + corner.dx, cy + corner.dy);
                if (!neighbor) continue;

                const neighborIdx = vGrid.chunks.findIndex(c => c.id === neighbor.id);
                const theirBase = neighborIdx * facesPerChunk * strideFace;

                for (const faceOffset of syncFaceOffsets) {
                    const myFaceBase = myBase + faceOffset * strideFace;
                    const theirFaceBase = theirBase + faceOffset * strideFace;

                    if (corner.name === 'TL') {
                        // My TL Ghost <- Their BR Real
                        syncTasks.push({ srcOffset: theirFaceBase + (ny * pNx + nx), dstOffset: myFaceBase + 0, count: 1, stride: 1 });
                    } else if (corner.name === 'TR') {
                        // My TR Ghost <- Their BL Real
                        syncTasks.push({ srcOffset: theirFaceBase + (ny * pNx + 1), dstOffset: myFaceBase + (nx + 1), count: 1, stride: 1 });
                    } else if (corner.name === 'BL') {
                        // My BL Ghost <- Their TR Real
                        syncTasks.push({ srcOffset: theirFaceBase + (1 * pNx + nx), dstOffset: myFaceBase + (ny + 1) * pNx, count: 1, stride: 1 });
                    } else if (corner.name === 'BR') {
                        // My BR Ghost <- Their TL Real
                        syncTasks.push({ srcOffset: theirFaceBase + (1 * pNx + 1), dstOffset: myFaceBase + ((ny + 1) * pNx + (nx + 1)), count: 1, stride: 1 });
                    }
                }
            }
        }

        if (syncTasks.length === 0) return;

        // 3. Dispatch GPU Batch
        this.dispatchBatch(gpuBuffer, syncTasks);
    }

    private dispatchBatch(dataBuffer: GPUBuffer, tasks: any[]) {
        const batchSize = tasks.length * 16; // 4 * u32
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
        pass.dispatchWorkgroups(tasks.length);
        pass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}
