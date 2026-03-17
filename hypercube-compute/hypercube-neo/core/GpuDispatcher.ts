import { IDispatcher } from './IDispatcher';
import { IVirtualGrid } from './topology/GridAbstractions';
import { IBufferBridge } from './IBufferBridge';
import { ParityManager } from './ParityManager';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';
import { DataContract } from './DataContract';
import { GpuKernelRegistry } from './kernels/GpuKernelRegistry';
import { TopologyResolver } from './topology/TopologyResolver';

/**
 * Orchestrates the numerical dispatch on the GPU (WebGPU).
 * Maintains physical parity with CPU dispatchers but uses WGSL kernels.
 * 
 */
export class GpuDispatcher implements IDispatcher {
    public device: GPUDevice;
    public pipelines: Map<string, GPUComputePipeline> = new Map();
    public uniformBuffer?: GPUBuffer;
    public bindGroups: Map<string, GPUBindGroup> = new Map();
    private topologyResolver: TopologyResolver = new TopologyResolver();

    constructor(
        private vGrid: IVirtualGrid,
        private bridge: IBufferBridge,
        private parityManager: ParityManager
    ) {
        if (!HypercubeGPUContext.isInitialized) {
            throw new Error("GpuDispatcher: HypercubeGPUContext must be initialized before use.");
        }
        this.device = HypercubeGPUContext.device;

        // Create a uniform buffer for simulation parameters (aligned to device limits)
        // Create a uniform buffer for simulation parameters (aligned to device limits)
        const bytesPerChunkAligned = HypercubeGPUContext.alignToUniform(384); // 32 slots base + 8*8 slots objects = 384 bytes
        this.uniformBuffer = this.device.createBuffer({
            size: bytesPerChunkAligned,
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
        const gpuBuffer = this.bridge.gpuBuffer;

        console.log(`GpuDispatcher: Dispatching Tick ${this.parityManager.currentTick}, Rules: ${descriptor.rules?.length || 0}`);

        if (!gpuBuffer) {
            throw new Error("GpuDispatcher: MasterBuffer does not contain a valid GPUBuffer.");
        }

        // 1. Prepare Aligned Uniforms for ALL chunks in this dispatch
        const bytesPerChunkAligned = HypercubeGPUContext.alignToUniform(384);
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

                const nx_chunk = Math.floor(this.vGrid.dimensions.nx / this.vGrid.chunkLayout.x);
                const ny_chunk = Math.floor(this.vGrid.dimensions.ny / this.vGrid.chunkLayout.y);

                u32Data[base + 0] = nx_chunk;
                u32Data[base + 1] = ny_chunk;
                u32Data[base + 2] = this.vGrid.chunkLayout.x;
                u32Data[base + 3] = this.vGrid.chunkLayout.y;
                f32[base + 4] = (scheme.params?.omega as number) ?? (scheme.params?.tau_0 as number) ?? 1.75;
                f32[base + 5] = (scheme.params?.inflowUx as number) ?? (scheme.params?.cflLimit as number) ?? 0.12;
                f32[base + 6] = t;
                u32Data[base + 7] = this.parityManager.currentTick;
                u32Data[base + 8] = vChunk.x;
                u32Data[base + 9] = vChunk.y;
                u32Data[base + 10] = this.bridge.strideFace;
                u32Data[base + 11] = grid.config.objects?.length || 0;

                // Absolute Physical Offsets via ParityManager (Agnostic layout)
                const getAbsoluteIdx = (name: string) => {
                    const mappings = (this.vGrid as any).dataContract.getFaceMappings();
                    const faceIdx = mappings.findIndex((m: any) => m.name === name);
                    if (faceIdx === -1) return 0;
                    let baseIdx = 0;
                    for (let k = 0; k < faceIdx; k++) {
                        baseIdx += mappings[k].isPingPong ? 2 : 1;
                    }
                    return baseIdx;
                };

                const tryGetIdx = (name: string, target: 'read' | 'write') => {
                    try {
                        const res = this.parityManager.getFaceIndices(name);
                        return target === 'read' ? res.read : res.write;
                    } catch (e) {
                        return 0; // Return 0 or a default invalid index if not found
                    }
                };

                u32Data[base + 12] = tryGetIdx('obstacles', 'read');
                u32Data[base + 13] = tryGetIdx('vx', 'read') !== 0 ? tryGetIdx('vx', 'read') : tryGetIdx('temperature', 'read');
                u32Data[base + 14] = tryGetIdx('vy', 'read');
                u32Data[base + 15] = tryGetIdx('vorticity', 'read') !== 0 ? tryGetIdx('vorticity', 'read') : tryGetIdx('rho', 'read');
                u32Data[base + 16] = tryGetIdx('smoke', 'read') !== 0 ? tryGetIdx('smoke', 'read') : tryGetIdx('biology', 'read');

                u32Data[base + 17] = tryGetIdx('vx', 'write') !== 0 ? tryGetIdx('vx', 'write') : tryGetIdx('temperature', 'write');
                u32Data[base + 18] = tryGetIdx('vy', 'write');
                u32Data[base + 19] = tryGetIdx('vorticity', 'write') !== 0 ? tryGetIdx('vorticity', 'write') : tryGetIdx('rho', 'write');
                u32Data[base + 20] = tryGetIdx('smoke', 'write') !== 0 ? tryGetIdx('smoke', 'write') : tryGetIdx('biology', 'write');

                // LBM populations base
                u32Data[base + 21] = getAbsoluteIdx('f0');

                // NeoSDF specific jfaStep OR Ocean bio parameters
                if (scheme.type === 'neo-sdf') {
                    u32Data[base + 22] = Math.floor(t);
                    // Pass specific face offsets for this SDF category
                    u32Data[base + 23] = getAbsoluteIdx(scheme.source + '_x');
                    u32Data[base + 30] = getAbsoluteIdx(scheme.source + '_y');
                } else if (scheme.type === 'neo-ocean-v1') {
                    f32[base + 22] = (scheme.params?.bioDiffusion as number) ?? 0.001;
                    f32[base + 23] = (scheme.params?.bioGrowth as number) ?? 0.01;
                }

                // --- Topology Integration ---
                const topo = this.topologyResolver.resolve(vChunk, this.vGrid.chunkLayout, grid.config.boundaries);
                u32Data[base + 24] = topo.leftRole;
                u32Data[base + 25] = topo.rightRole;
                u32Data[base + 26] = topo.topRole;
                u32Data[base + 27] = topo.bottomRole;
                u32Data[base + 28] = topo.frontRole;
                u32Data[base + 29] = topo.backRole;

                // Pack up to 8 objects
                const objects = grid.config.objects || [];
                const metadata = GpuKernelRegistry.getMetadata(scheme.type);
                const objOffset = metadata.uniformObjectOffset ?? 32;

                for (let j = 0; j < Math.min(objects.length, 8); j++) {
                    const objBase = base + objOffset + j * 8;
                    const obj = objects[j];

                    f32[objBase + 0] = obj.position.x;
                    f32[objBase + 1] = obj.position.y;
                    f32[objBase + 2] = obj.dimensions.w;
                    f32[objBase + 3] = obj.dimensions.h;
                    f32[objBase + 4] = obj.properties.isObstacle ?? obj.properties.obstacles ?? 0;
                    f32[objBase + 5] = obj.properties.isSmoke ?? obj.properties.smoke ?? obj.properties.biology ?? obj.properties.temperature ?? obj.properties.isTempInjection ?? 0;
                    u32Data[objBase + 6] = (obj.type === 'circle' ? 1 : (obj.type === 'rect' ? 2 : (obj.type === 'polygon' ? 3 : 0)));
                    f32[objBase + 7] = obj.properties.rho ?? 0;
                }
            }
            this.device.queue.writeBuffer(this.uniformBuffer!, 0, u32Data);

            for (let i = 0; i < this.vGrid.chunks.length; i++) {
                const vChunk = this.vGrid.chunks[i];
                const uniformOffset = i * bytesPerChunkAligned;

                const gid = vChunk.y * this.vGrid.chunkLayout.x + vChunk.x;
                const chunkBufferOffset = gid * this.bridge.totalSlotsPerChunk * this.bridge.strideFace * 4;
                const chunkBufferSize = this.bridge.totalSlotsPerChunk * this.bridge.strideFace * 4;

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

                const nx_chunk = Math.floor(this.vGrid.dimensions.nx / this.vGrid.chunkLayout.x);
                const ny_chunk = Math.floor(this.vGrid.dimensions.ny / this.vGrid.chunkLayout.y);

                passEncoder.dispatchWorkgroups(Math.ceil(nx_chunk / 16), Math.ceil(ny_chunk / 16));
                passEncoder.end();
            }
        }

        this.device.queue.submit([commandEncoder.finish()]);
    }

    public getChunkBufferParams(chunkIdx: number) {
        const strideFaceBytes = this.bridge.strideFace * 4;
        return {
            offset: chunkIdx * this.bridge.totalSlotsPerChunk * strideFaceBytes,
            size: this.bridge.totalSlotsPerChunk * strideFaceBytes
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
                { binding: 1, resource: { buffer: uniformBuffer, offset: uniformOffset, size: HypercubeGPUContext.alignToUniform(384) } }
            ],
            label: `BindGroup_${key}`
        });

        this.bindGroups.set(key, bindGroup);
        return bindGroup;
    }

    private async getPipeline(type: string): Promise<GPUComputePipeline> {
        if (this.pipelines.has(type)) return this.pipelines.get(type)!;

        console.info(`GpuDispatcher: Creating compute pipeline for "${type}"...`);
        const wgslSource = GpuKernelRegistry.getSource(type);
        const pipeline = HypercubeGPUContext.createComputePipeline(wgslSource, `Neo_${type}`);
        this.pipelines.set(type, pipeline);
        return pipeline;
    }
}
