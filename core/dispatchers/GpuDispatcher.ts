import { IDispatcher } from './IDispatcher';
import { IVirtualGrid } from '../topology/GridAbstractions';
import { IBufferBridge } from '../memory/IBufferBridge';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { DataContract } from '../DataContract';
import { GpuKernelRegistry } from '../kernels/GpuKernelRegistry';
import { TopologyResolver } from '../topology/TopologyResolver';

/**
 * Orchestrates the numerical dispatch on the GPU (WebGPU).
 * Maintains physical parity with CPU dispatchers but uses WGSL kernels.
 * 
 */
export class GpuDispatcher implements IDispatcher {
    public device: GPUDevice;
    public pipelines: Map<string, GPUComputePipeline> = new Map();
    public uniformBuffer?: GPUBuffer;
    private topologyResolver: TopologyResolver = new TopologyResolver();
    private faceIndexCache: Map<string, number> = new Map();
    private bindGroupCache: Map<string, GPUBindGroup> = new Map();

    constructor(
        private vGrid: IVirtualGrid,
        private bridge: IBufferBridge,
        private parityManager: ParityManager
    ) {
        if (!HypercubeGPUContext.isInitialized) {
            throw new Error("GpuDispatcher: HypercubeGPUContext must be initialized before use.");
        }
        this.device = HypercubeGPUContext.device;

        this.cacheAbsoluteIndices();

        // Create a uniform buffer for simulation parameters (aligned to device limits)
        const bytesPerChunkAligned = HypercubeGPUContext.alignToUniform(384); 
        this.uniformBuffer = this.device.createBuffer({
            size: bytesPerChunkAligned,
            usage: (GPUBufferUsage as any).UNIFORM | (GPUBufferUsage as any).STORAGE | (GPUBufferUsage as any).COPY_DST,
            label: 'Neo GpuDispatcher Uniforms'
        });
    }

    private cacheAbsoluteIndices(): void {
        const mappings = (this.vGrid as any).dataContract.getFaceMappings();
        let currentIdx = 0;
        for (const m of mappings) {
            this.faceIndexCache.set(m.name, currentIdx);
            currentIdx += m.isPingPong ? 2 : 1;
        }
    }

    /**
     * Executes all rules defined in the engine descriptor on the GPU.
     */
    public async dispatch(t: number = 0): Promise<void> {
        const grid = this.vGrid as any;
        const dataContract = grid.dataContract as DataContract;
        const descriptor = dataContract.descriptor;
        const gpuBuffer = this.bridge.gpuBuffer;

        if (!gpuBuffer) {
            throw new Error("GpuDispatcher: MasterBuffer does not contain a valid GPUBuffer.");
        }

        // 1. Prepare Aligned Uniforms
        const bytesPerChunkAligned = HypercubeGPUContext.alignToUniform(384);
        const totalUniformSize = this.vGrid.chunks.length * bytesPerChunkAligned;

        if (!this.uniformBuffer || this.uniformBuffer.size < totalUniformSize) {
            if (this.uniformBuffer) this.uniformBuffer.destroy();
            this.uniformBuffer = this.device.createBuffer({
                size: totalUniformSize,
                usage: (GPUBufferUsage as any).UNIFORM | (GPUBufferUsage as any).STORAGE | (GPUBufferUsage as any).COPY_DST,
                label: 'Neo GpuDispatcher Uniforms'
            });
            this.bindGroupCache.clear();
        }

        const u32Data = new Uint32Array(this.vGrid.chunks.length * (bytesPerChunkAligned / 4));
        const f32Data = new Float32Array(u32Data.buffer);

        const commandEncoder = this.device.createCommandEncoder();

        for (const scheme of descriptor.rules) {
            const pipeline = await this.getPipeline(scheme.type);

            // Helpers for offset resolution using optimized ParityManager and Local Cache
            const getIdx = (name: string, target: 'read' | 'write'): number => {
                try {
                    const res = this.parityManager.getFaceIndices(name);
                    return target === 'read' ? res.read : res.write;
                } catch (e) { return 4294967295; }
            };

            const findFirstIdx = (names: string[], target: 'read' | 'write'): number => {
                for (const name of names) {
                    try {
                        const res = this.parityManager.getFaceIndices(name);
                        return target === 'read' ? res.read : res.write;
                    } catch (e) { continue; }
                }
                return 0;
            };

            const nx_chunk = Math.floor(this.vGrid.dimensions.nx / this.vGrid.chunkLayout.x);
            const ny_chunk = Math.floor(this.vGrid.dimensions.ny / this.vGrid.chunkLayout.y);
            const objectsCount = grid.config.objects?.length || 0;
            const strideFace = this.bridge.strideFace;

            // Fill all chunk parameters for this scheme
            for (let i = 0; i < this.vGrid.chunks.length; i++) {
                const vChunk = this.vGrid.chunks[i];
                const base = i * (bytesPerChunkAligned / 4);

                if (scheme.type === 'neo-tensor-cp-v1') {
                    // Specialized Tensor Layout (Matches NeoTensor.wgsl Uniforms struct)
                    u32Data[base + 0] = nx_chunk;
                    u32Data[base + 1] = ny_chunk;
                    u32Data[base + 2] = this.vGrid.dimensions.nz || 1;
                    f32Data[base + 3] = (scheme.params?.rank as number) || 10;
                    f32Data[base + 4] = (scheme.params?.regularization as number) || 0.05;
                    u32Data[base + 5] = getIdx('mode_a', 'write');
                    u32Data[base + 6] = getIdx('mode_b', 'write');
                    u32Data[base + 7] = getIdx('mode_c', 'write');
                    u32Data[base + 8] = getIdx('target', 'read');
                    u32Data[base + 9] = getIdx('reconstruction', 'write');
                    u32Data[base + 10] = strideFace;
                    u32Data[base + 11] = this.parityManager.currentTick;
                    continue;
                }

                u32Data[base + 0] = nx_chunk;
                u32Data[base + 1] = ny_chunk;
                u32Data[base + 2] = this.vGrid.chunkLayout.x;
                u32Data[base + 3] = this.vGrid.chunkLayout.y;
                f32Data[base + 4] = (scheme.params?.omega as number) ?? (scheme.params?.tau_0 as number) ?? 1.75;
                f32Data[base + 5] = (scheme.params?.inflowUx as number) ?? (scheme.params?.cflLimit as number) ?? 0.12;
                f32Data[base + 6] = t;
                u32Data[base + 7] = this.parityManager.currentTick;
                u32Data[base + 8] = vChunk.x;
                u32Data[base + 9] = vChunk.y;
                u32Data[base + 10] = strideFace;
                u32Data[base + 11] = objectsCount;

                u32Data[base + 12] = getIdx('obstacles', 'read');
                u32Data[base + 13] = findFirstIdx(['vx', 'temperature'], 'read');
                u32Data[base + 14] = getIdx('vy', 'read');
                u32Data[base + 15] = findFirstIdx(['vorticity', 'rho'], 'read');
                u32Data[base + 16] = findFirstIdx(['smoke', 'biology'], 'read');

                u32Data[base + 17] = findFirstIdx(['vx', 'temperature'], 'write');
                u32Data[base + 18] = getIdx('vy', 'write');
                u32Data[base + 19] = findFirstIdx(['vorticity', 'rho'], 'write');
                u32Data[base + 20] = findFirstIdx(['smoke', 'biology'], 'write');

                u32Data[base + 21] = this.faceIndexCache.get('f0') || 0;

                if (scheme.type === 'neo-sdf') {
                    u32Data[base + 22] = Math.floor(t);
                    u32Data[base + 23] = this.faceIndexCache.get(scheme.source + '_x') || 0;
                    u32Data[base + 30] = this.faceIndexCache.get(scheme.source + '_y') || 0;
                } else if (scheme.type === 'neo-ocean-v1') {
                    f32Data[base + 22] = (scheme.params?.bioDiffusion as number) ?? 0.001;
                    f32Data[base + 23] = (scheme.params?.bioGrowth as number) ?? 0.01;
                }

                const topo = this.topologyResolver.resolve(vChunk, this.vGrid.chunkLayout, grid.config.boundaries);
                u32Data[base + 24] = topo.leftRole;
                u32Data[base + 25] = topo.rightRole;
                u32Data[base + 26] = topo.topRole;
                u32Data[base + 27] = topo.bottomRole;
                u32Data[base + 28] = topo.frontRole;
                u32Data[base + 29] = topo.backRole;

                const objects = grid.config.objects || [];
                const metadata = GpuKernelRegistry.getMetadata(scheme.type);
                const objOffset = metadata.uniformObjectOffset ?? 32;

                for (let j = 0; j < Math.min(objects.length, 8); j++) {
                    const objBase = base + objOffset + j * 8;
                    const obj = objects[j];
                    f32Data[objBase + 0] = obj.position.x;
                    f32Data[objBase + 1] = obj.position.y;
                    f32Data[objBase + 2] = obj.dimensions.w;
                    f32Data[objBase + 3] = obj.dimensions.h;
                    f32Data[objBase + 4] = obj.properties.isObstacle ?? obj.properties.obstacles ?? 0;
                    f32Data[objBase + 5] = obj.properties.isSmoke ?? obj.properties.smoke ?? obj.properties.biology ?? obj.properties.temperature ?? obj.properties.isTempInjection ?? 0;
                    u32Data[objBase + 6] = (obj.type === 'circle' ? 1 : (obj.type === 'rect' ? 2 : (obj.type === 'polygon' ? 3 : 0)));
                    f32Data[objBase + 7] = obj.properties.rho ?? 0;
                }
            }
            this.device.queue.writeBuffer(this.uniformBuffer!, 0, u32Data);

            for (let i = 0; i < this.vGrid.chunks.length; i++) {
                const vChunk = this.vGrid.chunks[i];
                const uniformOffset = i * bytesPerChunkAligned;

                const gid = vChunk.y * this.vGrid.chunkLayout.x + vChunk.x;
                const chunkBufferOffset = gid * this.bridge.totalSlotsPerChunk * strideFace * 4;
                const chunkBufferSize = this.bridge.totalSlotsPerChunk * strideFace * 4;

                const bindGroup = this.getCachedBindGroup(
                    pipeline,
                    gpuBuffer,
                    this.uniformBuffer!,
                    gid,
                    uniformOffset,
                    chunkBufferOffset,
                    chunkBufferSize
                );

                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setPipeline(pipeline);
                passEncoder.setBindGroup(0, bindGroup);
                
                if (scheme.type === 'neo-tensor-cp-v1') {
                    const nx_chunk = vChunk.localDimensions.nx;
                    const ny_chunk = vChunk.localDimensions.ny;
                    const nz = this.vGrid.dimensions.nz || 1;
                    const maxDim = Math.max(nx_chunk, ny_chunk, nz);
                    passEncoder.dispatchWorkgroups(Math.ceil(maxDim / 16), 1, 1);
                } else {
                    const nx_chunk = vChunk.localDimensions.nx;
                    const ny_chunk = vChunk.localDimensions.ny;
                    passEncoder.dispatchWorkgroups(Math.ceil(nx_chunk / 16), Math.ceil(ny_chunk / 16));
                }
                passEncoder.end();
            }
        }
        this.device.queue.submit([commandEncoder.finish()]);
    }

    private getCachedBindGroup(
        pipeline: GPUComputePipeline,
        dataBuffer: GPUBuffer,
        uniformBuffer: GPUBuffer,
        globalChunkIdx: number,
        uniformOffset: number,
        dataOffset: number,
        dataSize: number
    ): GPUBindGroup {
        const key = `${pipeline.label}_${globalChunkIdx}_${uniformOffset}_${dataOffset}`;
        let bg = this.bindGroupCache.get(key);
        if (bg) return bg;

        bg = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: dataBuffer, offset: dataOffset, size: dataSize } },
                { binding: 1, resource: { buffer: uniformBuffer, offset: uniformOffset, size: HypercubeGPUContext.alignToUniform(384) } }
            ]
        });

        this.bindGroupCache.set(key, bg);
        return bg;
    }

    public async getPipeline(type: string): Promise<GPUComputePipeline> {
        let p = this.pipelines.get(type);
        if (p) return p;

        console.info(`GpuDispatcher: Creating compute pipeline for "${type}"...`);
        const wgslSource = GpuKernelRegistry.getSource(type);
        p = HypercubeGPUContext.createComputePipeline(wgslSource, `Neo_${type}`);
        this.pipelines.set(type, p);
        return p;
    }

    public getChunkBufferParams(chunkIdx: number) {
        const strideFaceBytes = this.bridge.strideFace * 4;
        return {
            offset: chunkIdx * this.bridge.totalSlotsPerChunk * strideFaceBytes,
            size: this.bridge.totalSlotsPerChunk * strideFaceBytes
        };
    }
}
