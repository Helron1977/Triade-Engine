import { IVirtualGrid, IBoundarySynchronizer, IRasterizer } from './topology/GridAbstractions';
import { IBufferBridge } from './IBufferBridge';
import { ObjectRasterizer } from './ObjectRasterizer';
import { BoundarySynchronizer } from './topology/BoundarySynchronizer';
import { ParityManager } from './ParityManager';
import { IDispatcher } from './IDispatcher';

/**
 * Orchestrates the execution of a Hypercube Neo simulation.
 * Ensures the correct order of operations: Rasterize -> Compute -> Sync -> Swap.
 */
export class NeoEngineProxy {
    constructor(
        public readonly vGrid: IVirtualGrid,
        public readonly bridge: IBufferBridge,
        public readonly parityManager: ParityManager,
        public readonly rasterizer: IRasterizer,
        public readonly synchronizer: IBoundarySynchronizer,
        public readonly dispatcher: IDispatcher
    ) { }

    /**
     * Initializes the engine state (useful for GPU sync).
     */
    public async init(): Promise<void> {
        // Essential: Initialize LBM fluid to equilibrium to avoid NaN from zero-ghost-cells
        this.bridge.initializeEquilibrium();

        // Run initial rasterization for all chunks (Target 'read' buffer for initialization)
        for (const chunk of this.vGrid.chunks) {
            this.rasterizer.rasterizeChunk(chunk, this.vGrid, this.bridge, 0, 'read');
        }
        // First sync for ghost cells
        this.synchronizer.syncAll(this.vGrid, this.bridge, this.parityManager, 'read');

        // Upload to GPU if needed
        this.bridge.syncToDevice();
    }

    /**
     * Resolves a face name to its logical index in the engine descriptor's faces array.
     * Use this to pass a stable faceIndex to WebGpuRendererNeo.render(), which will
     * resolve it to the correct physical slot via parityManager internally.
     * @throws If the face name is not found.
     */
    public getFaceLogicalIndex(faceName: string): number {
        const descriptor = (this.vGrid as any).dataContract.descriptor;
        const idx = descriptor.faces.findIndex((f: any) => f.name === faceName);
        if (idx === -1) throw new Error(`NeoEngineProxy: Face "${faceName}" not found in descriptor.`);
        return idx;
    }

    /**
     * Executes a single simulation step at time 't'.
     */
    public async step(t: number): Promise<void> {
        const vGrid = this.vGrid as any;
        const isGpu = vGrid.config.mode === 'gpu';

        // 1. Compute: Invoke the numerical dispatcher
        await this.dispatcher.dispatch(t);

        // In GPU mode, injection and boundaries are handled natively in the mono-kernel
        // OR via synchronizer if we explicitly want to cross-sync chunks.
        if (!isGpu) {
            // 2. Rasterize VirtualObjects into the grid (Injection: Write)
            for (const chunk of this.vGrid.chunks) {
                this.rasterizer.rasterizeChunk(chunk, this.vGrid, this.bridge, t);
            }

            // 3. Synchronize boundaries
            this.synchronizer.syncAll(this.vGrid, this.bridge, this.parityManager, 'write');

            // 4. Sync CPU -> GPU (Essential for rasterization/sync to be visible on GPU)
            this.bridge.syncToDevice();
        } else {
            // GPU Initial Sync (Essential for seed injection from CPU)
            if (this.parityManager.currentTick === 0) {
                await this.bridge.syncToDevice();
            }

            // In GPU mode, we still NEED boundary sync for multi-chunk grids
            this.synchronizer.syncAll(this.vGrid, this.bridge, this.parityManager, 'write');
        }

        // 5. Increment the simulation parity
        this.parityManager.nextTick();
    }
}
