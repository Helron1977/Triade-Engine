import { IVirtualGrid, IBoundarySynchronizer, IRasterizer } from './topology/GridAbstractions';
import { IBufferBridge } from './memory/IBufferBridge';
import { ParityManager } from './ParityManager';
import { IDispatcher } from './dispatchers/IDispatcher';

/**
 * Orchestrates the execution of a Hypercube Neo simulation.
 * Ensures the correct order of operations: Compute -> Rasterize -> Sync -> Commit.
 * This class is strictly hardware-agnostic.
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
     * Initializes the engine state.
     */
    public async init(): Promise<void> {
        // 1. Initialize memory to equilibrium
        this.bridge.initializeEquilibrium();

        // 2. Initial rasterization (Read buffer)
        for (const chunk of this.vGrid.chunks) {
            this.rasterizer.rasterizeChunk(chunk, this.vGrid, this.bridge, 0, 'read');
        }

        // 3. Initial boundary sync
        this.synchronizer.syncAll(this.vGrid, this.bridge, this.parityManager, 'read');

        // 4. Initial commit (upload to hardware if needed)
        this.bridge.syncToDevice();
        this.bridge.commit();
    }

    /**
     * Resolves a face name to its logical index in the engine descriptor's faces array.
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
        // 1. Compute: Invoke the numerical dispatcher
        await this.dispatcher.dispatch(t);

        // 2. Injection: Rasterize VirtualObjects into the grid (Write buffer)
        for (const chunk of this.vGrid.chunks) {
            this.rasterizer.rasterizeChunk(chunk, this.vGrid, this.bridge, t);
        }

        // 3. Boundary Synchronization
        this.synchronizer.syncAll(this.vGrid, this.bridge, this.parityManager, 'write');

        // 4. Commit changes (Platform-specific finalization)
        this.bridge.commit();

        // 5. Increment the simulation parity
        this.parityManager.nextTick();
    }
}
