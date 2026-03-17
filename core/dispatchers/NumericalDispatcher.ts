import { IDispatcher } from './IDispatcher';
import { IVirtualGrid } from '../topology/GridAbstractions';
import { ParityManager } from '../ParityManager';
import { KernelRegistry } from '../kernels/KernelRegistry';
import { DataContract } from '../DataContract';

import { ComputeContext } from '../kernels/ComputeContext';
import { IBufferBridge } from '../memory/IBufferBridge';

/**
 * Orchestrates the numerical dispatch for all chunks (Single-threaded).
 * Bridges the declarative schemes with physical memory and kernels.
 */
export class NumericalDispatcher implements IDispatcher {
    // We use a mutable handle for the pooled object
    private pooledContext: any = {};

    constructor(
        private vGrid: IVirtualGrid,
        private bridge: IBufferBridge,
        private parityManager: ParityManager
    ) { }

    /**
     * Executes all rules defined in the engine descriptor for all chunks.
     */
    public dispatch(t: number = 0): void {
        const grid = this.vGrid as any;
        const dataContract = grid.dataContract as DataContract;
        const descriptor = dataContract.descriptor;

        // 1. Prepare indices for all faces once per step
        const faceIndices: Record<string, { read: number; write: number }> = {};
        for (const face of descriptor.faces) {
            faceIndices[face.name] = this.parityManager.getFaceIndices(face.name);
        }

        // 2. Pre-compute uniform parameters
        let maxNx = 0, maxNy = 0, maxNz = 0;
        for (const chunk of this.vGrid.chunks) {
            maxNx = Math.max(maxNx, chunk.localDimensions.nx);
            maxNy = Math.max(maxNy, chunk.localDimensions.ny);
            maxNz = Math.max(maxNz, chunk.localDimensions.nz || 1);
        }

        const padding = descriptor.requirements.ghostCells;
        const pNx = maxNx + 2 * padding;
        const pNy = maxNy + 2 * padding;

        const globalParams = {
            time: t,
            tick: this.parityManager.currentTick,
            objects: grid.config?.objects,
            chunksList: this.vGrid.chunks
        };

        // Initialize/Update persistent context properties
        this.pooledContext.pNx = pNx;
        this.pooledContext.pNy = pNy;
        this.pooledContext.padding = padding;
        this.pooledContext.indices = faceIndices;
        this.pooledContext.params = globalParams;
        this.pooledContext.gridConfig = grid.config;

        // 3. Iterate through all chunks
        for (const vChunk of this.vGrid.chunks) {
            const pViews = this.bridge.getChunkViews(vChunk.id);

            // Persistence Copy: Ensure ping-pong buffers stay in sync for persistent data
            for (const face of descriptor.faces) {
                const indices = faceIndices[face.name];
                if (indices.write !== indices.read && face.isPersistent !== false) {
                    pViews[indices.write].set(pViews[indices.read]);
                }
            }

            // 4. Compute Context Execution using pooled object
            this.pooledContext.nx = vChunk.localDimensions.nx;
            this.pooledContext.ny = vChunk.localDimensions.ny;
            this.pooledContext.nz = vChunk.localDimensions.nz || 1;
            this.pooledContext.chunk = vChunk;

            for (const scheme of descriptor.rules) {
                const kernel = KernelRegistry.get(scheme.type);
                if (kernel) {
                    this.pooledContext.scheme = scheme;
                    kernel.execute(pViews, this.pooledContext as ComputeContext);
                }
            }
        }
    }
}
