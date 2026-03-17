import { IDispatcher } from './IDispatcher';
import { IVirtualGrid } from './topology/GridAbstractions';
import { IBufferBridge } from './IBufferBridge';
import { ParityManager } from './ParityManager';
import { DataContract } from './DataContract';

/**
 * ParallelDispatcher leveraging Web Workers and SharedArrayBuffer.
 */
export class ParallelDispatcher implements IDispatcher {
    private workers: Worker[] = [];
    private initPromises: Promise<void>[] = [];
    private chunkResolvers: Map<string, () => void> = new Map();

    constructor(
        private vGrid: IVirtualGrid,
        private bridge: IBufferBridge,
        private parityManager: ParityManager
    ) {
        this.initWorkers();
    }


    private initWorkers() {
        const numWorkers = Math.min(this.vGrid.chunks.length, navigator.hardwareConcurrency || 4);
        console.log(`ParallelDispatcher: Scaling to ${numWorkers} workers...`);

        for (let i = 0; i < numWorkers; i++) {
            const worker = new Worker(new URL('./NeoWorker.ts', import.meta.url), { type: 'module' });

            const initPromise = new Promise<void>((resolve) => {
                worker.addEventListener('message', (e) => {
                    const { type, chunkId } = e.data;
                    if (type === 'READY') {
                        resolve();
                    } else if (type === 'DONE') {
                        const resolver = this.chunkResolvers.get(chunkId);
                        if (resolver) {
                            this.chunkResolvers.delete(chunkId);
                            resolver();
                        }
                    }
                }, { once: false });
            });
            this.initPromises.push(initPromise);

            worker.postMessage({
                type: 'INIT',
                payload: { sharedBuffer: this.bridge.rawBuffer }
            });

            this.workers.push(worker);
        }
    }

    public async dispatch(t: number): Promise<void> {
        // Initialization Barrier: Ensure all workers have received the SharedArrayBuffer
        await Promise.all(this.initPromises);

        const grid = this.vGrid as any;
        const descriptor = grid.dataContract.descriptor;

        // 1. Prepare shared metadata
        const faceIndices: Record<string, { read: number; write: number }> = {};
        for (const face of descriptor.faces) {
            faceIndices[face.name] = this.parityManager.getFaceIndices(face.name);
        }

        // Pre-compute uniform parameters once for all chunks
        let maxNx = 0, maxNy = 0;
        for (const chunk of this.vGrid.chunks) {
            maxNx = Math.max(maxNx, chunk.localDimensions.nx);
            maxNy = Math.max(maxNy, chunk.localDimensions.ny);
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

        // 2. Parallel Dispatch
        const chunkExecutions = this.vGrid.chunks.map((vChunk, idx) => {
            const worker = this.workers[idx % this.workers.length];
            const pViews = this.bridge.getChunkViews(vChunk.id);

            // Consistency Copy (Main Thread)
            for (const face of descriptor.faces) {
                const indices = faceIndices[face.name];
                if (indices.write !== indices.read && face.isPersistent !== false) {
                    pViews[indices.write].set(pViews[indices.read]);
                }
            }

            const viewsData = pViews.map(v => ({ offset: v.byteOffset, length: v.length }));

            return new Promise<void>((resolve) => {
                this.chunkResolvers.set(vChunk.id, resolve);

                worker.postMessage({
                    type: 'COMPUTE',
                    payload: {
                        chunk: vChunk,
                        schemes: descriptor.rules,
                        indices: faceIndices,
                        contextProps: {
                            pNx, pNy, padding,
                            params: globalParams,
                            gridConfig: grid.config
                        },
                        viewsData
                    }
                });
            });
        });

        await Promise.all(chunkExecutions);
    }
}
