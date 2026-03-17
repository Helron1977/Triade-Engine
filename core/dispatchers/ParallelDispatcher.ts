import { IDispatcher } from './IDispatcher';
import { IVirtualGrid } from '../topology/GridAbstractions';
import { IBufferBridge } from '../memory/IBufferBridge';
import { ParityManager } from '../ParityManager';

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
            const worker = new Worker(new URL('../NeoWorker.ts', import.meta.url), { type: 'module' });

            const initPromise = new Promise<void>((resolve) => {
                worker.addEventListener('message', (e) => {
                    const { type, chunkId, chunkIds } = e.data;
                    if (type === 'READY') {
                        resolve();
                    } else if (type === 'DONE_BATCH') {
                        for (const cid of chunkIds) {
                            const resolver = this.chunkResolvers.get(cid);
                            if (resolver) {
                                this.chunkResolvers.delete(cid);
                                resolver();
                            }
                        }
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

    private initMetadata(descriptor: any) {
        const persistentFaces = descriptor.faces
            .filter((f: any) => f.isPersistent !== false)
            .map((f: any) => f.name);

        for (const worker of this.workers) {
            worker.postMessage({
                type: 'SET_METADATA',
                payload: { persistentFaces }
            });
        }
    }

    public async dispatch(t: number): Promise<void> {
        await Promise.all(this.initPromises);

        const grid = this.vGrid as any;
        const descriptor = grid.dataContract.descriptor;

        // 1. One-time static structure caching
        if (!(this as any)._staticBatchReady) {
            this.initMetadata(descriptor);
            
            let maxNx = 0, maxNy = 0, maxNz = 0;
            for (const chunk of this.vGrid.chunks) {
                maxNx = Math.max(maxNx, chunk.localDimensions.nx);
                maxNy = Math.max(maxNy, chunk.localDimensions.ny);
                maxNz = Math.max(maxNz, chunk.localDimensions.nz || 1);
            }
            const padding = descriptor.requirements.ghostCells;
            const pNx = maxNx + 2 * padding;
            const pNy = maxNy + 2 * padding;
            const pNz = maxNz; // 3D slices usually not padded in Z for simple tensors

            const workerTasksMap = new Map<Worker, any[]>();
            for (const w of this.workers) workerTasksMap.set(w, []);

            for (let i = 0; i < this.vGrid.chunks.length; i++) {
                const vChunk = this.vGrid.chunks[i];
                const worker = this.workers[i % this.workers.length];
                const pViews = this.bridge.getChunkViews(vChunk.id);
                
                const viewsData = pViews.map(v => ({ offset: v.byteOffset, length: v.length }));
                workerTasksMap.get(worker)!.push({
                    chunk: vChunk,
                    schemes: descriptor.rules,
                    viewsData
                });
            }

            (this as any)._cachedWorkerTasks = Array.from(workerTasksMap.entries());
            (this as any)._cachedContextProps = { pNx, pNy, padding, gridConfig: grid.config };
            (this as any)._staticBatchReady = true;
        }

        // 2. Refresh dynamic indices (Parity based)
        const faceIndices: Record<string, { read: number; write: number }> = {};
        for (const face of descriptor.faces) {
            faceIndices[face.name] = this.parityManager.getFaceIndices(face.name);
        }

        const commonParams = {
            time: t,
            tick: this.parityManager.currentTick,
            objects: grid.config?.objects,
            chunksList: this.vGrid.chunks
        };

        const chunkPromises: Promise<void>[] = [];
        for (const chunk of this.vGrid.chunks) {
            chunkPromises.push(new Promise<void>((resolve) => {
                this.chunkResolvers.set(chunk.id, resolve);
            }));
        }

        // 3. Dispatch cached batches
        const batches: [Worker, any[]][] = (this as any)._cachedWorkerTasks;
        const ctxProps = (this as any)._cachedContextProps;

        for (let i = 0; i < batches.length; i++) {
            const [worker, tasks] = batches[i];
            worker.postMessage({
                type: 'COMPUTE_BATCH',
                payload: {
                    tasks,
                    indices: faceIndices,
                    contextProps: ctxProps,
                    commonParams
                }
            });
        }

        await Promise.all(chunkPromises);
    }
}
