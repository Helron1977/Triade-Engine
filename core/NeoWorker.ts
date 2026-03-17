// NeoWorker.ts - Lightweight kernel executor for Web Workers
// Note: This script will be bundled or loaded as a Blob/URL in the ParallelDispatcher.

import { KernelRegistry } from './kernels/KernelRegistry';
import { ComputeContext } from './kernels/ComputeContext';
import { initializeKernels } from './kernels/KernelInitializer';

// Initialize kernels in worker context
initializeKernels();

// Internal state
let sharedBuffer: SharedArrayBuffer | null = null;
const pooledContext: any = {};
const chunkViewsCache: Map<string, Float32Array[]> = new Map();
let persistentFaces: string[] = [];

self.onmessage = async (e: MessageEvent) => {
    const { type, payload } = e.data;

    switch (type) {
        case 'INIT':
            sharedBuffer = payload.sharedBuffer;
            console.log("NeoWorker: Initialized with SharedArrayBuffer");
            self.postMessage({ type: 'READY' });
            break;

        case 'SET_METADATA':
            persistentFaces = payload.persistentFaces || [];
            break;

        case 'COMPUTE_BATCH':
            if (!sharedBuffer) return;
            const { tasks, indices, contextProps, commonParams } = payload;

            for (const task of tasks) {
                const { chunk, schemes, viewsData } = task;

                // 1. Resolve/Cache physical views
                let physicalViews = chunkViewsCache.get(chunk.id);
                if (!physicalViews) {
                    physicalViews = viewsData.map((v: any) => new Float32Array(sharedBuffer!, v.offset, v.length));
                    chunkViewsCache.set(chunk.id, physicalViews!);
                }
                if (!physicalViews) continue;

                // 2. Persistence Copy (Only for truly persistent faces)
                for (const faceName of persistentFaces) {
                    const idx = indices[faceName];
                    if (idx && idx.read !== idx.write) {
                        physicalViews[idx.write].set(physicalViews[idx.read]);
                    }
                }

                // 3. Update pooled context
                pooledContext.nx = chunk.localDimensions.nx;
                pooledContext.ny = chunk.localDimensions.ny;
                pooledContext.nz = chunk.localDimensions.nz || 1;
                pooledContext.pNx = contextProps.pNx;
                pooledContext.pNy = contextProps.pNy;
                pooledContext.padding = contextProps.padding;
                pooledContext.indices = indices;
                pooledContext.params = commonParams;
                pooledContext.chunk = chunk;
                pooledContext.gridConfig = contextProps.gridConfig;

                if (Array.isArray(schemes)) {
                    for (const scheme of schemes) {
                        const kernel = KernelRegistry.get(scheme.type);
                        if (kernel) {
                            pooledContext.scheme = scheme;
                            kernel.execute(physicalViews, pooledContext as ComputeContext);
                        }
                    }
                }
            }

            self.postMessage({ type: 'DONE_BATCH', chunkIds: tasks.map((t:any) => t.chunk.id) });
            break;
    }
};
