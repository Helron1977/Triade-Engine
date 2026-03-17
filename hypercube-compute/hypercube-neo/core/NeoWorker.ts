// NeoWorker.ts - Lightweight kernel executor for Web Workers
// Note: This script will be bundled or loaded as a Blob/URL in the ParallelDispatcher.

import { KernelRegistry } from './kernels/KernelRegistry';
import { ComputeContext } from './kernels/ComputeContext';
import { initializeKernels } from './kernels/KernelInitializer';

// Initialize kernels in worker context
initializeKernels();

// Internal state
let sharedBuffer: SharedArrayBuffer | null = null;

self.onmessage = async (e: MessageEvent) => {
    const { type, payload } = e.data;

    switch (type) {
        case 'INIT':
            sharedBuffer = payload.sharedBuffer;
            console.log("NeoWorker: Initialized with SharedArrayBuffer");
            self.postMessage({ type: 'READY' });
            break;

        case 'COMPUTE':
            if (!sharedBuffer) return;
            const { chunk, schemes, indices, contextProps, viewsData } = payload;

            // Reconstruct views from SharedArrayBuffer offsets
            const physicalViews = viewsData.map((v: any) => new Float32Array(sharedBuffer!, v.offset, v.length));

            if (Array.isArray(schemes)) {
                for (const scheme of schemes) {
                    const kernel = KernelRegistry.get(scheme.type);
                    if (kernel) {
                        const context: ComputeContext = {
                            nx: chunk.localDimensions.nx,
                            ny: chunk.localDimensions.ny,
                            pNx: contextProps.pNx,
                            pNy: contextProps.pNy,
                            padding: contextProps.padding,
                            scheme,
                            indices,
                            params: contextProps.params,
                            chunk,
                            gridConfig: contextProps.gridConfig
                        };
                        kernel.execute(physicalViews, context);
                    }
                }
            }

            self.postMessage({ type: 'DONE', chunkId: chunk.id });
            break;
    }
};
