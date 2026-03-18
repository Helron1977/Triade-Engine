import { IKernel } from './IKernel';
import { ComputeContext } from './ComputeContext';

/**
 * NeoTensorKernel: Specialized kernel for Tensor CP Decomposition via ALS.
 * Implements factor matrix updates (Mode-A, B, C) using Khatri-Rao products.
 * Optimized for O(1) memory access via ComputeContext.
 */
export class NeoTensorKernel implements IKernel {
    public readonly metadata = {
        roles: {
            mode_a: 'factor', // [n_users x rank]
            mode_b: 'factor', // [n_films x rank]
            mode_c: 'factor', // [n_genres x rank]
            target: 'tensor',  // [n_users x n_films x n_genres]
            reconstruction: 'tensor' // [n_users x n_films x n_genres]
        }
    };

    public execute(
        views: Float32Array[],
        context: ComputeContext
    ): void {
        const { nx, ny, nz } = context;
        const schemeParams = (context as any).scheme?.params || {};
        const rank = (schemeParams.rank as number) || 10;
        const reg = (schemeParams.regularization as number) || 0.05;
        const lr = (schemeParams.learningRate as number) || 0.01;

        const mode_a = views[0];
        const mode_b = views[1];
        const mode_c = views[2];
        const target = views[3];
        const recon = views[4];

        for (let k = 0; k < nz; k++) {
            for (let j = 0; j < ny; j++) {
                for (let i = 0; i < nx; i++) {
                    const idx = i + j * nx + k * nx * ny;
                    const val = target[idx];

                    // Current reconstruction: sum_{r=0}^{rank-1} a[i,r] * b[j,r] * c[k,r]
                    let pred = 0;
                    for (let r = 0; r < rank; r++) {
                        pred += mode_a[i * rank + r] * mode_b[j * rank + r] * mode_c[k * rank + r];
                    }
                    
                    // Store reconstruction for visualization
                    if (recon) recon[idx] = pred;

                    if (val === 0) continue; // Sparse handling

                    const err = val - pred;

                    for (let r = 0; r < rank; r++) {
                        const idxA = i * rank + r;
                        const idxB = j * rank + r;
                        const idxC = k * rank + r;

                        const gradA = err * mode_b[idxB] * mode_c[idxC];
                        const gradB = err * mode_a[idxA] * mode_c[idxC];
                        const gradC = err * mode_a[idxA] * mode_b[idxB];

                        mode_a[idxA] += lr * (gradA - reg * mode_a[idxA]);
                        mode_b[idxB] += lr * (gradB - reg * mode_b[idxB]);
                        mode_c[idxC] += lr * (gradC - reg * mode_c[idxC]);
                    }
                }
            }
        }
    }
}
