import { describe, it, expect } from 'vitest';
import { NeoTensorKernel } from '../core/kernels/NeoTensorKernel';
import { ComputeContext } from '../core/kernels/ComputeContext';

describe('NeoTensorKernel (ALS)', () => {
    it('should converge on a simple rank-1 tensor', () => {
        const kernel = new NeoTensorKernel();
        const nx = 4, ny = 4, nz = 2;
        const rank = 1;

        // Target: Only one entry is 5
        const target = new Float32Array(nx * ny * nz).fill(0);
        target[0] = 5;

        const mode_a = new Float32Array(nx * rank).fill(0.1);
        const mode_b = new Float32Array(ny * rank).fill(0.1);
        const mode_c = new Float32Array(nz * rank).fill(0.1);

        const context: any = {
            nx, ny, nz,
            params: { rank, regularization: 0.001 }
        };

        const views = [mode_a, mode_b, mode_c, target];

        // Run many iterations
        for (let i = 0; i < 500; i++) {
            kernel.execute(views, context as ComputeContext);
        }

        // Reconstruction: a[0]*b[0]*c[0] should be close to 5
        const pred = mode_a[0] * mode_b[0] * mode_c[0];
        console.log('Final Prediction:', pred);
        expect(pred).toBeGreaterThan(4.5);
        expect(pred).toBeLessThan(5.5);
    });
});
