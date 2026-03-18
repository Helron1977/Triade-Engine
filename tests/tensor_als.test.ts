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
            scheme: { params: { rank, regularization: 0.001 } }
        };

        const recon = new Float32Array(nx * ny * nz).fill(0);
        const views = [mode_a, mode_b, mode_c, target, recon];

        // Run iterations for convergence with the faster learning rate
        for (let i = 0; i < 500; i++) {
            kernel.execute(views, context as ComputeContext);
        }

        // Reconstruction: a[0]*b[0]*c[0] should be close to 5
        const pred = mode_a[0] * mode_b[0] * mode_c[0];
        console.log('Final Prediction:', pred);
        expect(pred).toBeGreaterThan(4.5);
        expect(pred).toBeLessThan(5.5);
    });
    
    it('should handle rank-3 and sparse data', () => {
        const kernel = new NeoTensorKernel();
        const nx = 10, ny = 10, nz = 5;
        const rank = 3;

        // Target: Random sparse tensor (density ~10%)
        const target = new Float32Array(nx * ny * nz).fill(0);
        for (let i = 0; i < 50; i++) {
            const idx = Math.floor(Math.random() * target.length);
            target[idx] = 1.0 + Math.random();
        }

        const mode_a = new Float32Array(nx * rank).fill(0.1);
        const mode_b = new Float32Array(ny * rank).fill(0.1);
        const mode_c = new Float32Array(nz * rank).fill(0.1);
        const recon = new Float32Array(nx * ny * nz).fill(0);
        
        const views = [mode_a, mode_b, mode_c, target, recon];
        const context: any = { nx, ny, nz, scheme: { params: { rank, regularization: 0.05 } } };

        // Initial MSE
        let initialMSE = 0;
        let count = 0;
        for (let i = 0; i < target.length; i++) {
            if (target[i] > 0) {
                initialMSE += target[i] * target[i];
                count++;
            }
        }
        initialMSE /= count;

        // Run iterations
        for (let i = 0; i < 400; i++) {
            kernel.execute(views, context as ComputeContext);
        }

        // Final MSE
        let finalMSE = 0;
        for (let i = 0; i < target.length; i++) {
            if (target[i] > 0) {
                const diff = target[i] - recon[i];
                finalMSE += diff * diff;
            }
        }
        finalMSE /= count;

        console.log(`Rank-3 Sparse Test: Initial MSE=${initialMSE.toFixed(4)}, Final MSE=${finalMSE.toFixed(6)}`);
        expect(finalMSE).toBeLessThan(initialMSE);
        expect(Number.isNaN(finalMSE)).toBe(false);
    });
});
