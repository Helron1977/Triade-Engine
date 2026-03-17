import { IKernel } from './IKernel';
import { ComputeContext } from './ComputeContext';
import { KernelBinder } from './KernelBinder';

/**
 * NeoOceanKernel (CPU)
 * Ported from legacy OceanEngine (2.5D Isometric Ocean).
 * Includes LBM fluid dynamics + Biological advection/diffusion.
 * Refactored for Phase 3: Uses ComputeContext for agnostic memory access.
 */
export class NeoOceanKernel implements IKernel {
    public readonly metadata = {
        roles: {
            source: 'f0',
            destination: 'f0',
            obstacles: 'obstacles',
            auxiliary: [
                'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8',
                'rho', 'vx', 'vy', 'biology'
            ]
        }
    };

    public execute(
        views: Float32Array[],
        context: ComputeContext
    ): void {
        const { nx, ny, padding, pNx, pNy, scheme, indices, gridConfig, chunk } = context;

        const tau_0 = (scheme.params?.tau_0 as number) || 0.8;
        const cflLimit = (scheme.params?.cflLimit as number) || 0.38;
        const bioDiffusion = (scheme.params?.bioDiffusion as number) || 0.05;
        const bioGrowth = (scheme.params?.bioGrowth as number) || 0.0005;
        const invTau = 1.0 / (tau_0 + 1e-6);

        // Resolve views via Binder
        const bound = KernelBinder.bind(this, scheme, views, indices);
        const sourcePair = bound.source;
        
        const f0_in = sourcePair.read;
        const f0_out = bound.destination.write;

        const f1_p = bound.auxiliary[0] || sourcePair;
        const f2_p = bound.auxiliary[1] || sourcePair;
        const f3_p = bound.auxiliary[2] || sourcePair;
        const f4_p = bound.auxiliary[3] || sourcePair;
        const f5_p = bound.auxiliary[4] || sourcePair;
        const f6_p = bound.auxiliary[5] || sourcePair;
        const f7_p = bound.auxiliary[6] || sourcePair;
        const f8_p = bound.auxiliary[7] || sourcePair;

        const f1_in = f1_p.read; const f1_out = f1_p.write;
        const f2_in = f2_p.read; const f2_out = f2_p.write;
        const f3_in = f3_p.read; const f3_out = f3_p.write;
        const f4_in = f4_p.read; const f4_out = f4_p.write;
        const f5_in = f5_p.read; const f5_out = f5_p.write;
        const f6_in = f6_p.read; const f6_out = f6_p.write;
        const f7_in = f7_p.read; const f7_out = f7_p.write;
        const f8_in = f8_p.read; const f8_out = f8_p.write;

        const obstacles = bound.obstacles || new Float32Array(views[0].length);
        const rho_out = (bound.auxiliary[8] || sourcePair).write;
        const ux_out = (bound.auxiliary[9] || sourcePair).write;
        const uy_out = (bound.auxiliary[10] || sourcePair).write;
        const bio_pair = (bound.auxiliary[11] || sourcePair);
        const bio_in = bio_pair.read;
        const bio_out = bio_pair.write;

        const isLeft = chunk.x === 0;
        const isRight = chunk.x === gridConfig.chunks.x - 1;
        const isTop = chunk.y === 0;
        const isBottom = chunk.y === gridConfig.chunks.y - 1;

        // --- UNIFIED LBM + BIO PASS ---
        const pf = new Float32Array(9);
        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
        const cflLimitSq = cflLimit * cflLimit;

        for (let py = padding; py < ny + padding; py++) {
            const isWorldTop = isTop && py === padding;
            const isWorldBottom = isBottom && (py === ny + padding - 1);

            for (let px = padding; px < nx + padding; px++) {
                const i = py * pNx + px;
                const obs = obstacles[i];
                const isWorldLeft = isLeft && px === padding;
                const isWorldRight = isRight && px === nx + padding - 1;

                const isWallLeft = isWorldLeft && (gridConfig.boundaries?.left?.role === 'wall' || gridConfig.boundaries?.all?.role === 'wall');
                const isWallRight = isWorldRight && (gridConfig.boundaries?.right?.role === 'wall' || gridConfig.boundaries?.all?.role === 'wall');
                const isWallTop = isWorldTop && (gridConfig.boundaries?.top?.role === 'wall' || gridConfig.boundaries?.all?.role === 'wall');
                const isWallBottom = isWorldBottom && (gridConfig.boundaries?.bottom?.role === 'wall' || gridConfig.boundaries?.all?.role === 'wall');

                // Standard Pull
                pf[0] = f0_in[i];
                pf[1] = f1_in[i - 1]; // Pull from left
                pf[2] = f2_in[i - pNx]; // Pull from top
                pf[3] = f3_in[i + 1]; // Pull from right
                pf[4] = f4_in[i + pNx]; // Pull from bottom
                pf[5] = f5_in[i - pNx - 1]; // Pull from top-left
                pf[6] = f6_in[i - pNx + 1]; // Pull from top-right
                pf[7] = f7_in[i + pNx + 1]; // Pull from bottom-right
                pf[8] = f8_in[i + pNx - 1]; // Pull from bottom-left

                // Half-way bounce-back overrides for world boundaries (Clamp-to-Edge reflections)
                if (isWallLeft) { pf[1] = f3_in[i]; pf[5] = f7_in[i]; pf[8] = f6_in[i]; }
                if (isWallRight) { pf[3] = f1_in[i]; pf[6] = f8_in[i]; pf[7] = f5_in[i]; }
                if (isWallTop) { pf[2] = f4_in[i]; pf[5] = f7_in[i]; pf[6] = f8_in[i]; }
                if (isWallBottom) { pf[4] = f2_in[i]; pf[7] = f5_in[i]; pf[8] = f6_in[i]; }

                // Internal obstacles (full-way bounce-back, skip collision)
                if (obs > 0.5) {
                    let localRho = 0;
                    f0_out[i] = pf[0]; localRho += pf[0];
                    f1_out[i] = pf[3]; localRho += pf[3];
                    f2_out[i] = pf[4]; localRho += pf[4];
                    f3_out[i] = pf[1]; localRho += pf[1];
                    f4_out[i] = pf[2]; localRho += pf[2];
                    f5_out[i] = pf[7]; localRho += pf[7];
                    f6_out[i] = pf[8]; localRho += pf[8];
                    f7_out[i] = pf[5]; localRho += pf[5];
                    f8_out[i] = pf[6]; localRho += pf[6];
                    rho_out[i] = localRho;
                    continue;
                }

                let rho = 0; for (let k = 0; k < 9; k++) rho += pf[k];
                if (isNaN(rho) || rho < 0.1 || rho > 10.0) { rho = 1.0; for (let k = 0; k < 9; k++) pf[k] = w[k]; }

                const invRho = 1.0 / rho;
                let vx = (pf[1] + pf[5] + pf[8] - (pf[3] + pf[6] + pf[7])) * invRho;
                let vy = (pf[2] + pf[5] + pf[6] - (pf[4] + pf[7] + pf[8])) * invRho;

                const vMagSq = vx * vx + vy * vy;
                if (vMagSq > cflLimitSq) {
                    const scale = cflLimit / Math.sqrt(vMagSq);
                    vx *= scale; vy *= scale;
                }

                rho_out[i] = rho; ux_out[i] = vx; uy_out[i] = vy;

                const u2 = 1.5 * vMagSq;

                const cu1 = 3.0 * vx;
                const cu2 = 3.0 * vy;
                const cu3 = 3.0 * -vx;
                const cu4 = 3.0 * -vy;
                const cu5 = 3.0 * (vx + vy);
                const cu6 = 3.0 * (-vx + vy);
                const cu7 = 3.0 * (-vx - vy);
                const cu8 = 3.0 * (vx - vy);

                f0_out[i] = pf[0] - invTau * (pf[0] - w[0] * rho * (1.0 - u2));
                f1_out[i] = pf[1] - invTau * (pf[1] - w[1] * rho * (1.0 + cu1 + 0.5 * cu1 * cu1 - u2));
                f2_out[i] = pf[2] - invTau * (pf[2] - w[2] * rho * (1.0 + cu2 + 0.5 * cu2 * cu2 - u2));
                f3_out[i] = pf[3] - invTau * (pf[3] - w[3] * rho * (1.0 + cu3 + 0.5 * cu3 * cu3 - u2));
                f4_out[i] = pf[4] - invTau * (pf[4] - w[4] * rho * (1.0 + cu4 + 0.5 * cu4 * cu4 - u2));
                f5_out[i] = pf[5] - invTau * (pf[5] - w[5] * rho * (1.0 + cu5 + 0.5 * cu5 * cu5 - u2));
                f6_out[i] = pf[6] - invTau * (pf[6] - w[6] * rho * (1.0 + cu6 + 0.5 * cu6 * cu6 - u2));
                f7_out[i] = pf[7] - invTau * (pf[7] - w[7] * rho * (1.0 + cu7 + 0.5 * cu7 * cu7 - u2));
                f8_out[i] = pf[8] - invTau * (pf[8] - w[8] * rho * (1.0 + cu8 + 0.5 * cu8 * cu8 - u2));

                // --- BIO PASS (Advection / Diffusion) ---
                const b = bio_in[i];
                const lap = bio_in[i - 1] + bio_in[i + 1] + bio_in[i - pNx] + bio_in[i + pNx] - 4 * b;

                // Semi-Lagrangian Backwards Advection
                const ax = Math.max(0, Math.min(pNx - 2, px - vx * 0.8));
                const ay = Math.max(0, Math.min(pNy - 2, py - vy * 0.8));

                const ix = Math.floor(ax), iy = Math.floor(ay);
                const fx = ax - ix, fy = ay - iy;

                const v00 = bio_in[iy * pNx + ix], v10 = bio_in[iy * pNx + ix + 1];
                const v01 = bio_in[(iy + 1) * pNx + ix], v11 = bio_in[(iy + 1) * pNx + ix + 1];

                const adv = (1 - fy) * ((1 - fx) * v00 + fx * v10) + fy * ((1 - fx) * v01 + fx * v11);

                bio_out[i] = Math.max(0, Math.min(1, adv + bioDiffusion * lap + bioGrowth * b * (1 - b)));
            }
        }
    }
}
