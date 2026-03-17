import { IKernel } from './IKernel';
import { ComputeContext } from './ComputeContext';

/**
 * LBMSmokeKernel
 * Advects a passive scalar (smoke) using the macroscopic velocity field.
 * Includes slight diffusion and dissipation for visual stability.
 * Refactored for Phase 3: Uses ComputeContext for agnostic memory access.
 */
export class LBMSmokeKernel implements IKernel {
    public readonly metadata = {
        roles: {
            source: 'smoke',
            destination: 'smoke',
            auxiliary: ['vx', 'vy']
        }
    };

    public execute(
        views: Float32Array[],
        context: ComputeContext
    ): void {
        const { nx, ny, padding, pNx, scheme, indices } = context;

        const smokeInIdx = indices['smoke'].read;
        const smokeOutIdx = indices['smoke'].write;
        const vxIdx = indices['vx'].write; // Use the evolved velocity
        const vyIdx = indices['vy'].write;

        const sIn = views[smokeInIdx];
        const sOut = views[smokeOutIdx];
        const vx = views[vxIdx];
        const vy = views[vyIdx];

        const dissipation = (scheme.params?.dissipation as number) || 0.9995;
        const diffAlpha = (scheme.params?.diffusion as number) || 0.005;

        for (let py = padding; py < ny + padding; py++) {
            for (let px = padding; px < nx + padding; px++) {
                const idx = py * pNx + px;

                const ux = vx[idx];
                const uy = vy[idx];

                // Back-tracing position
                const sx = px - ux;
                const sy = py - uy;

                // Bilinear Interpolation
                const x0 = Math.floor(sx);
                const y0 = Math.floor(sy);
                const x1 = x0 + 1;
                const y1 = y0 + 1;
                const fx = sx - x0;
                const fy = sy - y0;

                let sample = 0;
                // Bounds check (including padding)
                if (x0 >= 0 && x1 < pNx && y0 >= 0 && y1 < (ny + 2 * padding)) {
                    const row0 = y0 * pNx;
                    const row1 = y1 * pNx;
                    const v00 = sIn[row0 + x0];
                    const v10 = sIn[row0 + x1];
                    const v01 = sIn[row1 + x0];
                    const v11 = sIn[row1 + x1];
                    sample = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
                }

                // Slight neighborhood average for diffusion
                const avg = (sIn[idx - 1] + sIn[idx + 1] + sIn[idx - pNx] + sIn[idx + pNx]) * 0.25;

                sOut[idx] = (sample * (1 - diffAlpha) + avg * diffAlpha) * dissipation;
            }
        }
    }
}
