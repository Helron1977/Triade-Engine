import { IKernel } from './IKernel';
import { ComputeContext } from './ComputeContext';

/**
 * LBMD2Q9Kernel
 * Implements the core Lattice Boltzmann D2Q9 step:
 * 1. Pull-Streaming (from neighbors, including ghost cells)
 * 2. Collision (BGK relaxation)
 * 3. Obstacle Bounce-Back
 * Refactored for Phase 3: Uses ComputeContext for agnostic memory access.
 */
export class LBMD2Q9Kernel implements IKernel {
    public readonly metadata = {
        roles: {
            source: 'f0',
            destination: 'f0',
            obstacles: 'obstacles',
            auxiliary: ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
        }
    };

    private readonly cx = [0, 1, 0, -1, 0, 1, -1, -1, 1]; // E, N, W, S, NE, NW, SW, SE
    private readonly cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    private readonly w = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];
    private readonly opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];

    public execute(
        views: Float32Array[],
        context: ComputeContext
    ): void {
        const { nx, ny, padding, pNx, scheme, indices } = context;

        const omega = (scheme.params?.omega as number) || 1.75;
        const om_1 = 1.0 - omega;

        // Face indices for f0..f8
        const fIdxs = [
            indices['f0'], indices['f1'], indices['f2'], indices['f3'], indices['f4'],
            indices['f5'], indices['f6'], indices['f7'], indices['f8']
        ];

        const obsIdx = indices['obstacles']?.read ?? -1;
        const obstacles = obsIdx !== -1 ? views[obsIdx] : null;

        const f_in = fIdxs.map(idx => views[idx.read]);
        const f_out = fIdxs.map(idx => views[idx.write]);

        for (let py = padding; py < ny + padding; py++) {
            for (let px = padding; px < nx + padding; px++) {
                const idx = py * pNx + px;

                // 1. Check for obstacles
                if (obstacles && obstacles[idx] > 0.5) {
                    for (let k = 0; k < 9; k++) f_out[k][idx] = this.w[k];
                    continue;
                }

                let rho = 0;
                let ux = 0;
                let uy = 0;

                const f_temp = new Float32Array(9);

                // 2. Pull Streaming
                for (let k = 0; k < 9; k++) {
                    const nx_k = px - this.cx[k];
                    const ny_k = py - this.cy[k];
                    const n_idx = ny_k * pNx + nx_k;

                    // Simple Bounce-Back if neighbor is obstacle
                    if (obstacles && obstacles[n_idx] > 0.5) {
                        f_temp[k] = f_in[this.opp[k]][idx];
                    } else {
                        f_temp[k] = f_in[k][n_idx];
                    }

                    rho += f_temp[k];
                    ux += this.cx[k] * f_temp[k];
                    uy += this.cy[k] * f_temp[k];
                }

                if (rho > 0) {
                    ux /= rho;
                    uy /= rho;
                }

                // 3. Collision (BGK)
                const u_sq_15 = 1.5 * (ux * ux + uy * uy);
                for (let k = 0; k < 9; k++) {
                    const cu = 3.0 * (this.cx[k] * ux + this.cy[k] * uy);
                    const feq = this.w[k] * rho * (1.0 + cu + 0.5 * cu * cu - u_sq_15);
                    f_out[k][idx] = f_temp[k] * om_1 + feq * omega;

                    // Stability Safeguard
                    if (!Number.isFinite(f_out[k][idx])) f_out[k][idx] = this.w[k];
                }
            }
        }
    }
}
