import { IKernel } from './IKernel';
import { ComputeContext } from './ComputeContext';
import { KernelBinder } from './KernelBinder';

/**
 * NeoLifeKernel: Implementation of Conway's Game of Life (B3/S23).
 * Uses a 3x3 stencil on a ping-pong face.
 * Refactored for Phase 3: Uses ComputeContext for agnostic memory access.
 */
export class NeoLifeKernel implements IKernel {
    public readonly metadata = {
        roles: {
            source: 'cells',
            destination: 'cells'
        }
    };

    public execute(
        views: Float32Array[],
        context: ComputeContext
    ): void {
        const { nx, ny, padding, pNx, scheme, indices } = context;

        // Resolve views via Binder
        const bound = KernelBinder.bind(this, scheme, views, indices);
        const uRead = bound.source.read;
        const uWrite = bound.destination.write;

        for (let py = padding; py < ny + padding; py++) {
            for (let px = padding; px < nx + padding; px++) {
                const i = py * pNx + px;
                
                // Count 8 neighbors
                let neighbors = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    const rowOffset = (py + dy) * pNx;
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        if (uRead[rowOffset + (px + dx)] > 0.5) {
                            neighbors++;
                        }
                    }
                }

                const alive = uRead[i] > 0.5;
                if (alive) {
                    // Survival: 2 or 3 neighbors
                    uWrite[i] = (neighbors === 2 || neighbors === 3) ? 1.0 : 0.0;
                } else {
                    // Birth: exactly 3 neighbors
                    uWrite[i] = (neighbors === 3) ? 1.0 : 0.0;
                }
            }
        }
    }
}
