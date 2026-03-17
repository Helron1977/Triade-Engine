import { IKernel } from './IKernel';
import { ComputeContext } from './ComputeContext';
import { KernelBinder } from './KernelBinder';

/**
 * NeoPathKernel: Wavefront propagation for pathfinding.
 * Similar to Dijkstra or Breadth-First Search on a grid.
 * Refactored for Phase 3: Uses ComputeContext for agnostic memory access.
 */
export class NeoPathKernel implements IKernel {
    public readonly metadata = {
        roles: {
            source: 'distance',
            destination: 'distance',
            obstacles: 'obstacles'
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
        const obstacles = bound.obstacles;

        for (let py = padding; py < ny + padding; py++) {
            for (let px = padding; px < nx + padding; px++) {
                const i = py * pNx + px;

                if (obstacles && obstacles[i] > 0.5) {
                    uWrite[i] = 1e9; // Wall
                    continue;
                }

                let minNeighborDist = uRead[i];

                // Check 4 cardinal neighbors
                const neighbors = [
                    (py - 1) * pNx + px,
                    (py + 1) * pNx + px,
                    py * pNx + (px - 1),
                    py * pNx + (px + 1)
                ];

                for (const ni of neighbors) {
                    const d = uRead[ni] + 1.0;
                    if (d < minNeighborDist) {
                        minNeighborDist = d;
                    }
                }

                uWrite[i] = minNeighborDist;
            }
        }
    }
}
