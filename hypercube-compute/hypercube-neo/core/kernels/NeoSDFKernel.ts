import { IKernel } from './IKernel';
import { ComputeContext } from './ComputeContext';

/**
 * NeoSDFKernel: Seed propagation for Signed Distance Fields (Jump Flooding-like).
 * Refactored for Phase 3: Uses ComputeContext for agnostic memory access.
 */
export class NeoSDFKernel implements IKernel {
    // Note: SDF is specialized and uses multiple faces based on scheme.source
    // We don't use the full KernelBinder here to maintain its dynamic dual-face logic
    public readonly metadata = {
        roles: {
            source: 'any',
            destination: 'any'
        }
    };

    public execute(
        views: Float32Array[],
        context: ComputeContext
    ): void {
        const { nx, ny, padding, pNx, pNy, scheme, indices, chunk, gridConfig } = context;

        const baseName = scheme.source;
        const uReadX = views[indices[baseName + '_x'].read];
        const uReadY = views[indices[baseName + '_y'].read];
        const uWriteX = views[indices[baseName + '_x'].write];
        const uWriteY = views[indices[baseName + '_y'].write];
        const obstacles = indices['obstacles'] ? views[indices['obstacles'].read] : null;

        // Correct World Offset Calculation for Phase 3
        let worldX0 = 0;
        let worldY0 = 0;
        const chunksList = context.params.chunksList;
        if (chunksList) {
            for (const c of chunksList) {
                if (c.y === chunk.y && c.z === chunk.z && c.x < chunk.x) worldX0 += c.localDimensions.nx;
                if (c.x === chunk.x && c.z === chunk.z && c.y < chunk.y) worldY0 += c.localDimensions.ny;
            }
        }

        const distSq = (x1: number, y1: number, x2: number, y2: number) => {
            if (x2 < -9000 || y2 < -9000) return 999999999;
            const dx = x1 - x2;
            const dy = y1 - y2;
            return dx * dx + dy * dy;
        };

        for (let py = padding; py < ny + padding; py++) {
            for (let px = padding; px < nx + padding; px++) {
                const i = py * pNx + px;

                if (obstacles && obstacles[i] > 0.5) {
                    uWriteX[i] = -10000;
                    uWriteY[i] = -10000;
                    continue;
                }

                const gX = worldX0 + (px - padding);
                const gY = worldY0 + (py - padding);

                let bestX = uReadX[i];
                let bestY = uReadY[i];
                let bestDist = distSq(gX, gY, bestX, bestY);

                for (let dy = -1; dy <= 1; dy++) {
                    const rowOffset = (py + dy) * pNx;
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const ni = rowOffset + (px + dx);
                        if (obstacles && obstacles[ni] > 0.5) continue;

                        const seedX = uReadX[ni];
                        const seedY = uReadY[ni];
                        const d = distSq(gX, gY, seedX, seedY);

                        if (d < bestDist) {
                            bestDist = d;
                            bestX = seedX;
                            bestY = seedY;
                        }
                    }
                }

                uWriteX[i] = bestX;
                uWriteY[i] = bestY;
            }
        }
    }
}
