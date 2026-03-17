import { IKernel } from './IKernel';
import { NumericalScheme } from '../types';
import { VirtualChunk } from '../topology/GridAbstractions';
import { KernelBinder } from './KernelBinder';
import { ComputeContext } from './ComputeContext';

/**
 * NeoHeatmapKernel
 * Implementation of 2D Heat Diffusion.
 * Refactored for Phase 3: Uses ComputeContext for agnostic memory access.
 */
export class NeoHeatmapKernel implements IKernel {
    public static readonly DEFAULT_DIFFUSION = 0.25;
    public static readonly DEFAULT_DECAY = 0.99;

    public readonly metadata = {
        roles: {
            source: 'temp',
            destination: 'temp',
            obstacles: 'obstacles',
            auxiliary: ['injection']
        }
    };

    public execute(
        views: Float32Array[],
        context: ComputeContext
    ): void {
        const { nx, ny, padding, pNx, scheme, indices, gridConfig, chunk } = context;

        const diffusionRate = (scheme.params?.diffusion_rate as number) || NeoHeatmapKernel.DEFAULT_DIFFUSION;
        const decayFactor = (scheme.params?.decay_factor as number) || NeoHeatmapKernel.DEFAULT_DECAY;

        // Resolve views via Binder (Nuclear Refactor Priority 2)
        const bound = KernelBinder.bind(this, scheme, views, indices);
        const uRead = bound.source.read;
        const uWrite = bound.destination.write;
        const obstacles = bound.obstacles;
        const injection = bound.auxiliary[0]?.read || null;

        for (let py = padding; py < ny + padding; py++) {
            for (let px = padding; px < nx + padding; px++) {
                const i = py * pNx + px;

                // 1. Is it a Wall?
                let isWall = (obstacles && obstacles[i] > 0.5);
                let injectionValue = (injection && injection[i] > 0) ? injection[i] : -1.0;

                // Dynamic objects support (matches GPU behavior)
                if (gridConfig.objects && gridConfig.objects.length > 0) {
                    let worldX0 = 0;
                    let worldY0 = 0;
                    const chunksList = context.params.chunksList;
                    if (chunksList) {
                        for (const c of chunksList) {
                            if (c.y === chunk.y && c.z === chunk.z && c.x < chunk.x) worldX0 += c.localDimensions.nx;
                            if (c.x === chunk.x && c.z === chunk.z && c.y < chunk.y) worldY0 += c.localDimensions.ny;
                        }
                    }

                    const worldX = worldX0 + (px - padding);
                    const worldY = worldY0 + (py - padding);

                    for (const obj of gridConfig.objects) {
                        if (obj.isBaked === true || obj.renderOnly === true) continue;
                        
                        let inObj = false;
                        if (obj.type === 'circle') {
                            const r = obj.dimensions.w * 0.5;
                            const dx = worldX - (obj.position.x + r);
                            const dy = worldY - (obj.position.y + r);
                            if (dx*dx + dy*dy <= r*r) inObj = true;
                        } else if (obj.type === 'rect') {
                            if (worldX >= obj.position.x && worldX < obj.position.x + obj.dimensions.w &&
                                worldY >= obj.position.y && worldY < obj.position.y + obj.dimensions.h) {
                                inObj = true;
                            }
                        }

                        if (inObj) {
                            if (obj.properties.obstacles > 0.5 || obj.properties.isObstacle > 0.5) isWall = true;
                            const temp = obj.properties.temp ?? obj.properties.temperature;
                            if (temp !== undefined) injectionValue = temp;
                        }
                    }
                }

                if (isWall) {
                    uWrite[i] = 0;
                    continue;
                }

                if (injectionValue >= 0) {
                    uWrite[i] = injectionValue;
                    continue;
                }

                // 2. Diffusion (Laplacian Operator)
                const center = uRead[i];
                const laplacian = (
                    uRead[i - 1] + uRead[i + 1] +
                    uRead[i - pNx] + uRead[i + pNx]
                ) - 4 * center;

                uWrite[i] = (center + diffusionRate * laplacian) * decayFactor;
            }
        }
    }
}
