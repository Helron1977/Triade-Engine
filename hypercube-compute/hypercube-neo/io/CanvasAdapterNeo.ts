import { NeoEngineProxy } from '../core/NeoEngineProxy';
import { VisualRegistry } from './VisualRegistry';
import { ColormapRegistry } from './ColormapRegistry';

export interface RenderOptions {
    faceIndex: number;
    colormap?: string;
    minVal?: number;
    maxVal?: number;
    obstaclesFace?: number;
    vorticityFace?: number; // Deprecated, use auxiliaryFaces
    sliceZ?: number;
    criteria?: { faceIndex: number, weight: number, distanceThreshold?: number }[];
    criteriaSDF?: { xFace: number, yFace: number, weight: number, distanceThreshold: number }[];
    auxiliaryFaces?: number[]; // [0]=vorticity, [1]=vx, [2]=vy, etc.
}

/**
 * CanvasAdapterNeo: Neo-native rendering orchestrator.
 * Understands multi-chunk grids and assembles them into a single canvas.
 */
export class CanvasAdapterNeo {

    /**
     * Renders a NeoEngineProxy (multi-chunk) to a single canvas.
     */
    static render(neo: NeoEngineProxy, canvas: HTMLCanvasElement, options: RenderOptions): void {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dims = neo.vGrid.dimensions;
        const totalW = dims.nx;
        const totalH = dims.ny;

        // Ensure canvas size matches simulation resolution
        if (canvas.width !== totalW || canvas.height !== totalH) {
            canvas.width = totalW;
            canvas.height = totalH;
        }

        const imageData = ctx.createImageData(totalW, totalH);
        const pixelData = new Uint32Array(imageData.data.buffer);

        const chunkLayout = neo.vGrid.chunkLayout;
        const nChunksX = chunkLayout.x;
        const nChunksY = chunkLayout.y;

        // Note: vnx/vny are no longer uniform. We calculate them per-chunk from localDimensions.

        // Colormap configuration
        const colormap = options.colormap || 'arctic';
        const minV = options.minVal ?? 0;
        const maxV = options.maxVal ?? 1;
        const invRange = 1.0 / (maxV - minV || 1.0);


        const descriptor = (neo.vGrid as any).dataContract.descriptor;
        const getPhysicalSlot = (logicalIdx: number | undefined): number | undefined => {
            if (logicalIdx === undefined || logicalIdx >= descriptor.faces.length) return undefined;
            const faceName = descriptor.faces[logicalIdx].name;
            return neo.parityManager.getFaceIndices(faceName).read;
        };

        const physFaceIdx = getPhysicalSlot(options.faceIndex);
        const physObsIdx  = getPhysicalSlot(options.obstaclesFace);
        const physVortIdx = getPhysicalSlot(options.vorticityFace);
        const auxIndices  = (options.auxiliaryFaces || []).map(idx => getPhysicalSlot(idx));

        const criteria = options.criteria || [];
        const criteriaPhysIndices = criteria.map(c => getPhysicalSlot(c.faceIndex));
        
        const criteriaSDF = options.criteriaSDF || [];
        const criteriaSDFPhysSets = criteriaSDF.map(c => ({
            x: getPhysicalSlot(c.xFace),
            y: getPhysicalSlot(c.yFace)
        }));

        // 1. Pooled context to avoid 250k+ allocations/frame
        const coloringCtx: any = {
            minV, maxV, invRange,
            vorticityFace: physVortIdx,
            auxiliaryFaces: auxIndices,
            options: { ...options }
        };
        const colormapFn = ColormapRegistry.get(colormap === 'heatmap' && criteria.length > 0 ? 'heatmap-criteria' : colormap);

        // Pre-calculate max dimensions for correct stride logic
        let maxNx = 0;
        for (const c of neo.vGrid.chunks) {
            maxNx = Math.max(maxNx, c.localDimensions.nx);
        }
        const padding = descriptor.requirements.ghostCells;
        const nxPhys = maxNx + 2 * padding;

        // Iterate through chunks and "tile" them into the global imageData
        for (const chunk of neo.vGrid.chunks) {
            const faces = neo.bridge.getChunkViews(chunk.id);
            const data = physFaceIdx !== undefined ? faces[physFaceIdx] : null;
            const obsData = physObsIdx !== undefined ? faces[physObsIdx] : null;

            // Resolve criteria views for this chunk (pre-loop)
            const resolvedCriteriaFaces = criteriaPhysIndices.map(pIdx => pIdx !== undefined ? faces[pIdx] : null);
            const resolvedCriteriaSDFFaces = criteriaSDFPhysSets.map(set => ({
                x: set.x !== undefined ? faces[set.x] : null,
                y: set.y !== undefined ? faces[set.y] : null
            }));

            coloringCtx.chunkFaces = faces;
            coloringCtx.options.resolvedCriteriaFaces = resolvedCriteriaFaces;
            coloringCtx.options.resolvedCriteriaSDFFaces = resolvedCriteriaSDFFaces;

            // Calculate world offsets ONCE per chunk
            const vChunkAny = chunk as any;
            if (vChunkAny._worldXOffset === undefined) {
                let wx = 0, wy = 0;
                const cList = (neo.vGrid as any).config?.chunksList || neo.vGrid.chunks;
                for (const c of cList) {
                    if (c.y === chunk.y && c.z === chunk.z && c.x < chunk.x) wx += c.localDimensions.nx;
                    if (c.x === chunk.x && c.z === chunk.z && c.y < chunk.y) wy += c.localDimensions.ny;
                }
                vChunkAny._worldXOffset = wx;
                vChunkAny._worldYOffset = wy;
            }

            const worldXOffset = vChunkAny._worldXOffset;
            const worldYOffset = vChunkAny._worldYOffset;
            const nx = chunk.localDimensions.nx;
            const ny = chunk.localDimensions.ny;

            for (let ly = padding; ly < ny + padding; ly++) {
                const worldY = worldYOffset + (ly - padding);
                const dstRowOffset = worldY * totalW;
                const srcRowOffset = ly * nxPhys;

                coloringCtx.worldY = worldY;

                for (let lx = padding; lx < nx + padding; lx++) {
                    const srcIdx = srcRowOffset + lx;
                    const worldX = worldXOffset + (lx - padding);
                    const dstIdx = dstRowOffset + worldX;

                    if (obsData && obsData[srcIdx] > 0.9) {
                        pixelData[dstIdx] = (colormap === 'spatial-decision') ? 0x00000000 : 0xff282828;
                        continue;
                    }

                    const val = data ? data[srcIdx] : 0.0;
                    coloringCtx.srcIdx = srcIdx;
                    coloringCtx.worldX = worldX;

                    pixelData[dstIdx] = colormapFn(val, coloringCtx);
                }
            }
        }

        ctx.putImageData(imageData, 0, 0);
    }
}
