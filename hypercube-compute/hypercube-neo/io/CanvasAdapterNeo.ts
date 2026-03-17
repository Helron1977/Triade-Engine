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


        // Resolve logical face indices → physical slots via parityManager
        // (same convention as WebGpuRendererNeo: faceIndex is a descriptor.faces index)
        const descriptor = (neo.vGrid as any).dataContract.descriptor;
        const getPhysicalSlot = (logicalIdx: number | undefined): number | undefined => {
            if (logicalIdx === undefined || logicalIdx >= descriptor.faces.length) return undefined;
            const faceName = descriptor.faces[logicalIdx].name;
            return neo.parityManager.getFaceIndices(faceName).read;
        };
        const physFaceIdx = getPhysicalSlot(options.faceIndex);
        const physObsIdx  = getPhysicalSlot(options.obstaclesFace);
        const physVortIdx = getPhysicalSlot(options.vorticityFace); // Keep for compatibility
        const auxIndices = (options.auxiliaryFaces || []).map(idx => getPhysicalSlot(idx));

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
            // ... (rest of getting data arrays)
            const data = physFaceIdx !== undefined ? faces[physFaceIdx] : null;
            const criteria = options.criteria || [];
            const criteriaFaces = criteria.map(c => {
                const phys = getPhysicalSlot(c.faceIndex);
                return phys !== undefined ? faces[phys] : new Float32Array(0);
            });
            const criteriaSDF = options.criteriaSDF || [];
            const criteriaSDFFaces = criteriaSDF.map(c => ({
                x: getPhysicalSlot(c.xFace) !== undefined ? faces[getPhysicalSlot(c.xFace)!] : new Float32Array(0),
                y: getPhysicalSlot(c.yFace) !== undefined ? faces[getPhysicalSlot(c.yFace)!] : new Float32Array(0)
            }));
            const obsData  = physObsIdx  !== undefined ? faces[physObsIdx]  : null;
            const vortData = physVortIdx !== undefined ? faces[physVortIdx] : null;

            // Calculate world offsets by summing preceding chunks
            let worldXOffset = 0;
            let worldYOffset = 0;
            for (const c of neo.vGrid.chunks) {
                if (c.y === chunk.y && c.z === chunk.z && c.x < chunk.x) worldXOffset += c.localDimensions.nx;
                if (c.x === chunk.x && c.z === chunk.z && c.y < chunk.y) worldYOffset += c.localDimensions.ny;
            }

            const nx = chunk.localDimensions.nx;
            const ny = chunk.localDimensions.ny;

            for (let ly = padding; ly < ny + padding; ly++) {
                const worldY = worldYOffset + (ly - padding);
                const dstRowOffset = worldY * totalW;
                const srcRowOffset = ly * nxPhys;

                for (let lx = padding; lx < nx + padding; lx++) {
                    const srcIdx = srcRowOffset + lx;
                    const worldX = worldXOffset + (lx - padding);
                    const dstIdx = dstRowOffset + worldX;

                        // 1. OBSTACLES (Static Priority)
                        if (obsData && obsData[srcIdx] > 0.9) {
                            if (colormap === 'spatial-decision') {
                                pixelData[dstIdx] = 0x00000000; // Transparent for Leaflet
                            } else {
                                pixelData[dstIdx] = 0xff282828; // ABGR: 255, 40, 40, 40
                            }
                            continue;
                        }

                        const val = data ? data[srcIdx] : 0.0;
                        
                        // Use Registry-based coloring (Nuclear Refactor Priority 1)
                        const coloringCtx = {
                            minV, maxV, invRange,
                            chunkFaces: faces,
                            srcIdx, worldX, worldY,
                            vorticityFace: physVortIdx,
                            auxiliaryFaces: auxIndices,
                            options: {
                                ...options,
                                resolvedCriteriaFaces: criteriaFaces,
                                resolvedCriteriaSDFFaces: criteriaSDFFaces
                            }
                        };
                        
                        // Select colormap variant (heatmap-criteria vs heatmap)
                        let activeColormap = colormap;
                        if (colormap === 'heatmap' && criteria.length > 0) activeColormap = 'heatmap-criteria';

                        pixelData[dstIdx] = ColormapRegistry.get(activeColormap)(val, coloringCtx);
                }
            }
        }

        ctx.putImageData(imageData, 0, 0);
    }
}
