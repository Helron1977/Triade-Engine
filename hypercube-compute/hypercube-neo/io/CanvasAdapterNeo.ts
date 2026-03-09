import { NeoEngineProxy } from '../core/NeoEngineProxy';
import { VisualRegistry } from './VisualRegistry';

export interface RenderOptions {
    faceIndex: number;
    colormap?: string;
    minVal?: number;
    maxVal?: number;
    obstaclesFace?: number;
    vorticityFace?: number;
    sliceZ?: number;
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

        // Virtual Dimensions (excluding ghost cells)
        const vnx = Math.floor(totalW / nChunksX);
        const vny = Math.floor(totalH / nChunksY);

        // Colormap configuration
        const colormap = options.colormap || 'arctic';
        const minV = options.minVal ?? 0;
        const maxV = options.maxVal ?? 1;
        const invRange = 1.0 / (maxV - minV || 1.0);

        const hasObs = options.obstaclesFace !== undefined;
        const hasVort = options.vorticityFace !== undefined;

        // Iterate through chunks and "tile" them into the global imageData
        for (let cy = 0; cy < nChunksY; cy++) {
            for (let cx = 0; cx < nChunksX; cx++) {
                const chunk = neo.vGrid.chunks.find((c: any) => c.x === cx && c.y === cy);
                if (!chunk) continue;

                const views = neo.mBuffer.getChunkViews(chunk.id);
                const faces = views.faces;
                const data = faces[options.faceIndex];
                const obsData = hasObs ? faces[options.obstaclesFace!] : null;
                const vortData = hasVort ? faces[options.vorticityFace!] : null;

                const nxPhys = vnx + 2;
                const nyPhys = vny + 2;

                const worldXOffset = cx * vnx;
                const worldYOffset = cy * vny;

                for (let ly = 1; ly < nyPhys - 1; ly++) {
                    const worldY = worldYOffset + (ly - 1);
                    const dstRowOffset = worldY * totalW;
                    const srcRowOffset = ly * nxPhys;

                    for (let lx = 1; lx < nxPhys - 1; lx++) {
                        const srcIdx = srcRowOffset + lx;
                        const worldX = worldXOffset + (lx - 1);
                        const dstIdx = dstRowOffset + worldX;

                        // 1. OBSTACLES (Static Priority)
                        if (obsData && obsData[srcIdx] > 0.9) {
                            pixelData[dstIdx] = 0xff282828; // ABGR: 255, 40, 40, 40
                            continue;
                        }

                        const val = data[srcIdx];
                        let r = 180, g = 220, b = 255; // Base color (Arctic background)

                        if (colormap === 'arctic') {
                            const s = Math.max(0, Math.min(1.0, (val - minV) * invRange));

                            // Smoke/Density: Blend to Navy (15, 30, 80)
                            // Replaced Math.pow(s, 0.35) with faster linear approximation for 60fps target
                            const tS = s * (2.0 - s);
                            r = r * (1 - tS) + 15 * tS;
                            g = g * (1 - tS) + 30 * tS;
                            b = b * (1 - tS) + 80 * tS;

                            // Vorticity: Red Highlights
                            if (vortData) {
                                const vMag = Math.min(1.0, Math.abs(vortData[srcIdx]) * 120.0);
                                if (vMag > 0.05) {
                                    const tC = Math.min(1.0, (vMag - 0.05) * 1.5);
                                    r = r * (1 - tC) + 255 * tC;
                                    g = g * (1 - tC);
                                    b = b * (1 - tC);
                                }
                            }
                        } else {
                            const gray = Math.floor(Math.max(0, Math.min(1.0, (val - minV) * invRange)) * 255);
                            r = g = b = gray;
                        }

                        // Write pixel (ABGR format for Little Endian architecture)
                        pixelData[dstIdx] = (255 << 24) | (b << 16) | (g << 8) | r;
                    }
                }
            }
        }

        ctx.putImageData(imageData, 0, 0);
    }
}
