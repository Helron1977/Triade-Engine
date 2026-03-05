export class CanvasAdapter {
    private ctx: CanvasRenderingContext2D;
    private canvas: HTMLCanvasElement;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        const context = canvas.getContext('2d', { alpha: false });
        if (!context) throw new Error("Could not get 2D context from canvas");
        this.ctx = context;
    }

    /**
     * Renders a multi-chunk grid of faces into the single large canvas.
     */
    public renderFromFaces(
        gridFaces: Float32Array[][][], // [row][col][faceIndex]
        nx: number,
        ny: number,
        cols: number,
        rows: number,
        options: {
            faceIndex: number,
            colormap: 'grayscale' | 'heatmap' | 'vorticity' | 'ocean',
            minVal?: number,
            maxVal?: number,
            sliceZ?: number,
            obstaclesFace?: number
        }
    ) {
        const sliceZ = options.sliceZ || 0;
        const totalW = nx * cols;
        const totalH = ny * rows;

        const imgData = this.ctx.getImageData(0, 0, totalW, totalH);
        const pixelData = imgData.data;

        const faceIdx = options.faceIndex;
        const obsIdx = options.obstaclesFace;
        const minV = options.minVal ?? 0;
        const maxV = options.maxVal ?? 1;
        const range = (maxV - minV) || 0.0001;

        for (let gy = 0; gy < rows; gy++) {
            for (let gx = 0; gx < cols; gx++) {
                const faces = gridFaces[gy][gx];
                const data = faces[faceIdx];
                const obs = obsIdx !== undefined ? faces[obsIdx] : null;

                const zOff = sliceZ * ny * nx;

                for (let ly = 0; ly < ny; ly++) {
                    const py = gy * ny + ly;
                    for (let lx = 0; lx < nx; lx++) {
                        const px = gx * nx + lx;
                        const srcIdx = zOff + ly * nx + lx;
                        const dstIdx = (py * totalW + px) * 4;

                        // Obstacle check
                        if (obs && obs[srcIdx] > 0.5) {
                            pixelData[dstIdx] = 50;
                            pixelData[dstIdx + 1] = 50;
                            pixelData[dstIdx + 2] = 50;
                            pixelData[dstIdx + 3] = 255;
                            continue;
                        }

                        const val = (data[srcIdx] - minV) / range;
                        const v = Math.max(0, Math.min(1, val));

                        if (options.colormap === 'heatmap') {
                            pixelData[dstIdx] = v * 255;
                            pixelData[dstIdx + 1] = v > 0.5 ? (v - 0.5) * 510 : 0;
                            pixelData[dstIdx + 2] = v * 50;
                        } else if (options.colormap === 'vorticity') {
                            // -1 to 1 range mapping
                            const norm = (data[srcIdx] - minV) / (maxV - minV);
                            const vn = Math.max(0, Math.min(1, norm));
                            pixelData[dstIdx] = vn > 0.5 ? (vn - 0.5) * 510 : 0;
                            pixelData[dstIdx + 1] = vn * 255;
                            pixelData[dstIdx + 2] = vn < 0.5 ? (0.5 - vn) * 510 : 0;
                        } else {
                            const c = v * 255;
                            pixelData[dstIdx] = c;
                            pixelData[dstIdx + 1] = c;
                            pixelData[dstIdx + 2] = c;
                        }

                        pixelData[dstIdx + 3] = 255;
                    }
                }
            }
        }

        this.ctx.putImageData(imgData, 0, 0);
    }
}
