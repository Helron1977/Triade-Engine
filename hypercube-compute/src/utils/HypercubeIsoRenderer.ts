/**
 * HypercubeIsoRenderer
 * Special renderer for 2.5D Isometric volumetric views of multi-chunk grids.
 */
export class HypercubeIsoRenderer {
    private ctx: CanvasRenderingContext2D;
    private canvas: HTMLCanvasElement;
    private scale: number;

    constructor(canvas: HTMLCanvasElement, options?: any, scale: number = 4.0) {
        this.canvas = canvas;
        const context = canvas.getContext('2d', { alpha: false });
        if (!context) throw new Error("Could not get 2D context");
        this.ctx = context;
        this.scale = scale;
    }

    public clearAndSetup(r: number, g: number, b: number) {
        this.ctx.fillStyle = `rgb(${r},${g},${b})`;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * Renders a multi-chunk volume in 2.5D isometric view.
     * Logic: Bottom-to-top, Right-to-left painter's algorithm.
     */
    public renderMultiChunkVolume(
        gridFaces: Float32Array[][][],
        nx: number,
        ny: number,
        cols: number,
        rows: number,
        options: { densityFaceIndex: number, obstacleFaceIndex?: number }
    ) {
        const { densityFaceIndex: dfi, obstacleFaceIndex: ofi } = options;
        const scale = this.scale;

        // Isometric offsets
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2 + (ny * rows * scale * 0.2);

        // For each chunk
        for (let gy = 0; gy < rows; gy++) {
            for (let gx = 0; gx < cols; gx++) {
                const faces = gridFaces[gy][gx];
                const density = faces[dfi];
                const obs = ofi !== undefined ? faces[ofi] : null;

                // Base offset for this chunk in isometric space
                // Grid (gx, gy) -> Iso (X, Y)
                const chunkIsoX = (gx - gy) * (nx * scale * 0.866);
                const chunkIsoY = (gx + gy) * (nx * scale * 0.5);

                for (let ly = 0; ly < ny; ly += 2) { // Subsampled for speed
                    for (let lx = 0; lx < nx; lx += 2) {
                        const idx = ly * nx + lx;
                        const val = density[idx];
                        const isObs = obs ? obs[idx] > 0.5 : false;

                        if (val < 0.05 && !isObs) continue;

                        // Screen Position
                        const x = centerX + chunkIsoX + (lx - ly) * (scale * 0.866);
                        const y = centerY + chunkIsoY + (lx + ly) * (scale * 0.5);

                        // Pseudo-3D Height based on density
                        const h = isObs ? scale * 10 : val * scale * 5;

                        if (isObs) {
                            this.ctx.fillStyle = '#333';
                        } else {
                            const b = Math.floor(Math.min(255, val * 100));
                            const g = Math.floor(Math.min(255, val * 200));
                            this.ctx.fillStyle = `rgb(0, ${g}, ${200 + b})`;
                        }

                        this.ctx.fillRect(x, y - h, scale * 2, h || scale);
                    }
                }
            }
        }
    }
}
