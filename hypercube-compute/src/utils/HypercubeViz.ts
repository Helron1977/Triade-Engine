import { HypercubeChunk } from "../core/HypercubeChunk";

/**
 * HypercubeViz
 * Collection de helpers pour la visualisation et le débogage de données volumétriques.
 */
export class HypercubeViz {
    /**
     * Extrait une tranche 2D (Slice Z) d'une face spécifique à la profondeur lz.
     * @returns Un Float32Array de taille nx * ny.
     */
    static getSliceZ(chunk: HypercubeChunk, faceIndex: number, lz: number): Float32Array {
        if (lz < 0 || lz >= chunk.nz) {
            throw new Error(`Slice index ${lz} out of bounds (0-${chunk.nz - 1})`);
        }

        const nx = chunk.nx;
        const ny = chunk.ny;
        const face = chunk.faces[faceIndex];
        const sliceSize = nx * ny;
        const offset = lz * sliceSize;

        // On retourne une sous-vue (subarray) pour éviter la copie si possible, 
        // ou on peut copier si on veut une extraction isolée.
        return face.slice(offset, offset + sliceSize);
    }

    /**
     * Projection Isométrique simplifiée (Top-down accumulation).
     * Projette les valeurs 3D sur un plan 2D en prenant la valeur max ou la moyenne.
     * Utile pour avoir une vue d'ensemble sans moteur 3D.
     */
    static projectIso(chunk: HypercubeChunk, faceIndex: number, mode: 'max' | 'average' = 'max'): Float32Array {
        const nx = chunk.nx;
        const ny = chunk.ny;
        const nz = chunk.nz;
        const face = chunk.faces[faceIndex];
        const result = new Float32Array(nx * ny);

        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                let acc = 0;
                let maxVal = -Infinity;

                for (let z = 0; z < nz; z++) {
                    const val = face[(z * ny * nx) + (y * nx) + x];
                    acc += val;
                    if (val > maxVal) maxVal = val;
                }

                const idx = y * nx + x;
                result[idx] = (mode === 'max') ? maxVal : acc / nz;
            }
        }

        return result;
    }

    /**
     * Exporte une face volumétrique complète dans un format binaire brut (Buffer).
     * Utile pour importer dans des outils comme ImageJ, Slicer, ou des voxels viewers.
     */
    static exportVolume(chunk: HypercubeChunk, faceIndex: number): Uint8Array {
        const face = chunk.faces[faceIndex];
        // On convertit les Float32 en Uint8 (0.0-1.0 -> 0-255) pour la portabilité standard
        const buffer = new Uint8Array(face.length);
        for (let i = 0; i < face.length; i++) {
            buffer[i] = Math.max(0, Math.min(255, face[i] * 255));
        }
        return buffer;
    }

    /**
     * Injection d'une sphère de densité dans le volume.
     */
    static injectSphere(chunk: HypercubeChunk, faceIndex: number, cx: number, cy: number, cz: number, radius: number, value: number = 1.0): void {
        const face = chunk.faces[faceIndex];
        const { nx, ny, nz } = chunk;
        const r2 = radius * radius;

        for (let z = 0; z < nz; z++) {
            const zOff = z * ny * nx;
            const dz = z - cz;
            for (let y = 0; y < ny; y++) {
                const yOff = y * nx;
                const dy = y - cy;
                for (let x = 0; x < nx; x++) {
                    const dx = x - cx;
                    if (dx * dx + dy * dy + dz * dz <= r2) {
                        face[zOff + yOff + x] = value;
                    }
                }
            }
        }
    }

    /**
     * Injection d'un plan (slice) de densité constante.
     */
    static injectSlice(chunk: HypercubeChunk, faceIndex: number, axis: 'x' | 'y' | 'z', index: number, value: number = 1.0): void {
        const face = chunk.faces[faceIndex];
        const { nx, ny, nz } = chunk;

        for (let z = 0; z < nz; z++) {
            if (axis === 'z' && z !== index) continue;
            const zOff = z * ny * nx;
            for (let y = 0; y < ny; y++) {
                if (axis === 'y' && y !== index) continue;
                const yOff = y * nx;
                for (let x = 0; x < nx; x++) {
                    if (axis === 'x' && x !== index) continue;
                    face[zOff + yOff + x] = value;
                }
            }
        }
    }
    /**
     * Renders a 2D Float32Array face directly to an HTML5 Canvas.
     * Uses an optimized 32-bit pixel buffer and supports premium colormaps.
     */
    static renderToCanvas(
        canvas: HTMLCanvasElement,
        data: Float32Array,
        width: number,
        height: number,
        colors: 'green' | 'heat' | 'grayscale' | 'viridis' | 'plasma' | 'magma' | 'bipolar' = 'green',
        normalize: boolean = true
    ): void {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }

        const imgData = ctx.createImageData(width, height);
        const buf = new Uint32Array(imgData.data.buffer);

        let maxVal = 1.0;
        let minVal = 0.0;
        if (normalize) {
            maxVal = 0.0001;
            minVal = -0.0001;
            for (let i = 0; i < data.length; i++) {
                if (data[i] > maxVal) maxVal = data[i];
                if (data[i] < minVal) minVal = data[i];
            }
        }

        // Pour le mode bipolaire, on normalise par rapport à la plus grande valeur absolue
        const absMax = Math.max(Math.abs(maxVal), Math.abs(minVal));

        for (let i = 0; i < data.length; i++) {
            let v = data[i] / maxVal; // Normalized default behavior
            if (colors === 'bipolar') {
                v = data[i] / absMax; // -1.0 to 1.0 range
            }
            v = Math.max(-1, Math.min(1, v)); // Clamp between -1 and 1

            if (colors === 'bipolar') {
                if (v < 0) {
                    const t = -v; // negative: blue -> cyan
                    const r = 0;
                    const g = Math.floor(255 * t);
                    const b = 255;
                    buf[i] = 0xFF000000 | (b << 16) | (g << 8) | r;
                } else {
                    const t = v; // positive: white -> red
                    const r = 255;
                    const g = Math.floor(255 * (1 - t));
                    const b = Math.floor(255 * (1 - t));
                    buf[i] = 0xFF000000 | (b << 16) | (g << 8) | r;
                }
            } else if (colors === 'viridis') {
                const r = Math.floor(v * v * 255);
                const g = Math.floor(v * 255);
                const b = Math.floor((1 - v) * 128 + v * 32);
                buf[i] = 0xFF000000 | (b << 16) | (g << 8) | r;
            } else if (colors === 'plasma') {
                const r = Math.floor(v * 255);
                const g = Math.floor(Math.pow(v, 3) * 255);
                const b = Math.floor((1 - v) * 255);
                buf[i] = 0xFF000000 | (b << 16) | (g << 8) | r;
            } else if (colors === 'magma') {
                const r = Math.floor(v * 255);
                const g = Math.floor(Math.pow(v, 2) * 200);
                const b = Math.floor(Math.pow(v, 4) * 100);
                buf[i] = 0xFF000000 | (b << 16) | (g << 8) | r;
            } else if (colors === 'green') {
                const l = Math.floor(v * 255);
                buf[i] = 0xFF000000 | (0 << 16) | (l << 8) | 0;
            } else if (colors === 'heat') {
                const r = Math.floor(v * 255);
                const g = Math.floor(Math.max(0, v - 0.5) * 510);
                buf[i] = 0xFF000000 | (0 << 16) | (g << 8) | r;
            } else {
                const c = Math.floor(v * 255);
                buf[i] = 0xFF000000 | (c << 16) | (c << 8) | c;
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }

    /**
     * Extremely simple one-liner to see the state of a chunk's face.
     * Automatically extracts dimensions from the chunk.
     */
    static quickRender(
        canvas: HTMLCanvasElement,
        chunk: HypercubeChunk,
        faceIndex: number = 0,
        colormap: 'green' | 'viridis' | 'plasma' | 'magma' | 'bipolar' = 'viridis'
    ): void {
        const faceData = chunk.faces[faceIndex];
        this.renderToCanvas(canvas, faceData, chunk.nx, chunk.ny, colormap, true);
    }
}
