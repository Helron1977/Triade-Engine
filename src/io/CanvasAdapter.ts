export class CanvasAdapter {
    /**
     * Lit un Tenseur Plat (Face de Float32Array) et le peint sur un contexte Canvas Native.
     * Cette interface sépare la logique de rendu (UI) du moteur mathématique (Triade).
     */
    static renderFaceToCanvas(
        faceData: Float32Array,
        mapSize: number,
        ctx: CanvasRenderingContext2D,
        options: { colorScheme: 'heat' | 'grayscale', normalizeMax?: number } = { colorScheme: 'grayscale' }
    ) {
        const imgData = ctx.getImageData(0, 0, mapSize, mapSize);
        const data = imgData.data;

        // Auto-normalization si l'utilisateur ne connait pas le Max possible de sa matrice.
        let max = options.normalizeMax || 0.0001;
        if (!options.normalizeMax) {
            for (let i = 0; i < faceData.length; i++) {
                if (faceData[i] > max) max = faceData[i];
            }
        }

        // Colorisation O(N)
        for (let i = 0; i < faceData.length; i++) {
            const val = faceData[i] / max;
            const p = i * 4;

            if (options.colorScheme === 'heat') {
                data[p] = val * 255;                           // R
                data[p + 1] = (val > 0.5 ? (val - 0.5) * 510 : 0); // G (jaunit si forte intensité)
                data[p + 2] = val * 50;                          // B
                data[p + 3] = 255;                               // Alpha
            } else {
                const c = val * 255;
                data[p] = c; data[p + 1] = c; data[p + 2] = c;
                data[p + 3] = 255;
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }
}
