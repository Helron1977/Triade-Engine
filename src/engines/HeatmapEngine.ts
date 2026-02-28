import type { ITriadeEngine } from './ITriadeEngine';

export class HeatmapEngine implements ITriadeEngine {
    public readonly name = "Heatmap (O1 Spatial Convolution)";
    public radius: number;
    public weight: number;

    /**
     * @param radius Rayon d'influence en cellules
     * @param weight Coefficient multiplicateur à l'arrivée
     */
    constructor(radius: number = 10, weight: number = 1.0) {
        this.radius = radius;
        this.weight = weight;
    }

    /**
     * Exécute le Summed Area Table Algorithm (Face 5) suivi 
     * d'un Box Filter O(1) vers la Synthèse (Face 3).
     */
    compute(faces: Float32Array[], mapSize: number): void {
        const face2 = faces[1]; // Contexte Binaire d'entrée
        const face3 = faces[2]; // Synthèse de Diffusion
        const face5 = faces[4]; // Cheat-code O(1) SAT

        // 1. O(N) : Génération Cristallisée (Integral Image)
        for (let y = 0; y < mapSize; y++) {
            for (let x = 0; x < mapSize; x++) {
                const idx = y * mapSize + x;
                const val = face2[idx];
                const top = y > 0 ? face5[(y - 1) * mapSize + x] : 0;
                const left = x > 0 ? face5[y * mapSize + (x - 1)] : 0;
                const topLeft = (y > 0 && x > 0) ? face5[(y - 1) * mapSize + (x - 1)] : 0;

                face5[idx] = val + top + left - topLeft;
            }
        }

        // 2. O(N) : Extraction d'Influence Indépendante du Rayon
        for (let y = 0; y < mapSize; y++) {
            for (let x = 0; x < mapSize; x++) {
                // Clamping (Borne Map) rapide
                const minX = Math.max(0, x - this.radius);
                const minY = Math.max(0, y - this.radius);
                const maxX = Math.min(mapSize - 1, x + this.radius);
                const maxY = Math.min(mapSize - 1, y + this.radius);

                // Récupération O(1) des Opcodes d'angles
                const A = (minX > 0 && minY > 0) ? face5[(minY - 1) * mapSize + (minX - 1)] : 0;
                const B = (minY > 0) ? face5[(minY - 1) * mapSize + maxX] : 0;
                const C = (minX > 0) ? face5[maxY * mapSize + (minX - 1)] : 0;
                const D = face5[maxY * mapSize + maxX];

                // Différence des coins
                const sum = D - B - C + A;
                face3[y * mapSize + x] = sum * this.weight;
            }
        }
    }
}
