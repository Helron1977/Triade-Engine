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

                const sum = D - B - C + A;
                face3[y * mapSize + x] = sum * this.weight;
            }
        }
    }

    /**
     * @WebGPU
     * Code WGSL statique pour décharger le Box Filter SAT O(N) sur le GPU.
     * Binding 0: Face 2 (Input Binary Map)
     * Binding 1: Face 5 (SAT Buffer Intermédiaire)
     * Binding 2: Face 3 (Output Diffusion)
     * Binding 3: Config Uniforms (mapSize, radius, weight)
     */
    get wgslSource(): string {
        return `
            struct Uniforms {
                mapSize: u32,
                radius: i32,
                weight: f32,
            };

            @group(0) @binding(0) var<storage, read> in_map: array<f32>;
            @group(0) @binding(1) var<storage, read_write> sat_map: array<f32>;
            @group(0) @binding(2) var<storage, write> out_diffusion: array<f32>;
            @group(0) @binding(3) var<uniform> config: Uniforms;

            // --- PASS 1: Prefix Sum Horizontal (Lignes) ---
            @compute @workgroup_size(256)
            fn compute_sat_horizontal(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let y = global_id.y;
                if (y >= config.mapSize) { return; }

                // Chaque thread gère une ligne entière (optimisation naive temporaire. FIXME: Parallel Prefix Sum)
                var current_sum: f32 = 0.0;
                for (var x: u32 = 0u; x < config.mapSize; x++) {
                    let idx = y * config.mapSize + x;
                    current_sum += in_map[idx];
                    sat_map[idx] = current_sum;
                }
            }

            // --- PASS 2: Prefix Sum Vertical (Colonnes) ---
            @compute @workgroup_size(256)
            fn compute_sat_vertical(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                if (x >= config.mapSize) { return; }

                // Chaque thread gère une colonne entière
                var current_sum: f32 = 0.0;
                for (var y: u32 = 0u; y < config.mapSize; y++) {
                    let idx = y * config.mapSize + x;
                    current_sum += sat_map[idx];
                    sat_map[idx] = current_sum;
                }
            }

            // --- PASS 3: Extraction Box Filter (Diffusion) ---
            @compute @workgroup_size(16, 16)
            fn compute_diffusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = i32(global_id.x);
                let y = i32(global_id.y);
                let mapSize = i32(config.mapSize);

                if (x >= mapSize || y >= mapSize) { return; }

                let min_x = max(0, x - config.radius);
                let min_y = max(0, y - config.radius);
                let max_x = min(mapSize - 1, x + config.radius);
                let max_y = min(mapSize - 1, y + config.radius);

                // Opcodes O(1)
                var A: f32 = 0.0;
                var B: f32 = 0.0;
                var C: f32 = 0.0;
                
                if (min_x > 0 && min_y > 0) { A = sat_map[u32((min_y - 1) * mapSize + (min_x - 1))]; }
                if (min_y > 0) { B = sat_map[u32((min_y - 1) * mapSize + max_x)]; }
                if (min_x > 0) { C = sat_map[u32(max_y * mapSize + (min_x - 1))]; }
                
                let D: f32 = sat_map[u32(max_y * mapSize + max_x)];

                let sum = D - B - C + A;
                out_diffusion[u32(y * mapSize + x)] = sum * config.weight;
            }
        `;
    }
}
