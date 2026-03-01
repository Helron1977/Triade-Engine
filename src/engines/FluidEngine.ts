import type { IHypercubeEngine } from './IHypercubeEngine';

/**
 * FluidEngine (Moteur de Dynamique des Fluides Simplifiée)
 * Simule des écoulements et la flottabilité thermique via l'Advection Pure.
 * Idéal pour simuler de la fumée, des nuages ou des fluides réactifs.
 * 
 * Mapping des Faces :
 * Face 1: Density Map (Quantité de matière, ex: 0 = vide, 255 = très dense)
 * Face 2: Heat/Pressure Map (La température induit une force ascendante continue)
 * Face 3: Velocity X (Champ vectoriel horizontal de la force de déplacement)
 * Face 4: Velocity Y (Champ vectoriel vertical de la force de déplacement)
 */
export class FluidEngine implements IHypercubeEngine {
    public get name(): string {
        return "Fluid Engine (O1 Tensor Nav-Stokes)";
    }

    public getRequiredFaces(): number {
        return 6;
    }

    // CPU: Buffers temporaires ("Ping-Pong") pour stocker l'état précédent lors de l'advection
    private prevDensity: Float32Array | null = null;
    private prevHeat: Float32Array | null = null;
    private prevVelX: Float32Array | null = null;
    private prevVelY: Float32Array | null = null;

    // WebGPU: Pipelines d'Advection (Fluides)
    private pipelineForce: GPUComputePipeline | null = null;
    private pipelineAdvection: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;

    /**
     * @param dt Delta Time virtuel. Contrôle la vitesse apparente de l'écoulement.
     * @param buoyancy Force de flottabilité (Lévitation liée à la température).
     * @param dissipation Taux de disparition du fluide par frame (ex: 0.99 = la fumée s'estompe lentement).
     */
    constructor(
        public dt: number = 0.8,
        public buoyancy: number = 0.3,
        public dissipation: number = 0.995
    ) { }

    /**
     * Interpole (Bilinear Sampling) une valeur sur une grille 2D
     */
    private bilerp(x: number, y: number, buffer: Float32Array, mapSize: number): number {
        const x0 = Math.max(0, Math.min(Math.floor(x), mapSize - 1));
        const x1 = Math.max(0, Math.min(x0 + 1, mapSize - 1));
        const y0 = Math.max(0, Math.min(Math.floor(y), mapSize - 1));
        const y1 = Math.max(0, Math.min(y0 + 1, mapSize - 1));

        const tx = x - x0;
        const ty = y - y0;

        const v00 = buffer[y0 * mapSize + x0];
        const v10 = buffer[y0 * mapSize + x1];
        const v01 = buffer[y1 * mapSize + x0];
        const v11 = buffer[y1 * mapSize + x1];

        const lerpX1 = v00 * (1 - tx) + v10 * tx;
        const lerpX2 = v01 * (1 - tx) + v11 * tx;

        return lerpX1 * (1 - ty) + lerpX2 * ty;
    }

    /**
     * Calcule la dynamique de fluid (Ajout de forces -> Advection)
     * Version CPU.
     */
    compute(faces: Float32Array[], mapSize: number): void {
        const face1_Density = faces[0];
        const face2_Heat = faces[1];
        const face3_VelX = faces[2];
        const face4_VelY = faces[3];

        const totalCells = mapSize * mapSize;

        // Allocation Ping-Pong (Lazy init)
        if (!this.prevDensity || this.prevDensity.length !== totalCells) {
            this.prevDensity = new Float32Array(totalCells);
            this.prevHeat = new Float32Array(totalCells);
            this.prevVelX = new Float32Array(totalCells);
            this.prevVelY = new Float32Array(totalCells);
        }

        // Sauvegarde de l'état "Précédent"
        this.prevDensity!.set(face1_Density);
        this.prevHeat!.set(face2_Heat);
        this.prevVelX!.set(face3_VelX);
        this.prevVelY!.set(face4_VelY);

        // --- ETAPE 1 : AJOUT DE FORCES (Flottabilité liée à la Température) ---
        // Une case chaude poussera le gaz vers le "Haut" (-Y en coordonnées d'écran/grille classiques)
        for (let i = 0; i < totalCells; i++) {
            const heat = this.prevHeat![i];
            if (heat > 0) {
                // Application de la flottabilité à Vélocité Y
                face4_VelY[i] -= heat * this.buoyancy * this.dt;
            }
        }

        // --- ETAPE 2 & 3 : ADVECTION (Rétro-projection) ---
        // Pour chaque cellule, on regarde "d'où vient" le fluide en remontant le vecteur vélocité.
        for (let y = 0; y < mapSize; y++) {
            for (let x = 0; x < mapSize; x++) {
                const idx = y * mapSize + x;

                // Vélocité locale
                const vx = face3_VelX[idx];
                const vy = face4_VelY[idx];

                // Point source (rétrograde dans le temps par rapport à la vitesse)
                const sourceX = x - vx * this.dt;
                const sourceY = y - vy * this.dt;

                // Advection des Propriétés (Interpolation Bilinéaire de l'état précédent)
                face1_Density[idx] = this.bilerp(sourceX, sourceY, this.prevDensity!, mapSize) * this.dissipation;
                face2_Heat[idx] = this.bilerp(sourceX, sourceY, this.prevHeat!, mapSize) * this.dissipation;
                face3_VelX[idx] = this.bilerp(sourceX, sourceY, this.prevVelX!, mapSize) * 0.99; // Légère dissipation de l'impulsion
                face4_VelY[idx] = this.bilerp(sourceX, sourceY, this.prevVelY!, mapSize) * 0.99;
            }
        }

        // Note: L'algorithme des fluides stable de Stam requerrait une étape de "Projection"
        // (Conservation de la masse en rendant le fluide Incompressible, pour éviter
        // qu'il ne se compacte dans les coins). 
        // Cette passe demande la résolution d'un système de Poisson, très lourd sur CPU JS.
        // On se contente d'une Advection Simple (qui ressemble à du gaz diffus) pour le moment CPU.
        // Le Compute Shader WebGPU prendra le relai pour la divergence.
    }
}




































