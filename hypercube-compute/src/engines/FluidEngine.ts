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
     * @param velocityDissipation Taux de dissipation de la vélocité.
     * @param boundary Type de condition aux limites: 'clamp' (bloqué aux bords) ou 'periodic' (tore).
     * @param useProjection Flag pour la passe de projection de Poisson (incompressibilité). Désactivé par défaut sur CPU pour perfs.
     */
    constructor(
        public dt: number = 0.8,
        public buoyancy: number = 0.3,
        public dissipation: number = 0.995,
        public velocityDissipation: number = 0.99,
        public boundary: 'clamp' | 'periodic' = 'clamp',
        public useProjection: boolean = false
    ) { }

    /**
     * Interpole (Bilinear Sampling) une valeur sur une grille 2D
     */
    private bilerp(x: number, y: number, buffer: Float32Array, mapSize: number): number {
        if (this.boundary === 'periodic') {
            x = ((x % mapSize) + mapSize) % mapSize;
            y = ((y % mapSize) + mapSize) % mapSize;
        } else {
            x = Math.max(0, Math.min(x, mapSize - 1));
            y = Math.max(0, Math.min(y, mapSize - 1));
        }

        const x0 = Math.floor(x);
        const y0 = Math.floor(y);

        let x1, y1;
        if (this.boundary === 'periodic') {
            x1 = (x0 + 1) % mapSize;
            y1 = (y0 + 1) % mapSize;
        } else {
            x1 = Math.min(x0 + 1, mapSize - 1);
            y1 = Math.min(y0 + 1, mapSize - 1);
        }

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
        const face5_Curl = faces[4]; // Optionnel: Vorticity/Curl output

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
                face3_VelX[idx] = this.bilerp(sourceX, sourceY, this.prevVelX!, mapSize) * this.velocityDissipation;
                face4_VelY[idx] = this.bilerp(sourceX, sourceY, this.prevVelY!, mapSize) * this.velocityDissipation;
            }
        }

        // --- ETAPE 4 : CALCUL DU CURL (Vorticity) SUR LA FACE 5 ---
        // Optionnel, sert à la visualisation des turbulences
        if (face5_Curl) {
            for (let y = 1; y < mapSize - 1; y++) {
                for (let x = 1; x < mapSize - 1; x++) {
                    const idx = y * mapSize + x;

                    const dVx_dy = (this.prevVelX![(y + 1) * mapSize + x] - this.prevVelX![(y - 1) * mapSize + x]) * 0.5;
                    const dVy_dx = (this.prevVelY![y * mapSize + x + 1] - this.prevVelY![y * mapSize + x - 1]) * 0.5;

                    // curl = dVy/dx - dVx/dy
                    face5_Curl[idx] = dVy_dx - dVx_dy;
                }
            }
        }

        if (this.useProjection) {
            this.project(face3_VelX, face4_VelY, new Float32Array(mapSize * mapSize), new Float32Array(mapSize * mapSize), mapSize);
        }

        // Note: L'algorithme des fluides stable de Stam requerrait une étape de "Projection"
        // (Conservation de la masse en rendant le fluide Incompressible, pour éviter
        // qu'il ne se compacte dans les coins). 
        // Cette passe demande la résolution d'un système de Poisson, très lourd sur CPU JS.
        // On se contente d'une Advection Simple (qui ressemble à du gaz diffus) pour le moment CPU.
        // Le Compute Shader WebGPU prendra le relai pour la divergence.
    }

    /**
     * Méthode de projection (Incompressibilité de Poisson).
     * Stubbé pour l'instant car très lourd en CPU.
     */
    private project(velX: Float32Array, velY: Float32Array, p: Float32Array, div: Float32Array, mapSize: number, iter: number = 20): void {
        // Résolution de l'incompressibilité (Poisson solver)
        // À implémenter avec Jacobi itérations ou à laisser pour le GPU !
    }

    /**
     * Ajoute un "splat" (source) de densité, chaleur et vélocité. Idéal pour les inputs utilisateur (souris, clavier).
     * @param faces Les buffers de fluide
     * @param mapSize La taille de la grille
     * @param cx Centre X du splat (en coordonnées de grille)
     * @param cy Centre Y du splat (en coordonnées de grille)
     * @param vx Vélocité X à injecter
     * @param vy Vélocité Y à injecter
     * @param radius Rayon d'action du splat (en cellules)
     * @param densityAmt Quantité de densité à ajouter
     * @param heatAmt Quantité de chaleur à ajouter
     */
    public addSplat(faces: Float32Array[], mapSize: number, cx: number, cy: number, vx: number, vy: number, radius: number = 20, densityAmt: number = 1.0, heatAmt: number = 5.0): void {
        const face1_Density = faces[0];
        const face2_Heat = faces[1];
        const face3_VelX = faces[2];
        const face4_VelY = faces[3];

        const r2 = radius * radius;
        for (let y = 0; y < mapSize; y++) {
            const dy = y - cy;
            const dy2 = dy * dy;
            if (dy2 > r2) continue; // Optimisation: ignorer les lignes hors de portée

            for (let x = 0; x < mapSize; x++) {
                const dx = x - cx;
                if (dx * dx + dy2 <= r2) {
                    const idx = y * mapSize + x;
                    // Falloff pour un bord doux (gaussien-ish)
                    const falloff = 1.0 - (dx * dx + dy2) / r2;
                    const f = Math.max(0, falloff);

                    face1_Density[idx] += densityAmt * f;
                    face2_Heat[idx] += heatAmt * f;
                    face3_VelX[idx] += vx * f;
                    face4_VelY[idx] += vy * f;
                }
            }
        }
    }

    /**
     * Calcule la masse totale du fluide, utile pour vérifier s'il se dissipe correctement.
     */
    public getTotalDensity(faces: Float32Array[]): number {
        let total = 0;
        const density = faces[0];
        for (let i = 0; i < density.length; i++) {
            total += density[i];
        }
        return total;
    }

    /**
     * Réinitialise toutes les faces du fluide.
     */
    public reset(faces: Float32Array[]): void {
        for (const face of faces) {
            if (face) face.fill(0);
        }
    }
}




































