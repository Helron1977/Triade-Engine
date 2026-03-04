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

    constructor(
        public dt: number = 0.8,
        public buoyancy: number = 0.3,
        public dissipation: number = 0.995,
        public velocityDissipation: number = 0.99,
        public boundary: 'clamp' | 'periodic' = 'clamp',
        public useProjection: boolean = false
    ) { }

    public getConfig(): Record<string, any> {
        return {
            dt: this.dt,
            buoyancy: this.buoyancy,
            dissipation: this.dissipation,
            velocityDissipation: this.velocityDissipation,
            boundary: this.boundary,
            useProjection: this.useProjection
        };
    }

    public init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker: boolean = false): void {
        const totalCells = nx * ny * nz;
        if (!this.prevDensity || this.prevDensity.length !== totalCells) {
            this.prevDensity = new Float32Array(totalCells);
            this.prevHeat = new Float32Array(totalCells);
            this.prevVelX = new Float32Array(totalCells);
            this.prevVelY = new Float32Array(totalCells);
        }
    }

    /**
     * Interpole (Bilinear Sampling) une valeur sur une grille 2D
     */
    private bilerp(x: number, y: number, buffer: Float32Array, nx: number, ny: number, zOff: number): number {
        if (this.boundary === 'periodic') {
            x = ((x % nx) + nx) % nx;
            y = ((y % ny) + ny) % ny;
        } else {
            x = Math.max(0, Math.min(x, nx - 1));
            y = Math.max(0, Math.min(y, ny - 1));
        }

        const x0 = Math.floor(x);
        const y0 = Math.floor(y);

        let x1, y1;
        if (this.boundary === 'periodic') {
            x1 = (x0 + 1) % nx;
            y1 = (y0 + 1) % ny;
        } else {
            x1 = Math.min(x0 + 1, nx - 1);
            y1 = Math.min(y0 + 1, ny - 1);
        }

        const tx = x - x0;
        const ty = y - y0;

        const v00 = buffer[zOff + y0 * nx + x0];
        const v10 = buffer[zOff + y0 * nx + x1];
        const v01 = buffer[zOff + y1 * nx + x0];
        const v11 = buffer[zOff + y1 * nx + x1];

        const lerpX1 = v00 * (1 - tx) + v10 * tx;
        const lerpX2 = v01 * (1 - tx) + v11 * tx;

        return lerpX1 * (1 - ty) + lerpX2 * ty;
    }

    /**
     * Calcule la dynamique de fluid (Ajout de forces -> Advection)
     * Version CPU.
     */
    compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        const face1_Density = faces[0];
        const face2_Heat = faces[1];
        const face3_VelX = faces[2];
        const face4_VelY = faces[3];
        const face5_Curl = faces[4];

        const totalCells = nx * ny * nz;

        // Sauvegarde de l'état "Précédent"
        this.prevDensity!.set(face1_Density);
        this.prevHeat!.set(face2_Heat);
        this.prevVelX!.set(face3_VelX);
        this.prevVelY!.set(face4_VelY);

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;

            // --- ETAPE 1 : AJOUT DE FORCES ---
            for (let i = 0; i < nx * ny; i++) {
                const idx = zOff + i;
                const heat = this.prevHeat![idx];
                if (heat > 0) {
                    face4_VelY[idx] -= heat * this.buoyancy * this.dt;
                }
            }

            // --- ETAPE 2 & 3 : ADVECTION ---
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const idx = zOff + y * nx + x;
                    const vx = face3_VelX[idx];
                    const vy = face4_VelY[idx];
                    const sourceX = x - vx * this.dt;
                    const sourceY = y - vy * this.dt;

                    face1_Density[idx] = this.bilerp(sourceX, sourceY, this.prevDensity!, nx, ny, zOff) * this.dissipation;
                    face2_Heat[idx] = this.bilerp(sourceX, sourceY, this.prevHeat!, nx, ny, zOff) * this.dissipation;
                    face3_VelX[idx] = this.bilerp(sourceX, sourceY, this.prevVelX!, nx, ny, zOff) * this.velocityDissipation;
                    face4_VelY[idx] = this.bilerp(sourceX, sourceY, this.prevVelY!, nx, ny, zOff) * this.velocityDissipation;
                }
            }

            // --- ETAPE 4 : CALCUL DU CURL ---
            if (face5_Curl) {
                for (let y = 1; y < ny - 1; y++) {
                    for (let x = 1; x < nx - 1; x++) {
                        const idx = zOff + y * nx + x;
                        const dVx_dy = (this.prevVelX![zOff + (y + 1) * nx + x] - this.prevVelX![zOff + (y - 1) * nx + x]) * 0.5;
                        const dVy_dx = (this.prevVelY![zOff + y * nx + x + 1] - this.prevVelY![zOff + y * nx + x - 1]) * 0.5;
                        face5_Curl[idx] = dVy_dx - dVx_dy;
                    }
                }
            }
        }

        if (this.useProjection) {
            for (let lz = 0; lz < nz; lz++) {
                const zOff = lz * ny * nx;
                const vX = faces[2].subarray(zOff, zOff + nx * ny);
                const vY = faces[3].subarray(zOff, zOff + nx * ny);
                this.project(vX, vY, new Float32Array(nx * ny), new Float32Array(nx * ny), nx, ny);
            }
        }
    }

    /**
     * Méthode de projection (Incompressibilité de Poisson).
     * Stubbé pour l'instant car très lourd en CPU.
     */
    private project(velX: Float32Array, velY: Float32Array, p: Float32Array, div: Float32Array, nx: number, ny: number, iter: number = 20): void {
        // Résolution de l'incompressibilité (Poisson solver)
    }

    /**
     * Ajoute un "splat" (source) de densité, chaleur et vélocité. Idéal pour les inputs utilisateur (souris, clavier).
     */
    public addSplat(faces: Float32Array[], nx: number, ny: number, nz: number, lz: number, cx: number, cy: number, vx: number, vy: number, radius: number = 20, densityAmt: number = 1.0, heatAmt: number = 5.0): void {
        const face1_Density = faces[0];
        const face2_Heat = faces[1];
        const face3_VelX = faces[2];
        const face4_VelY = faces[3];
        const zOff = lz * ny * nx;

        const r2 = radius * radius;
        for (let y = 0; y < ny; y++) {
            const dy = y - cy;
            const dy2 = dy * dy;
            if (dy2 > r2) continue;

            for (let x = 0; x < nx; x++) {
                const dx = x - cx;
                if (dx * dx + dy2 <= r2) {
                    const idx = zOff + y * nx + x;
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
     * Calcule la masse totale du fluide.
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
