import type { ITriadeEngine } from "./ITriadeEngine";

export class AerodynamicsEngine implements ITriadeEngine {
    public dragScore: number = 0;
    private initialized: boolean = false;

    public get name(): string {
        return "Lattice Boltzmann D2Q9 (O(1))";
    }

    public compute(faces: Float32Array[], mapSize: number): void {
        const N = mapSize;
        const obstacles = faces[18];
        const ux_out = faces[19];
        const uy_out = faces[20];
        const curl_out = faces[21];

        // Vecteurs du modèle D2Q9
        const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
        const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
        const w = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];
        const opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]; // Rebonds opposés

        const u0 = 0.12; // Vitesse de Mach 0.12 pour générer de sublimes tourbillons
        const omega = 1.95; // Relaxation (Haute turbulence si proche de 2.0)

        // 0. INITIALISATION (F_eq)
        if (!this.initialized) {
            for (let idx = 0; idx < N * N; idx++) {
                const rho = 1.0;
                const ux = u0; const uy = 0.0;
                const u_sq = ux * ux + uy * uy;
                for (let i = 0; i < 9; i++) {
                    const cu = cx[i] * ux + cy[i] * uy;
                    const feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
                    faces[i][idx] = feq;
                    faces[i + 9][idx] = feq;
                }
            }
            this.initialized = true;
        }

        let frameDrag = 0;

        // 1. LBM CORE (Collision & Streaming O(1))
        for (let y = 1; y < N - 1; y++) {
            for (let x = 1; x < N - 1; x++) {
                const idx = y * N + x;

                if (obstacles[idx] > 0) {
                    // BOUNCE BACK: L'air se cogne contre l'Aileron et repart en arrière
                    for (let i = 1; i < 9; i++) {
                        const originX = x - cx[i];
                        const originY = y - cy[i];
                        const originIdx = originY * N + originX;
                        // Renvoi de la distribution
                        faces[opp[i] + 9][originIdx] = faces[i][originIdx];

                        // On mesure la puissance de l'impact direct (Portance/Traînée)
                        if (i === 1) frameDrag += faces[1][originIdx];
                    }
                    ux_out[idx] = 0;
                    uy_out[idx] = 0;
                    continue; // Skip the fluid equations for solid material
                }

                // Macroscopique : Densité et Vitesse
                let rho = 0;
                let ux = 0;
                let uy = 0;
                for (let i = 0; i < 9; i++) {
                    const f_val = faces[i][idx];
                    rho += f_val;
                    ux += cx[i] * f_val;
                    uy += cy[i] * f_val;
                }

                // INLET : Vent continu forcé à gauche (Tunnel Aerodynamique)
                if (x === 1) {
                    ux = u0 * rho;
                    uy = 0.0;
                }

                if (rho > 0) {
                    ux /= rho;
                    uy /= rho;
                }
                ux_out[idx] = ux;
                uy_out[idx] = uy;

                // BGK COLLISION EQUATION
                const u_sq = ux * ux + uy * uy;
                for (let i = 0; i < 9; i++) {
                    const cu = cx[i] * ux + cy[i] * uy;
                    const feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);

                    // Relaxation de Navier-Stokes (Le secret profond du gaz)
                    const f_post = faces[i][idx] * (1.0 - omega) + feq * omega;

                    // STREAMING : Propagation vers les 8 voisins (+ le centre)
                    let nx = x + cx[i];
                    let ny = y + cy[i];

                    // Effet de bord (Enroulement du tunnel en haut / en bas)
                    if (ny < 1) ny = N - 2;
                    else if (ny > N - 2) ny = 1;

                    // OUTLET : Sortie libre (Gradient zero)
                    if (nx > N - 2) nx = N - 2;

                    faces[i + 9][ny * N + nx] = f_post;
                }
            }
        }

        // 2. SWAP BUFFERS
        for (let i = 0; i < 9; i++) {
            faces[i].set(faces[i + 9]);
        }

        // Exagération du calcul de drag pour UI
        this.dragScore = this.dragScore * 0.9 + (frameDrag * 1000) * 0.1;

        // 3. VORTICITY (Curl) POUR L'EFFET "WOW"
        for (let y = 1; y < N - 1; y++) {
            for (let x = 1; x < N - 1; x++) {
                const idx = y * N + x;
                const dUy_dx = uy_out[idx + 1] - uy_out[idx - 1];
                const dUx_dy = ux_out[idx + N] - ux_out[idx - N];
                curl_out[idx] = dUy_dx - dUx_dy;
            }
        }
    }
}
