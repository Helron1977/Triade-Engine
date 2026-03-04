import type { IHypercubeEngine } from './IHypercubeEngine';

export interface OceanEngineParams {
    tau_0: number;
    smagorinsky: number;
    cflLimit: number;
    bioDiffusion: number;
    bioGrowth: number;
    vortexRadius: number;
    vortexStrength: number;
    closedBounds: boolean;
}

/**
 * OceanEngine – Shallow Water + Plankton Dynamics (D2Q9 LBM)
 * Simulation océanique simplifiée : courants, tourbillons, forcing interactif, et bio-diffusion.
 * 
 * @faces
 * - 0–8   : f (populations LBM)
 * - 9–17  : f_post (post-collision temp buffers)
 * - 18    : obst (murs/îles statiques > 0.5)
 * - 19    : ux (vitesse X vectorielle)
 * - 20    : uy (vitesse Y vectorielle)
 * - 21    : curl (vorticité pour rendu)
 * - 22    : rho (densité de masse locale)
 * - 23    : bio (plancton / concentration passive)
 * - 24    : bio_next (temp buffer pour bio)
 * 
 * Note globale : La propriété `interaction` doit être mise à jour chaque frame 
 * par l'environnement ou un `EventListener` de type "mousemove & mousedown".
 */
export class OceanEngine implements IHypercubeEngine {
    public get name(): string {
        return "OceanEngine";
    }

    public getRequiredFaces(): number {
        return 25; // Suite faces 0-17 + 18-24
    }

    public getSyncFaces(): number[] {
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20]; // LBM pop (0-8) + macros (ux, uy, rho)
    }

    // Re-use lab-perfect constants
    private readonly w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
    private readonly cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
    private readonly cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    private readonly opp = [0, 3, 4, 1, 2, 7, 8, 5, 6];

    // Caches to avoid per-frame allocations
    private feq_cache = new Float32Array(9);
    private pulled_f = new Float32Array(9);

    public params: OceanEngineParams = {
        tau_0: 0.8,
        smagorinsky: 0.2,
        cflLimit: 0.38,
        bioDiffusion: 0.05,
        bioGrowth: 0.0005,
        vortexRadius: 28,
        vortexStrength: 0.02,
        closedBounds: false
    };

    public stats = {
        maxU: 0,
        avgTau: 0,
        avgRho: 0
    };

    // UI Input simulation (will be fed by the high-level framework/addon)
    public interaction = {
        mouseX: 0,
        mouseY: 0,
        active: false
    };

    constructor() { }

    public getConfig(): Record<string, any> {
        return {
            ...this.params
        };
    }

    public init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker: boolean = false): void {
        if (isWorker) return; // Main thread already initialized SAB

        const u0 = 0.0;
        const v0 = 0.0;
        const rho0 = 1.0;
        const u2 = u0 * u0 + v0 * v0;

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;
            for (let i = 0; i < nx * ny; i++) {
                const idx = zOff + i;
                faces[22][idx] = rho0;
                faces[19][idx] = u0;
                faces[20][idx] = v0;
                faces[23][idx] = 0.01; // Initial small plankton amount

                for (let k = 0; k < 9; k++) {
                    const cu = 3 * (this.cx[k] * u0 + this.cy[k] * v0);
                    const feq = this.w[k] * rho0 * (1 + cu + 0.5 * cu * cu - 1.5 * u2);
                    faces[k][idx] = feq;
                    faces[k + 9][idx] = feq;
                }
            }
        }
    }

    public addGlobalCurrent(faces: Float32Array[], targetUx: number, targetUy: number): void {
        const nx = 256; // Fallback size, ideally should get nx/ny
        const ny = 256;
        const ux = faces[19];
        const uy = faces[20];
        for (let i = 0; i < nx * ny; i++) {
            ux[i] += targetUx;
            uy[i] += targetUy;
        }
    }

    public addVortex(faces: Float32Array[], mx: number, my: number, strength: number = 10.0): void {
        this.interaction.mouseX = mx;
        this.interaction.mouseY = my;
        this.interaction.active = true;

        // Disable after a frame to simulate impulse
        setTimeout(() => { this.interaction.active = false; }, 50);
    }

    /**
     * Entry point: Orchestrates LBM and Bio steps
     */
    compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        for (let lz = 0; lz < nz; lz++) {
            this.stepLBM(faces, nx, ny, lz);
            this.stepBio(faces, nx, ny, lz);
        }
    }

    private stepLBM(faces: Float32Array[], nx: number, ny: number, lz: number): void {
        const size = nx;
        const rho = faces[22], ux = faces[19], uy = faces[20], obst = faces[18];
        const zOff = lz * ny * nx;

        let maxU = 0;
        let sumTau = 0;
        let sumRho = 0;
        let activeCells = 0;

        // 0. CLEAR NEXT FRAME BUFFERS (Only the slice part)
        for (let k = 0; k < 9; k++) {
            for (let i = 0; i < nx * ny; i++) faces[k + 9][zOff + i] = 0;
        }

        const mx = this.interaction.mouseX;
        const my = this.interaction.mouseY;
        const isForcing = this.interaction.active;
        const vr2 = this.params.vortexRadius * this.params.vortexRadius;

        // 1. PULL-STREAMING, MACROS & COLLISION (O1 Optimized)
        for (let y = 1; y < ny - 1; y++) {
            for (let x = 1; x < nx - 1; x++) {
                const i = zOff + y * nx + x;

                if (obst[i] > 0.5) {
                    for (let k = 0; k < 9; k++) faces[k + 9][i] = this.w[k];
                    continue;
                }

                // --- PULL STREAMING ---
                let r = 0, vx = 0, vy = 0;

                for (let k = 0; k < 9; k++) {
                    const local_nx = x - this.cx[k];
                    const local_ny = y - this.cy[k];

                    if (this.params.closedBounds && (local_nx <= 0 || local_nx >= nx - 1 || local_ny <= 0 || local_ny >= ny - 1)) {
                        this.pulled_f[k] = faces[this.opp[k]][i];
                    } else {
                        const ni = zOff + local_ny * nx + local_nx;
                        if (obst[ni] > 0.5) {
                            this.pulled_f[k] = faces[this.opp[k]][i];
                        } else {
                            this.pulled_f[k] = faces[k][ni];
                        }
                    }

                    r += this.pulled_f[k];
                    vx += this.pulled_f[k] * this.cx[k];
                    vy += this.pulled_f[k] * this.cy[k];
                }

                // Stability Clamping
                let isShockwave = false;
                if (r < 0.8 || r > 1.2 || r < 0.0001) {
                    const targetRho = Math.max(0.8, Math.min(1.2, r < 0.0001 ? 1.0 : r));
                    const scale = targetRho / r;
                    for (let k = 0; k < 9; k++) this.pulled_f[k] *= scale;
                    r = targetRho;
                    isShockwave = true;
                }

                vx /= r;
                vy /= r;

                // Vortex Forcing
                let Fx = 0;
                let Fy = 0;
                if (isForcing) {
                    const dx = x - mx;
                    const dy = y - my;
                    const dist2 = dx * dx + dy * dy;
                    if (dist2 < vr2) {
                        const forceScale = this.params.vortexStrength * 0.005 * (1.0 - Math.sqrt(dist2) / this.params.vortexRadius);
                        Fx = -dy * forceScale;
                        Fy = dx * forceScale;
                        vx += Fx / r;
                        vy += Fy / r;
                    }
                }

                const v2 = vx * vx + vy * vy;
                const speed = Math.sqrt(v2);
                if (speed > maxU) maxU = speed;

                let u2_clamped = v2;
                if (speed > this.params.cflLimit) {
                    const scale = this.params.cflLimit / speed;
                    vx *= scale;
                    vy *= scale;
                    u2_clamped = vx * vx + vy * vy;
                    isShockwave = true;
                }

                // Store Macros
                rho[i] = r;
                ux[i] = vx;
                uy[i] = vy;

                // --- COLLISION ---
                if (isShockwave) {
                    for (let k = 0; k < 9; k++) {
                        const cu = 3 * (this.cx[k] * vx + this.cy[k] * vy);
                        faces[k + 9][i] = this.w[k] * r * (1 + cu + 0.5 * cu * cu - 1.5 * u2_clamped);
                    }
                } else {
                    let Pxx = 0, Pyy = 0, Pxy = 0;
                    for (let k = 0; k < 9; k++) {
                        const cu = 3 * (this.cx[k] * vx + this.cy[k] * vy);
                        this.feq_cache[k] = this.w[k] * r * (1 + cu + 0.5 * cu * cu - 1.5 * u2_clamped);
                        const fneq = this.pulled_f[k] - this.feq_cache[k];
                        Pxx += fneq * this.cx[k] * this.cx[k];
                        Pyy += fneq * this.cy[k] * this.cy[k];
                        Pxy += fneq * this.cx[k] * this.cy[k];
                    }

                    let S_norm = Math.sqrt(2 * (Pxx * Pxx + Pyy * Pyy + 2 * Pxy * Pxy));
                    if (S_norm > 10.0 || isNaN(S_norm)) S_norm = 10.0;
                    let tau_eff = this.params.tau_0 + this.params.smagorinsky * S_norm;
                    if (isNaN(tau_eff) || tau_eff < 0.505) tau_eff = 0.505;
                    else if (tau_eff > 2.0) tau_eff = 2.0;

                    sumTau += tau_eff;
                    sumRho += r;
                    activeCells++;

                    for (let k = 0; k < 9; k++) {
                        faces[k + 9][i] = this.pulled_f[k] - (this.pulled_f[k] - this.feq_cache[k]) / tau_eff;
                    }
                }
            }
        }

        // 2. VORTICITY / CURL Calculation (Face 21 - needed for visualization)
        const curl_out = faces[21];
        for (let y = 1; y < ny - 1; y++) {
            for (let x = 1; x < nx - 1; x++) {
                const i = zOff + y * nx + x;
                const xM = x > 1 ? x - 1 : 1;
                const xP = x < nx - 2 ? x + 1 : nx - 2;
                const dxDist = (x === 1 || x === nx - 2) ? 1.0 : 2.0;

                const yM_idx = y > 1 ? y - 1 : 1;
                const yP_idx = y < ny - 2 ? y + 1 : ny - 2;
                const dyDist = (y === 1 || y === ny - 2) ? 1.0 : 2.0;

                const dUy_dx = (uy[zOff + y * nx + xP] - uy[zOff + y * nx + xM]) / dxDist;
                const dUx_dy = (ux[zOff + yP_idx * nx + x] - ux[zOff + yM_idx * nx + x]) / dyDist;
                curl_out[i] = dUy_dx - dUx_dy;
            }
        }

        if (activeCells > 0) {
            this.stats.avgTau = sumTau / activeCells;
            this.stats.avgRho = sumRho / activeCells;
        }
        this.stats.maxU = maxU;

        // 4. MEMORY SWAP (Only for the slice)
        for (let k = 0; k < 9; k++) {
            for (let i = 0; i < nx * ny; i++) {
                const idx = zOff + i;
                const tmp = faces[k][idx];
                faces[k][idx] = faces[k + 9][idx];
                faces[k + 9][idx] = tmp;
            }
        }
    }

    private stepBio(faces: Float32Array[], nx: number, ny: number, lz: number): void {
        const bio = faces[23];
        const bio_next = faces[24];
        const zOff = lz * ny * nx;

        for (let y = 1; y < ny - 1; y++) {
            for (let x = 1; x < nx - 1; x++) {
                const i = zOff + y * nx + x;

                // Diffusion laplacienne
                const lap = bio[i - 1] + bio[i + 1] + bio[i - nx] + bio[i + nx] - 4 * bio[i];
                let next = bio[i] + this.params.bioDiffusion * lap + this.params.bioGrowth * bio[i] * (1 - bio[i]);

                // Advection
                const ux = faces[18][i];
                const uy = faces[19][i];
                const ax = Math.max(1, Math.min(nx - 2, x - ux * 0.8));
                const ay = Math.max(1, Math.min(ny - 2, y - uy * 0.8));
                const ix = Math.floor(ax);
                const iy = Math.floor(ay);
                const fx = ax - ix;
                const fy = ay - iy;

                const v00 = bio[zOff + iy * nx + ix];
                const v10 = bio[zOff + iy * nx + Math.min(ix + 1, nx - 2)];
                const v01 = bio[zOff + Math.min(iy + 1, ny - 2) * nx + ix];
                const v11 = bio[zOff + Math.min(iy + 1, ny - 2) * nx + Math.min(ix + 1, nx - 2)];

                const advected = (1 - fy) * ((1 - fx) * v00 + fx * v10) + fy * ((1 - fx) * v01 + fx * v11);
                next = advected + this.params.bioDiffusion * lap + this.params.bioGrowth * bio[i] * (1 - bio[i]);

                if (next < 0) next = 0;
                if (next > 1) next = 1;
                bio_next[i] = next;
            }
        }

        for (let i = 0; i < nx * ny; i++) bio[zOff + i] = bio_next[zOff + i];
    }
}


