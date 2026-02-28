import type { ITriadeEngine } from '../../engines/ITriadeEngine';

export interface OceanEngineParams {
    tau_0: number;
    smagorinsky: number;
    cflLimit: number;
    bioDiffusion: number;
    bioGrowth: number;
    vortexRadius: number;
    vortexStrength: number;
}

export class OceanEngine implements ITriadeEngine {
    public readonly name = "OceanEngine";

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
        vortexStrength: 0.02
    };

    public stats = {
        maxU: 0,
        avgTau: 0
    };

    // UI Input simulation (will be fed by the high-level framework/addon)
    public interaction = {
        mouseX: 0,
        mouseY: 0,
        active: false
    };

    constructor() { }

    /**
     * Entry point: Orchestrates LBM and Bio steps
     */
    compute(faces: Float32Array[], size: number): void {
        this.stepLBM(faces, size);
        this.stepBio(faces, size);
    }

    private stepLBM(m: Float32Array[], size: number): void {
        const rho = m[20], ux = m[18], uy = m[19], obst = m[22];

        let maxU = 0;
        let sumTau = 0;
        let activeCells = 0;

        // 0. CLEAR NEXT FRAME BUFFERS
        for (let k = 0; k < 9; k++) m[k + 9].fill(0);

        const mx = this.interaction.mouseX;
        const my = this.interaction.mouseY;
        const isForcing = this.interaction.active;
        const vr2 = this.params.vortexRadius * this.params.vortexRadius;

        // 1. PULL-STREAMING & COLLISION (O1 Optimized)
        // We only compute inner domain. Ghost cells (0 and size-1) are managed by Grid Boundary Exchange.
        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                const i = y * size + x;

                if (obst[i] > 0.5) {
                    for (let k = 0; k < 9; k++) m[k + 9][i] = this.w[k];
                    continue;
                }

                // --- PULL STREAMING ---
                let r = 0, vx = 0, vy = 0;

                for (let k = 0; k < 9; k++) {
                    const nx = x - this.cx[k];
                    const ny = y - this.cy[k];

                    const ni = ny * size + nx;
                    if (obst[ni] > 0.5) {
                        this.pulled_f[k] = m[this.opp[k]][i]; // Bounce back against internal obstacles
                    } else {
                        this.pulled_f[k] = m[k][ni]; // Stream from adjacent cell (can be Ghost Cell)
                    }

                    r += this.pulled_f[k];
                    vx += this.pulled_f[k] * this.cx[k];
                    vy += this.pulled_f[k] * this.cy[k];
                }

                // Stability Clamping
                let isShockwave = false;
                if (r < 0.8) { r = 0.8; isShockwave = true; }
                else if (r > 1.2) { r = 1.2; isShockwave = true; }
                else if (r < 0.0001) r = 0.0001;

                vx /= r;
                vy /= r;

                // Vortex Forcing
                if (isForcing) {
                    const dx = x - mx;
                    const dy = y - my;
                    const dist2 = dx * dx + dy * dy;
                    if (dist2 < vr2) {
                        const forceScale = this.params.vortexStrength * 0.005 * (1.0 - Math.sqrt(dist2) / this.params.vortexRadius);
                        vx += -dy * forceScale;
                        vy += dx * forceScale;
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
                        m[k + 9][i] = this.w[k] * r * (1 + cu + 0.5 * cu * cu - 1.5 * u2_clamped);
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

                    const S_norm = Math.sqrt(2 * (Pxx * Pxx + Pyy * Pyy + 2 * Pxy * Pxy));
                    let tau_eff = this.params.tau_0 + this.params.smagorinsky * S_norm;
                    if (isNaN(tau_eff) || tau_eff < 0.505) tau_eff = 0.505;

                    sumTau += tau_eff;
                    activeCells++;

                    for (let k = 0; k < 9; k++) {
                        m[k + 9][i] = this.pulled_f[k] - (this.pulled_f[k] - this.feq_cache[k]) / tau_eff;
                    }
                }
            }
        }

        if (activeCells > 0) this.stats.avgTau = sumTau / activeCells;
        this.stats.maxU = maxU;

        // 4. MEMORY SWAP
        for (let k = 0; k < 9; k++) {
            const tmp = m[k];
            m[k] = m[k + 9];
            m[k + 9] = tmp;
        }
    }

    private stepBio(m: Float32Array[], size: number): void {
        const bio = m[21];
        const bio_next = m[17]; // Proxy temp
        const area = size * size;

        for (let y = 1; y < size - 1; y++) {
            for (let x = 1; x < size - 1; x++) {
                const i = y * size + x;

                const lap = bio[i - 1] + bio[i + 1] + bio[i - size] + bio[i + size] - 4 * bio[i];
                let next = bio[i] + this.params.bioDiffusion * lap + this.params.bioGrowth * bio[i] * (1 - bio[i]);

                if (next < 0) next = 0;
                if (next > 1) next = 1;
                bio_next[i] = next;

                // Bio no longer acts as physical obstacle to allow the fixed Island to persist.
            }
        }

        for (let i = 0; i < area; i++) bio[i] = bio_next[i];
    }
}
