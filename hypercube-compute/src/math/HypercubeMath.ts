/**
 * Generic mathematics and physical injection helpers for Hypercube grids.
 * These functions directly mutate the face arrays BEFORE grid.compute().
 */
export class HypercubeMath {
    /**
     * Injects passive scalar (e.g., Smoke or Plankton) into a specific face.
     */
    static injectScalar(
        faces: Float32Array[],
        faceIndex: number,
        nx: number,
        ny: number,
        mx: number,
        my: number,
        radius: number = 10.0,
        amount: number = 1.0
    ): void {
        const targetFace = faces[faceIndex];
        const r2 = radius * radius;

        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                const ix = Math.floor(mx + dx);
                const iy = Math.floor(my + dy);
                if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
                    const d2 = dx * dx + dy * dy;
                    if (d2 < r2) {
                        targetFace[iy * nx + ix] = amount;
                    }
                }
            }
        }
    }

    /**
     * Injects momentum (velocity) directly into a D2Q9 LBM distribution field.
     * This modifies faces 0-8 proportionally to the requested delta velocity.
     * Must provide face 22 as the density field.
     */
    static injectMomentumD2Q9(
        faces: Float32Array[],
        nx: number,
        ny: number,
        mx: number,
        my: number,
        radius: number,
        vortexStrength: number,
        densityFaceIndex: number = 22
    ): void {
        const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
        const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
        const w = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];

        const rhoFace = faces[densityFaceIndex];
        const r2 = radius * radius;

        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                const ix = Math.floor(mx + dx);
                const iy = Math.floor(my + dy);

                if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
                    const dist2 = dx * dx + dy * dy;
                    if (dist2 < r2) {
                        const idx = iy * nx + ix;
                        const rho = rhoFace[idx] || 1.0;

                        // Calculate Force vector as a vortex
                        const forceScale = vortexStrength * 0.05 * (1.0 - Math.sqrt(dist2) / radius);
                        const Fx = -dy * forceScale;
                        const Fy = dx * forceScale;

                        // Inject momentum into distributions
                        const parity = (faces.length > 9) ? ((faces[9][0] !== 0) ? 0 : 0) : 0;
                        // Wait, parity detection is hard here. Let's just inject into BOTH to be safe.
                        for (let k = 0; k < 9; k++) {
                            const cu = 3.0 * (cx[k] * Fx + cy[k] * Fy);
                            const val = w[k] * rho * cu;
                            faces[k][idx] += val;
                            if (faces[k + 9]) faces[k + 9][idx] += val;
                        }
                    }
                }
            }
        }
    }
}
