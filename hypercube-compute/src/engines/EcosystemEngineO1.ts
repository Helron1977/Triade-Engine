import type { IHypercubeEngine } from "./IHypercubeEngine";

export class EcosystemEngineO1 implements IHypercubeEngine {
    public get name(): string {
        return "Ecosystem O1 (Agent-less Pheromones)";
    }

    public getRequiredFaces(): number {
        return 6;
    }

    public init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker: boolean = false): void {
        if (isWorker) return;
        const current = faces[1];
        for (let i = 0; i < nx * ny * nz; i++) {
            if (Math.random() < 0.1) {
                current[i] = Math.random() < 0.5 ? 2 : 3;
            }
        }
    }

    public compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        const current = faces[1];
        const next = faces[2];

        // Automate Cellulaire Combat
        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;

            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const idx = zOff + y * nx + x;
                    const state = current[idx];

                    let blues = 0;
                    let reds = 0;

                    const yM = y > 0 ? y - 1 : ny - 1;
                    const yP = y < ny - 1 ? y + 1 : 0;
                    const xM = x > 0 ? x - 1 : nx - 1;
                    const xP = x < nx - 1 ? x + 1 : 0;

                    // Moore neighborhood
                    const check = (offY: number, offX: number) => {
                        const val = current[zOff + offY * nx + offX];
                        if (val === 2) blues++; else if (val === 3) reds++;
                    };

                    check(yM, xM); check(yM, x); check(yM, xP);
                    check(y, xM); check(y, xP);
                    check(yP, xM); check(yP, x); check(yP, xP);

                    const total = blues + reds;

                    if (state !== 2 && state !== 3) {
                        if (total === 3) {
                            next[idx] = (blues > reds) ? 2 : 3;
                        } else if (Math.random() < 0.0005) {
                            next[idx] = Math.random() < 0.5 ? 2 : 3;
                        } else {
                            next[idx] = 0;
                        }
                    } else if (state === 2) {
                        if (reds >= 2) next[idx] = 0;
                        else if (total === 2 || total === 3) next[idx] = 2;
                        else next[idx] = 0;
                    } else if (state === 3) {
                        if (blues >= 2) next[idx] = 0;
                        else if (total === 2 || total === 3) next[idx] = 3;
                        else next[idx] = 0;
                    }
                }
            }
        }

        current.set(next);
    }
}




































