import type { IHypercubeEngine } from './IHypercubeEngine';

/**
 * GrayScottEngine
 * Simule des motifs de Turing (Réaction-Diffusion) haute performance.
 * Équations :
 * dA/dt = Da * lap(A) - AB^2 + f(1-A)
 * dB/dt = Db * lap(B) + AB^2 - (f+k)B
 */
export class GrayScottEngine implements IHypercubeEngine {
    public get name(): string {
        return "GrayScottEngine";
    }

    public getSyncFaces(): number[] {
        return [0, 1]; // Sync Substance A and B across chunk boundaries
    }

    public getRequiredFaces(): number {
        return 4; // A, B, A_next, B_next
    }

    constructor(
        public Da: number = 0.2,
        public Db: number = 0.1,
        public feed: number = 0.035,
        public kill: number = 0.06
    ) { }

    public init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker: boolean = false): void {
        if (isWorker) return;
        // Initial state: Fill A with 1.0, B with 0.0
        faces[0].fill(1.0);
        faces[1].fill(0.0);
    }

    public compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        const A = faces[0];
        const B = faces[1];
        const Anext = faces[2];
        const Bnext = faces[3];

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;
            for (let ly = 1; ly < ny - 1; ly++) {
                const yOff = ly * nx;
                for (let lx = 1; lx < nx - 1; lx++) {
                    const idx = zOff + yOff + lx;

                    const a = A[idx];
                    const b = B[idx];

                    // 5-point laplacian (2D slice focus for performance & patterns)
                    const lapA = (A[idx - 1] + A[idx + 1] + A[idx - nx] + A[idx + nx] - 4 * a);
                    const lapB = (B[idx - 1] + B[idx + 1] + B[idx - nx] + B[idx + nx] - 4 * b);

                    const react = a * b * b;

                    Anext[idx] = a + (this.Da * lapA - react + this.feed * (1 - a));
                    Bnext[idx] = b + (this.Db * lapB + react - (this.feed + this.kill) * b);
                }
            }
        }

        A.set(Anext);
        B.set(Bnext);
    }
}
