import { IHypercubeEngine } from './IHypercubeEngine';

/**
 * FlowFieldEngine V12 (Pure Euclidean Gravity)
 * - Single gravity well (Bowling Ball).
 * - NO wrap-around/periodicity logic.
 * - O(N) Analytical calculation (Perfect circles).
 * - Zero performance overhead (60 FPS stable).
 */
export class FlowFieldEngine implements IHypercubeEngine {
    public get name(): string {
        return "FlowFieldEngine-V12";
    }

    public getRequiredFaces(): number {
        return 6; // Standard 6 faces
    }
    public targetX: number = 256;
    public targetY: number = 256;

    constructor(gpuPassCount: number = 30) {
        // isPeriodic is handled by the IHypercubeEngine interface and Grid
    }

    public setTarget(x: number, y: number): void {
        this.targetX = x;
        this.targetY = y;
    }

    public async compute(faces: Float32Array[], nx: number, ny: number, nz: number, chunkX: number = 0, chunkY: number = 0, chunkZ: number = 0): Promise<void> {
        const face0_Distance = faces[0];
        const face4_ForceX = faces[3];
        const face5_ForceY = faces[4];

        // Global offsets
        const offsetX = chunkX * nx;
        const offsetY = chunkY * ny;

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;

            for (let y = 0; y < ny; y++) {
                const globalY = offsetY + y;
                const dy = globalY - this.targetY;
                const dySq = dy * dy;

                for (let x = 0; x < nx; x++) {
                    const idx = zOff + y * nx + x;
                    const globalX = offsetX + x;
                    const dx = globalX - this.targetX;
                    const distSq = dx * dx + dySq;
                    const dist = Math.sqrt(distSq);

                    // Potential for visualization (Face 0)
                    face0_Distance[idx] = dist;

                    // Force Vector pointing directly to target
                    if (dist > 0.1) {
                        face4_ForceX[idx] = -dx / dist;
                        face5_ForceY[idx] = -dy / dist;
                    } else {
                        face4_ForceX[idx] = 0;
                        face5_ForceY[idx] = 0;
                    }
                }
            }
        }
    }
}



































