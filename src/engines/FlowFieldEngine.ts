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

    public async compute(faces: Float32Array[], mapSize: number, chunkX: number = 0, chunkY: number = 0): Promise<void> {
        const face3_Integration = faces[2];
        const face4_ForceX = faces[3];
        const face5_ForceY = faces[4];

        // Global offsets
        const offsetX = chunkX * mapSize;
        const offsetY = chunkY * mapSize;

        // Pure Euclidean Gravitational Field (No Wrap-around)
        // This is the "Bowling Ball on a Trampoline" model.
        for (let y = 0; y < mapSize; y++) {
            const rowOffset = y * mapSize;
            const globalY = offsetY + y;
            const dy = globalY - this.targetY;
            const dySq = dy * dy;

            for (let x = 0; x < mapSize; x++) {
                const idx = rowOffset + x;
                const globalX = offsetX + x;
                const dx = globalX - this.targetX;
                const distSq = dx * dx + dySq;
                const dist = Math.sqrt(distSq);

                // Potential for visualization (Face 3)
                face3_Integration[idx] = dist;

                // Force Vector pointing directly to target
                if (dist > 0.1) {
                    // Perfect radial alignment
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



































