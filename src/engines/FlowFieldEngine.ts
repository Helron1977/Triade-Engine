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
    public isPeriodic: boolean = false;

    constructor(gpuPassCount: number = 30, isPeriodic: boolean = false) {
        this.isPeriodic = false;
    }

    public async compute(faces: Float32Array[], mapSize: number): Promise<void> {
        const face2_Target = faces[1];
        const face3_Integration = faces[2];
        const face4_ForceX = faces[3];
        const face5_ForceY = faces[4];

        // 1. Find the target position (O(N) search)
        let tx = -1, ty = -1;
        const len = face2_Target.length;
        for (let i = 0; i < len; i++) {
            if (face2_Target[i] === 0) {
                tx = i % mapSize;
                ty = (i / mapSize) | 0;
                break;
            }
        }

        if (tx === -1) return;

        // 2. Pure Euclidean Gravitational Field (No Wrap-around)
        // This is the "Bowling Ball on a Trampoline" model.
        for (let y = 0; y < mapSize; y++) {
            const rowOffset = y * mapSize;
            const dy = y - ty;
            const dySq = dy * dy;

            for (let x = 0; x < mapSize; x++) {
                const idx = rowOffset + x;
                const dx = x - tx;
                const distSq = dx * dx + dySq;
                const dist = Math.sqrt(distSq);

                // Potential for visualization (Face 3)
                face3_Integration[idx] = dist;

                // Force Vector pointing directly to target
                if (dist > 0.01) {
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



































