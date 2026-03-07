import type { HypercubeCpuGrid } from '../HypercubeCpuGrid';

/**
 * Defines what happens at the outer edges of the entire Simulation Grid.
 * Replaces hardcoded behaviors previously buried inside mathematical engines.
 */
export enum BoundaryType {
    PERIODIC = "PERIODIC",
    WALL = "WALL",
    INFLOW = "INFLOW",
    OUTFLOW = "OUTFLOW"
}

export interface BoundaryConfig {
    left: BoundaryType;
    right: BoundaryType;
    top: BoundaryType;
    bottom: BoundaryType;

    // Optional parameters for INFLOW specific behaviors
    inflowUx?: number;
    inflowUy?: number;
    inflowDensity?: number;
}

export class BoundaryConditions {
    /**
     * Applies the configured boundary conditions to the logical external edges of the Grid.
     * This function should be called inside or exactly after grid.synchronizeBoundaries().
     */
    static apply(
        grid: HypercubeCpuGrid,
        config: BoundaryConfig,
        faceIndicesToSync: number[]
    ) {
        // Enforcing specific edge behaviors per Chunk if it touches the global boundary
        for (let y = 0; y < grid.rows; y++) {
            for (let x = 0; x < grid.cols; x++) {
                const chunk = grid.cubes[y][x]!;

                // Boundary: Left (x = 0)
                if (x === 0 && config.left !== BoundaryType.PERIODIC) {
                    this.applyEdgeCondition(chunk.faces, chunk.nx, chunk.ny, 'LEFT', config.left, config, faceIndicesToSync);
                }

                // Boundary: Right (x = cols - 1)
                if (x === grid.cols - 1 && config.right !== BoundaryType.PERIODIC) {
                    this.applyEdgeCondition(chunk.faces, chunk.nx, chunk.ny, 'RIGHT', config.right, config, faceIndicesToSync);
                }

                // Boundary: Top (y = 0)
                if (y === 0 && config.top !== BoundaryType.PERIODIC) {
                    this.applyEdgeCondition(chunk.faces, chunk.nx, chunk.ny, 'TOP', config.top, config, faceIndicesToSync);
                }

                // Boundary: Bottom (y = rows - 1)
                if (y === grid.rows - 1 && config.bottom !== BoundaryType.PERIODIC) {
                    this.applyEdgeCondition(chunk.faces, chunk.nx, chunk.ny, 'BOTTOM', config.bottom, config, faceIndicesToSync);
                }
            }
        }
    }

    private static applyEdgeCondition(
        faces: Float32Array[], nx: number, ny: number,
        edge: 'LEFT' | 'RIGHT' | 'TOP' | 'BOTTOM',
        type: BoundaryType,
        config: BoundaryConfig,
        facesToSync: number[]
    ) {
        // D2Q9 constants
        const cx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
        const cy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
        const w = [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];

        let targetX = -1;
        let targetY = -1;
        let loopLimit = 0;

        if (edge === 'LEFT') { targetX = 1; loopLimit = ny; }
        else if (edge === 'RIGHT') { targetX = nx - 2; loopLimit = ny; }
        else if (edge === 'TOP') { targetY = 1; loopLimit = nx; }
        else if (edge === 'BOTTOM') { targetY = ny - 2; loopLimit = nx; }

        if (type === BoundaryType.WALL) {
            // Half-Way Bounceback implies setting obstacles on the boundary
            if (faces[18]) {
                const obst = faces[18];
                for (let i = 0; i < loopLimit; i++) {
                    const idx = (targetX !== -1) ? i * nx + targetX : targetY * nx + i;
                    obst[idx] = 1.0;
                }
            }
        } else if (type === BoundaryType.INFLOW) {
            // Active momentum injection
            const u0 = config.inflowUx ?? 0.0;
            const v0 = config.inflowUy ?? 0.0;
            const rho0 = config.inflowDensity ?? 1.0;
            const u2 = u0 * u0 + v0 * v0;

            for (let i = 0; i < loopLimit; i++) {
                const idx = (targetX !== -1) ? i * nx + targetX : targetY * nx + i;

                if (faces[19]) faces[19][idx] = u0; // macro ux
                if (faces[20]) faces[20][idx] = v0; // macro uy
                if (faces[22]) faces[22][idx] = rho0; // macro rho

                // Reconstruct equilibrium
                for (let k = 0; k < 9; k++) {
                    const cu = 3 * (cx[k] * u0 + cy[k] * v0);
                    const feq = w[k] * rho0 * (1 + cu + 0.5 * cu * cu - 1.5 * u2);
                    if (faces[k]) faces[k][idx] = feq;
                    if (faces[k + 9]) faces[k + 9][idx] = feq;
                }
            }
        } else if (type === BoundaryType.OUTFLOW) {
            // Zero-gradient Condition (copy from neighbor)
            const nX = edge === 'LEFT' ? targetX + 1 : (edge === 'RIGHT' ? targetX - 1 : targetX);
            const nY = edge === 'TOP' ? targetY + 1 : (edge === 'BOTTOM' ? targetY - 1 : targetY);

            for (let i = 2; i < loopLimit - 2; i++) {
                const idx = (targetX !== -1) ? i * nx + targetX : targetY * nx + i;
                const neighborIdx = (targetX !== -1) ? i * nx + nX : nY * nx + i;

                // Copy macroscopic and microscopic distributions to mimic free-flow
                for (const faceIdx of facesToSync) {
                    if (faces[faceIdx]) faces[faceIdx][idx] = faces[faceIdx][neighborIdx];
                }
            }
        }
    }
}
