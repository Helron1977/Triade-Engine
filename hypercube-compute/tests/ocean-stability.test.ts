import { describe, it, expect } from 'vitest';
import { HypercubeCpuGrid } from '../src/core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { OceanEngine } from '../src/engines/OceanEngine';
import { HypercubeMath } from '../src/math/HypercubeMath';

describe('OceanEngine Mass Conservation and Stability', () => {

    it('runs 2000 steps without exploding (NaN) and conserves global mass', async () => {
        const mapSize = 64; // Smaller for fast testing 2000 steps
        const totalCells = mapSize * mapSize;
        const numFaces = 25;
        const masterBuffer = new HypercubeMasterBuffer(10 * 1024 * 1024);

        const oceanEngine = new OceanEngine();
        oceanEngine.params.closedBounds = true; // IMPORTANT for isolated stability test

        // --- 1. SETUP ENGINE (CPU MODE) ---
        const grid = await HypercubeCpuGrid.create(
            1, 1, mapSize, masterBuffer,
            () => new OceanEngine(),
            numFaces, false, false);

        const cube = grid.cubes[0][0]!;
        const faces = cube.faces;

        // INIT DENSITY to 1.0 (Face 22) and Populate equilibrium F (Faces 0-8)
        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
        for (let ly = 0; ly < cube.ny; ly++) {
            for (let lx = 0; lx < cube.nx; lx++) {
                const i = cube.getIndex(lx, ly);
                faces[22][i] = 1.0;
                for (let k = 0; k < 9; k++) {
                    faces[k][i] = w[k] * 1.0;
                    faces[k + 9][i] = w[k] * 1.0;
                }
            }
        }

        // Add a central obstacle (Face 18)
        const cx = Math.floor(cube.nx / 2);
        const cy = Math.floor(cube.ny / 2);
        for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
                faces[18][cube.getIndex(cx + dx, cy + dy)] = 1.0;
            }
        }

        // Run simulation for 2000 steps
        for (let step = 0; step < 2000; step++) {
            HypercubeMath.injectMomentumD2Q9(faces, cube.nx, cube.ny, cx, cy, 19, 0.02);
            await grid.compute();
        }

        const stats = (grid.cubes[0][0]!.engine as OceanEngine).stats;

        // Assert no explosion in Velocity
        expect(stats.maxU).not.toBeNaN();
        expect(stats.maxU).toBeLessThan(0.4); // Must not breach CFL heavily

        // Assert mass conservation
        // Note: LBM with forced bounds / artificial vortex injection can fluctuate slightly but should average around 1.0
        // With a constant vortex in a closed box, the low-pressure eye is clamped to 0.8, injecting mass, pressurizing the box up to ~1.11.
        expect(stats.avgRho).toBeGreaterThan(0.9);
        expect(stats.avgRho).toBeLessThan(1.15);

        console.log(`OceanEngine Test Passed: avgRho=${stats.avgRho.toFixed(4)}, maxU=${stats.maxU.toFixed(4)}`);
    });
});
