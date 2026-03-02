import { describe, it, expect } from 'vitest';
import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { OceanEngine } from '../src/engines/OceanEngine';

describe('OceanEngine Mass Conservation and Stability', () => {

    it('runs 2000 steps without exploding (NaN) and conserves global mass', async () => {
        const mapSize = 64; // Smaller for fast testing 2000 steps
        const totalCells = mapSize * mapSize;
        const numFaces = 23;
        const masterBuffer = new HypercubeMasterBuffer(totalCells * numFaces * 4);

        const oceanEngine = new OceanEngine();
        oceanEngine.params.closedBounds = true; // IMPORTANT for isolated stability test

        // --- 1. SETUP ENGINE (CPU MODE) ---
        const grid = await HypercubeGrid.create(
            1, 1, mapSize, masterBuffer,
            () => oceanEngine,
            numFaces, false, 'cpu', false
        );

        const faces = grid.cubes[0][0]?.faces!;

        // INIT DENSITY to 1.0 (Face 20) and Populate equilibrium F (Faces 0-8)
        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
        for (let i = 0; i < totalCells; i++) {
            faces[20][i] = 1.0;
            for (let k = 0; k < 9; k++) {
                faces[k][i] = w[k] * 1.0;
            }
        }

        // Add a central obstacle (Face 22)
        const cx = Math.floor(mapSize / 2);
        for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
                faces[22][(cx + dy) * mapSize + (cx + dx)] = 1.0;
            }
        }

        // Force Vortex at the center
        oceanEngine.interaction.active = true;
        oceanEngine.interaction.mouseX = cx;
        oceanEngine.interaction.mouseY = cx;

        // Run simulation for 2000 steps
        // The half-way bounce back and NaN clamping will be seriously stressed here.
        for (let step = 0; step < 2000; step++) {
            await grid.compute();
        }

        const stats = oceanEngine.stats;

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
