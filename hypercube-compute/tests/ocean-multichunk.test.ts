import { describe, it, expect } from 'vitest';
import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { OceanEngine } from '../src/engines/OceanEngine';

describe('OceanEngine Multi-Chunk (Grid 2x2 Boundary Exchange)', () => {

    it('conserves mass exactly across 4 chunks with periodic boundaries enabled', async () => {
        // Here we test the engine inside a 2x2 Grid where internal ghosts are exchanged.
        const numChunksX = 2;
        const numChunksY = 2;
        const mapSize = 32; // MapSize per chunk
        const totalCellsStrided = mapSize * mapSize * numChunksX * numChunksY;
        const numFaces = 23;
        const masterBuffer = new HypercubeMasterBuffer(totalCellsStrided * numFaces * 4);

        const oceanEngine = new OceanEngine();

        // --- 1. SETUP ENGINE (CPU MODE) MULTI-CHUNK ---
        const grid = await HypercubeGrid.create(
            numChunksX, numChunksY, mapSize, masterBuffer,
            () => oceanEngine,
            numFaces, true, 'cpu', false // isPeriodic = true
        );

        const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];

        // INITIALIZE DENSITY ACROSS ALL CHUNKS to exactly 1.0 everywhere
        for (let cy = 0; cy < numChunksY; cy++) {
            for (let cx = 0; cx < numChunksX; cx++) {
                const faces = grid.cubes[cy][cx]?.faces!;
                for (let i = 0; i < mapSize * mapSize; i++) {
                    faces[20][i] = 1.0;          // rho
                    for (let k = 0; k < 9; k++) {
                        faces[k][i] = w[k] * 1.0;
                    }
                }
            }
        }

        // Apply forcing to create macro movement across chunk seams
        oceanEngine.interaction.active = true;
        oceanEngine.interaction.mouseX = 32; // Exactly at boundary
        oceanEngine.interaction.mouseY = 32;

        let totalMassStart = 0;
        for (let cy = 0; cy < numChunksY; cy++) {
            for (let cx = 0; cx < numChunksX; cx++) {
                const faces = grid.cubes[cy][cx]?.faces!;
                for (let y = 1; y < mapSize - 1; y++) {
                    for (let x = 1; x < mapSize - 1; x++) {
                        totalMassStart += faces[20][y * mapSize + x];
                    }
                }
            }
        }

        // SIMULATE
        for (let i = 0; i < 50; i++) {
            await grid.compute([0, 1, 2, 3, 4, 5, 6, 7, 8]); // MUST Sync ALL 9 LBM Populations
        }

        // CALCULATE FINAL MASS
        let totalMassEnd = 0;
        for (let cy = 0; cy < numChunksY; cy++) {
            for (let cx = 0; cx < numChunksX; cx++) {
                const faces = grid.cubes[cy][cx]?.faces!;
                for (let y = 1; y < mapSize - 1; y++) {
                    for (let x = 1; x < mapSize - 1; x++) {
                        totalMassEnd += faces[20][y * mapSize + x];
                    }
                }
            }
        }

        // For periodic Grid the mass is inherently closed!
        // Should be strictly equal, but floats can have tiny variations
        const diff = Math.abs(totalMassStart - totalMassEnd);
        console.log(`Multi-Chunk Mass conservation differencial: ${diff}`);

        expect(diff).toBeLessThan(0.1);
    });
});
