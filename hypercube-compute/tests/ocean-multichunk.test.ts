import { describe, it, expect } from 'vitest';
import { HypercubeCpuGrid } from '../src/core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { OceanEngine } from '../src/engines/OceanEngine';

describe('OceanEngine Multi-Chunk (Grid 2x2 Boundary Exchange)', () => {

    it('conserves mass exactly across 4 chunks with periodic boundaries enabled', async () => {
        const numChunksX = 2;
        const numChunksY = 2;
        const mapSize = 32;
        const numFaces = 25;
        const masterBuffer = new HypercubeMasterBuffer(10 * 1024 * 1024);

        // 1. Create Grid
        const grid = await HypercubeCpuGrid.create(
            numChunksX, numChunksY, mapSize, masterBuffer,
            () => new OceanEngine(),
            numFaces, true, true, undefined, 'cpu'
        );

        // 2. Initialize with a splash
        grid.applyEquilibrium(16, 16, 0, 8, 1.8, 0.5, 0.5);

        let totalMassStart = 0;
        for (const c of grid.cubes.flat()) {
            const rho = c!.faces[22];
            for (let ly = 1; ly < c!.ny - 1; ly++) {
                for (let lx = 1; lx < c!.nx - 1; lx++) {
                    totalMassStart += rho[c!.getIndex(lx, ly)];
                }
            }
        }

        // 3. Compute 1000 steps
        // NOTE: In Vitest/Node, Worker may be missing, so it falls back to single-threaded.
        for (let i = 0; i < 1000; i++) {
            await grid.compute();
        }

        // 4. Verify Mass Conservation & Movement
        let totalMassEnd = 0;
        let movingCells = 0;
        for (const c of grid.cubes.flat()) {
            const rho = c!.faces[22];
            for (let ly = 1; ly < c!.ny - 1; ly++) {
                for (let lx = 1; lx < c!.nx - 1; lx++) {
                    const val = rho[c!.getIndex(lx, ly)];
                    totalMassEnd += val;
                    if (Math.abs(val - 1.0) > 0.0001) movingCells++;
                }
            }
        }

        const diff = Math.abs(totalMassStart - totalMassEnd);
        console.log(`Final Mass conservation diff: ${diff.toFixed(6)}`);
        console.log(`Final Moving cells: ${movingCells}`);

        expect(diff).toBeLessThan(0.1);
        expect(movingCells).toBeGreaterThan(0);
    });
});
