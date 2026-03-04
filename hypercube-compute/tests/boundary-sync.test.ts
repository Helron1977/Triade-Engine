import { describe, it, expect } from 'vitest';
import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { BlankEngine } from '../src/templates/BlankEngine'; // We can use BlankEngine for pure math testing

describe('HypercubeGrid Boundary Synchronization', () => {

    it('syncs correctly across a 2x1 horizontal chunk layout (Left-to-Right propagation)', async () => {
        const nx = 8, ny = 8, nz = 1;
        const totalFaces = 1;
        const master = new HypercubeMasterBuffer();

        // 2 cols, 1 row, non-periodic
        const grid = await HypercubeGrid.create(2, 1, { nx, ny, nz }, master, () => new BlankEngine(), totalFaces, false, 'cpu', false);

        const leftChunk = grid.cubes[0][0]!;
        const rightChunk = grid.cubes[0][1]!;

        const faceIdx = 0;

        // Populate the inner right-edge of the Left Chunk (x = nx - 2) with a specific marker
        for (let ly = 1; ly < ny - 1; ly++) {
            leftChunk.faces[faceIdx][leftChunk.getIndex(nx - 2, ly, 0)] = 99.0;
        }

        // Populate the inner left-edge of the Right Chunk (x = 1) with another marker
        for (let ly = 1; ly < ny - 1; ly++) {
            rightChunk.faces[faceIdx][rightChunk.getIndex(1, ly, 0)] = 88.0;
        }

        // Trigger manual synchronization of face 0
        (grid as any).synchronizeBoundaries(faceIdx);

        // Verify: The Left Chunk's right inner edge should have been copied to Right Chunk's left ghost cell (x = 0)
        for (let ly = 1; ly < ny - 1; ly++) {
            expect(rightChunk.faces[faceIdx][rightChunk.getIndex(0, ly, 0)]).toBe(99.0);
        }

        // Verify: The Right Chunk's left inner edge should have been copied to Left Chunk's right ghost cell (x = nx - 1)
        for (let ly = 1; ly < ny - 1; ly++) {
            expect(leftChunk.faces[faceIdx][leftChunk.getIndex(nx - 1, ly, 0)]).toBe(88.0);
        }
    });

    it('syncs correctly across a 1x2 vertical chunk layout (Top-to-Bottom array set propagation)', async () => {
        const nx = 8, ny = 8, nz = 1;
        const totalFaces = 1;
        const master = new HypercubeMasterBuffer();

        // 1 col, 2 rows, non-periodic
        const grid = await HypercubeGrid.create(1, 2, { nx, ny, nz }, master, () => new BlankEngine(), totalFaces, false, 'cpu', false);

        const topChunk = grid.cubes[0][0]!;
        const botChunk = grid.cubes[1][0]!;

        const faceIdx = 0;

        // Populate the inner bottom-edge of the Top Chunk (y = ny - 2)
        for (let lx = 1; lx < nx - 1; lx++) {
            topChunk.faces[faceIdx][topChunk.getIndex(lx, ny - 2, 0)] = 77.0;
        }

        // Populate the inner top-edge of the Bottom Chunk (y = 1)
        for (let lx = 1; lx < nx - 1; lx++) {
            botChunk.faces[faceIdx][botChunk.getIndex(lx, 1, 0)] = 66.0;
        }

        // Trigger manual synchronization of face 0
        (grid as any).synchronizeBoundaries(faceIdx);

        // Verify: The Top Chunk's inner bottom edge -> Bottom Chunk's top ghost cell (y = 0)
        for (let lx = 1; lx < nx - 1; lx++) {
            expect(botChunk.faces[faceIdx][botChunk.getIndex(lx, 0, 0)]).toBe(77.0);
        }

        // Verify: The Bottom Chunk's inner top edge -> Top Chunk's bottom ghost cell (y = ny - 1)
        for (let lx = 1; lx < nx - 1; lx++) {
            expect(topChunk.faces[faceIdx][topChunk.getIndex(lx, ny - 1, 0)]).toBe(66.0);
        }
    });
});
