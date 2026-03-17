import { describe, it, expect } from 'vitest';
import { VirtualGrid } from '../core/topology/VirtualGrid';
import { MasterBuffer } from '../core/MasterBuffer';
import { CpuBufferBridge } from '../core/CpuBufferBridge';
import { BoundarySynchronizer } from '../core/topology/BoundarySynchronizer';
import { EngineDescriptor, HypercubeConfig } from '../core/types';

describe('Hypercube Neo: Boundary Synchronization', () => {
    const lbmDescriptor: EngineDescriptor = {
        name: 'LBM-D2Q9',
        version: '1.0.0',
        faces: [{ name: 'fi', type: 'population', isSynchronized: true }],
        parameters: {},
        rules: [],
        outputs: [],
        requirements: { ghostCells: 1, pingPong: true }
    };

    const config: HypercubeConfig = {
        dimensions: { nx: 32, ny: 32, nz: 1 },
        chunks: { x: 2, y: 2 }, // 2x2 grid of 16x16 chunks
        boundaries: { all: { role: 'wall' } },
        engine: 'LBM-D2Q9',
        params: {},
        mode: 'cpu'
    };

    const vGrid = new VirtualGrid(config, lbmDescriptor);
    const mBuffer = new MasterBuffer(vGrid);
    const synchronizer = new BoundarySynchronizer();

    it('should transfer data across direct faces (Left/Right)', () => {
        const bridge = new CpuBufferBridge(mBuffer);
        const c0_0 = bridge.getChunkViews('chunk_0_0_0');
        const c1_0 = bridge.getChunkViews('chunk_1_0_0');

        const fi_0 = c0_0[0];
        const fi_1 = c1_0[0];

        // nx=16, pNx=18. 
        // c0_0 is at x=0..15. its Right Real is x=16. 
        // c1_0 is at x=16..31. its Left Ghost is x=0.

        // Write to c0_0's Right Real boundary cell (16, 8)
        const pNx = 18;
        fi_0[8 * pNx + 16] = 42;

        synchronizer.syncAll(vGrid, bridge);

        // c1_0's Left Ghost (0, 8) should now have 42
        expect(fi_1[8 * pNx + 0]).toBe(42);
    });

    it('should transfer data across diagonal corners (The "Fool-Proof" Test)', () => {
        const bridge = new CpuBufferBridge(mBuffer);
        const c0_0 = bridge.getChunkViews('chunk_0_0_0'); // Top-Left chunk
        const c1_1 = bridge.getChunkViews('chunk_1_1_0'); // Bottom-Right chunk

        const fi_0_0 = c0_0[0];
        const fi_1_1 = c1_1[0];

        const pNx = 18;
        const nx = 16;
        const ny = 16;

        // c0_0's Bottom-Right Real cell is at (nx, ny) = (16, 16)
        fi_0_0[ny * pNx + nx] = 99;

        synchronizer.syncAll(vGrid, bridge);

        // c1_1's Top-Left Ghost cell is at (0, 0)
        // It should have received 99 from c0_0
    });
});

describe('Hypercube Neo: Periodic Boundaries', () => {
    const lbmDescriptor: EngineDescriptor = {
        name: 'LBM-D2Q9',
        version: '1.0.0',
        faces: [{ name: 'fi', type: 'population', isSynchronized: true }],
        parameters: {},
        rules: [],
        outputs: [],
        requirements: { ghostCells: 1, pingPong: true }
    };

    const config: HypercubeConfig = {
        dimensions: { nx: 32, ny: 32, nz: 1 },
        chunks: { x: 2, y: 2 }, // 2x2 grid of 16x16 chunks
        boundaries: { all: { role: 'periodic' } },
        engine: 'LBM-D2Q9',
        params: {},
        mode: 'cpu'
    };

    const vGrid = new VirtualGrid(config, lbmDescriptor);
    const mBuffer = new MasterBuffer(vGrid);
    const synchronizer = new BoundarySynchronizer();

    it('should seamlessly wrap left/right world edges', () => {
        const bridge = new CpuBufferBridge(mBuffer);
        const c0_0 = bridge.getChunkViews('chunk_0_0_0'); // Top-Left
        const c1_0 = bridge.getChunkViews('chunk_1_0_0'); // Top-Right

        const fi_0 = c0_0[0];
        const fi_1 = c1_0[0];

        // Write to Rightmost real cell of c1_0
        const nx = 16;
        const pNx = 18;
        fi_1[8 * pNx + nx] = 77;

        synchronizer.syncAll(vGrid, bridge);

        // c0_0's Leftmost ghost cell (0) should receive it
        expect(fi_0[8 * pNx + 0]).toBe(77);
    });

    it('should seamlessly wrap diagonal world corners', () => {
        const bridge = new CpuBufferBridge(mBuffer);
        const c0_0 = bridge.getChunkViews('chunk_0_0_0'); // Top-Left
        const c1_1 = bridge.getChunkViews('chunk_1_1_0'); // Bottom-Right

        const fi_0 = c0_0[0];
        const fi_1_1 = c1_1[0];

        const nx = 16;
        const ny = 16;
        const pNx = 18;

        // Write to Bottom-Right real cell of the entire world (c1_1)
        fi_1_1[ny * pNx + nx] = 88;

        synchronizer.syncAll(vGrid, bridge);

        // It should wrap around to the Top-Left ghost cell of the entire world (c0_0)
        expect(fi_0[0 * pNx + 0]).toBe(88);
    });
});
