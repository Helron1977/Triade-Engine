import { describe, it, expect } from 'vitest';
import { VirtualGrid } from '../core/topology/VirtualGrid';
import { MasterBuffer } from '../core/memory/MasterBuffer';
import { EngineDescriptor, HypercubeConfig } from '../core/types';

describe('Hypercube Neo: Physical MasterBuffer', () => {
    const lbmDescriptor: EngineDescriptor = {
        name: 'LBM-D2Q9',
        version: '1.0.0',
        faces: [
            { name: 'fi', type: 'population', isSynchronized: true }, // Should be Ping-Pong
            { name: 'rho', type: 'macro', isSynchronized: false }    // Should be Single
        ],
        parameters: { viscosity: { name: 'Viscosity', type: 'number', default: 0.1 } },
        rules: [{ type: 'lbm-d2q9', method: 'Custom', source: 'fi' }],
        outputs: [],
        requirements: { ghostCells: 1, pingPong: true }
    };

    const config: HypercubeConfig = {
        dimensions: { nx: 16, ny: 16, nz: 1 },
        chunks: { x: 1, y: 1 },
        boundaries: { all: { role: 'wall' } },
        engine: 'LBM-D2Q9',
        params: { viscosity: 0.1 },
        mode: 'cpu'
    };

    it('should allocate the correct total byte length', () => {
        const vGrid = new VirtualGrid(config, lbmDescriptor);
        const mBuffer = new MasterBuffer(vGrid);
        // Physical cells: (16+2)*(16+2) = 324
        // Bytes per face raw: 324 * 4 = 1296
        // Aligned per face: 1536 (next multiple of 256)
        // Total buffers = 3. 
        // 3 * 1536 bytes = 4608 bytes.
        expect(mBuffer.byteLength).toBe(4608);
    });

    it('should provide zero-copy views into the same buffer', () => {
        const vGrid = new VirtualGrid(config, lbmDescriptor);
        const mBuffer = new MasterBuffer(vGrid);
        const chunk = mBuffer.getChunkViews('chunk_0_0_0');
        expect(chunk.faces.length).toBe(3);

        const fi_A = chunk.faces[0];
        const fi_B = chunk.faces[1];
        const rho = chunk.faces[2];

        // Verify sharing
        expect(fi_A.buffer).toBe(mBuffer.rawBuffer);
        expect(fi_B.buffer).toBe(mBuffer.rawBuffer);
        expect(rho.buffer).toBe(mBuffer.rawBuffer);

        // Verify isolation (different offsets)
        fi_A[0] = 777;
        expect(fi_B[0]).toBe(0);
        expect(rho[0]).toBe(0);

        // Verify write through to raw buffer
        const rawView = new Float32Array(mBuffer.rawBuffer);
        expect(rawView[0]).toBe(777);
    });
});
