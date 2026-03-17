import { describe, it, expect } from 'vitest';
import { NeoSDFKernel } from '../core/kernels/NeoSDFKernel';
import { HypercubeConfig, NumericalScheme } from '../core/types';
import { VirtualChunk } from '../core/topology/GridAbstractions';

describe('Hypercube Neo: Math SDF Kernels', () => {
    const kernel = new NeoSDFKernel();
    const config: HypercubeConfig = {
        dimensions: { nx: 8, ny: 8, nz: 1 },
        chunks: { x: 1, y: 1 },
        boundaries: { all: { role: 'wall' } },
        engine: 'test',
        params: {}, mode: 'cpu'
    };

    it('should compute valid SDF for a simple seed', () => {
        const nx = 8, ny = 8, padding = 1;
        const pNx = nx + 2 * padding;
        const pNy = ny + 2 * padding;
        const size = pNx * pNy;

        const sdfX = new Float32Array(size).fill(-10000);
        const sdfY = new Float32Array(size).fill(-10000);
        
        // Seed at local coords (4,4) -> world (3,3)
        // px = 3 + 1 = 4, py = 3 + 1 = 4
        // The coordinate stored is the world coordinate (3,3)
        sdfX[4 * pNx + 4] = 3;
        sdfY[4 * pNx + 4] = 3;

        const views = [sdfX, sdfY];
        const scheme: NumericalScheme = { type: 'neo-sdf', source: 'sdf', params: {} };
        const indices = { 
            'sdf_x': { read: 0, write: 0 },
            'sdf_y': { read: 1, write: 1 }
        };

        const chunk: VirtualChunk = {
            id: 'test',
            x: 0, y: 0, z: 0,
            localDimensions: { nx: 100, ny: 100, nz: 1 },
            joints: []
        };

        kernel.execute(views, {
            nx: 8, ny: 8, pNx, pNy, padding: 1,
            scheme, indices, gridConfig: config, chunk, params: {}
        } as any);

        // Best seed at (4,4) should be (3,3)
        expect(sdfX[4 * pNx + 4]).toBe(3);
        expect(sdfY[4 * pNx + 4]).toBe(3);
        
        // At (5,4) -> world (4,3). Neighbors include (4,4) which has seed (3,3).
        // Distance from (4,3) to (3,3) is 1.
        expect(sdfX[4 * pNx + 5]).toBe(3);
        expect(sdfY[4 * pNx + 5]).toBe(3);
    });
});
