import { describe, it, expect } from 'vitest';
import { NeoHeatmapKernel } from '../core/kernels/NeoHeatmapKernel';
import { HypercubeConfig, NumericalScheme } from '../core/types';
import { VirtualChunk } from '../core/topology/GridAbstractions';

describe('NeoHeatmapKernel (CPU Physics)', () => {
    const kernel = new NeoHeatmapKernel();
    const nx = 4;
    const ny = 4;
    const pNx = nx + 2;
    const pNy = ny + 2;
    const size = pNx * pNy;

    const gridConfig: HypercubeConfig = {
        dimensions: { nx, ny, nz: 1 },
        chunks: { x: 1, y: 1, z: 1 },
        mode: 'cpu'
    } as any;

    const chunk: VirtualChunk = {
        id: 'chunk-0-0',
        x: 0, y: 0, z: 0,
        localDimensions: { nx: 4, ny: 4, nz: 1 },
        joints: []
    };

    it('should diffuse heat correctly (4-point Laplacian)', () => {
        const uRead = new Float32Array(size);
        const uWrite = new Float32Array(size);
        const obstacles = new Float32Array(size);
        const injection = new Float32Array(size);

        // Center point at (2,2) in physical coordinates (1-based padding)
        const centerIdx = 2 * pNx + 2;
        uRead[centerIdx] = 1.0;

        const scheme: NumericalScheme = {
            type: 'neo-heat',
            source: 'temp',
            params: {
                diffusion_rate: 0.1,
                decay_factor: 1.0 // Disable decay for this test
            }
        };

        const indices = {
            'temp': { read: 0, write: 1 },
            'obstacles': { read: 2, write: 2 },
            'injection': { read: 3, write: 3 }
        };

        kernel.execute([uRead, uWrite, obstacles, injection], {
            nx, ny, pNx, pNy, padding: 1,
            scheme,
            indices,
            gridConfig,
            chunk,
            params: {}
        } as any);

        // Laplacian = (0+0+0+0) - 4*1.0 = -4.0
        // nextVal = 1.0 + 0.1 * (-4.0) = 0.6
        expect(uWrite[centerIdx]).toBeCloseTo(0.6);

        // Neighbors should receive heat
        expect(uWrite[centerIdx + 1]).toBeCloseTo(0.1);
        expect(uWrite[centerIdx - 1]).toBeCloseTo(0.1);
        expect(uWrite[centerIdx + pNx]).toBeCloseTo(0.1);
        expect(uWrite[centerIdx - pNx]).toBeCloseTo(0.1);
    });

    it('should respect obstacles', () => {
        const uRead = new Float32Array(size).fill(1.0);
        const uWrite = new Float32Array(size);
        const obstacles = new Float32Array(size);
        const injection = new Float32Array(size);

        const centerIdx = 2 * pNx + 2;
        obstacles[centerIdx] = 1.0;

        const indices = {
            'temp': { read: 0, write: 1 },
            'obstacles': { read: 2, write: 2 },
            'injection': { read: 3, write: 3 }
        };

        kernel.execute([uRead, uWrite, obstacles, injection], {
            nx, ny, pNx, pNy, padding: 1,
            scheme: { type: 'neo-heat', source: 'temp', params: { decay_factor: 1.0 } },
            indices, gridConfig, chunk, params: {}
        } as any);

        expect(uWrite[centerIdx]).toBe(0.0);
    });

    it('should respect continuous injection (Radiators)', () => {
        const uRead = new Float32Array(size);
        const uWrite = new Float32Array(size);
        const obstacles = new Float32Array(size);
        const injection = new Float32Array(size);

        const centerIdx = 2 * pNx + 2;
        injection[centerIdx] = 5.0; // Radiator set at 5.0

        const indices = {
            'temp': { read: 0, write: 1 },
            'obstacles': { read: 2, write: 2 },
            'injection': { read: 3, write: 3 }
        };

        kernel.execute([uRead, uWrite, obstacles, injection], {
            nx, ny, pNx, pNy, padding: 1,
            scheme: { type: 'neo-heat', source: 'temp', params: { decay_factor: 1.0 } },
            indices, gridConfig, chunk, params: {}
        } as any);

        expect(uWrite[centerIdx]).toBe(5.0);
    });

    it('should apply thermodynamic decay', () => {
        const uRead = new Float32Array(size).fill(1.0);
        const uWrite = new Float32Array(size);
        const obstacles = new Float32Array(size);
        const injection = new Float32Array(size);

        const indices = {
            'temp': { read: 0, write: 1 },
            'obstacles': { read: 2, write: 2 },
            'injection': { read: 3, write: 3 }
        };

        // If dt=0, only decay should apply
        kernel.execute([uRead, uWrite, obstacles, injection], {
            nx, ny, pNx, pNy, padding: 1,
            scheme: { type: 'neo-heat', source: 'temp', params: { diffusion_rate: 0, decay_factor: 0.9 } },
            indices, gridConfig, chunk, params: {}
        } as any);

        const centerIdx = 2 * pNx + 2;
        expect(uWrite[centerIdx]).toBeCloseTo(0.9);
    });
});
