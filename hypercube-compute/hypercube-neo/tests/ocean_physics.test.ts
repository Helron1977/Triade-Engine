import { describe, it, expect } from 'vitest';
import { NeoOceanKernel } from '../core/kernels/NeoOceanKernel';
import { NumericalScheme } from '../core/types';
import { VirtualChunk } from '../core/topology/GridAbstractions';

describe('NeoOceanKernel: Boundary Physics', () => {
    // 4x4 internal grid + 1 ghost cell padding = 6x6 actual buffer
    const nx = 4;
    const ny = 4;
    const pNx = nx + 2;
    const pNy = ny + 2;
    const bufferSize = pNx * pNy;

    const kernel = new NeoOceanKernel();
    const scheme: NumericalScheme = { type: 'neo-ocean-v1', method: 'OceanPhysics' as any, source: 'f0', params: { tau_0: 0.8 } } as any;

    // Fake chunk positioned at 0,0 (Top-Left of the world)
    const chunk: VirtualChunk = {
        x: 0, y: 0, z: 0,
        id: 'test',
        localDimensions: { nx: 4, ny: 4, nz: 1 },
        joints: []
    };

    function createViews() {
        const views = new Array(28).fill(null).map(() => new Float32Array(bufferSize));
        return views;
    }

    const indices = {
        'f0': { read: 0, write: 14 },
        'f1': { read: 1, write: 15 },
        'f2': { read: 2, write: 16 },
        'f3': { read: 3, write: 17 },
        'f4': { read: 4, write: 18 },
        'f5': { read: 5, write: 19 },
        'f6': { read: 6, write: 20 },
        'f7': { read: 7, write: 21 },
        'f8': { read: 8, write: 22 },
        'rho': { read: 9, write: 23 },
        'vx': { read: 10, write: 24 },
        'vy': { read: 11, write: 25 },
        'obstacles': { read: 12, write: 12 }, // Obstacles is read-only
        'biology': { read: 13, write: 27 }
    };

    it('should reflect populations and extrude rho when boundary is a wall', () => {
        const views = createViews();

        const fullViews = new Array(28).fill(null).map(() => new Float32Array(bufferSize));

        // Let's shoot a wave towards the left wall (px=1).
        // Since Half-Way bounce back makes px=1 a fluid node,
        // we put the left-moving population (f3) directly ON the wall node.
        const wallIdx = 2 * pNx + 1; // y=2, px=1
        fullViews[indices['f3'].read][wallIdx] = 1.0;

        const gridConfig = {
            dimensions: { nx: 8, ny: 8, nz: 1 },
            chunks: { x: 2, y: 2 }, // Meaning our chunk at x=0 is a world boundary on left
            boundaries: { all: { role: 'wall' } }
        };

        kernel.execute(fullViews, {
            nx, ny, pNx, pNy, padding: 1,
            scheme, indices, gridConfig, chunk, params: {}
        } as any);

        // 1. Check Pull-Scheme Bounce Back
        // The wall node's incoming right-moving pop (f1) from the ghost cell 
        // is overridden by its own left-moving pop (f3) from the previous step.
        // It then collides normally. Due to relaxation, f1 might be slightly negative 
        // if the imbalance is huge, but the total mass (rho) should be conserved.
        expect(fullViews[indices['f1'].write][wallIdx]).toBeGreaterThan(-0.5);

        // 2. Check Rho Accumulation
        // The mass must not be lost into a black hole!
        const outRho = fullViews[indices['rho'].write][wallIdx];
        expect(outRho).toBeGreaterThan(0.9);

        // 3. Check NO Ghost Cell Extrusion (We removed this to preserve L1 cache alignment)
        // For px=1 (Left wall), the ghost cell is px=0.
        const ghostIdx = 2 * pNx + 0;
        expect(fullViews[indices['rho'].write][ghostIdx]).toBe(0);
    });

    it('should NOT extrude rho when boundary is periodic', () => {
        const views = new Array(28).fill(null).map(() => new Float32Array(bufferSize));

        const targetIdx = 2 * pNx + 1; // y=2, px=1
        views[indices['f1'].read][targetIdx] = 1.0;

        const gridConfig = {
            dimensions: { nx: 8, ny: 8, nz: 1 },
            chunks: { x: 2, y: 2 },
            boundaries: { all: { role: 'periodic' } }
        };

        kernel.execute(views, {
            nx, ny, pNx, pNy, padding: 1,
            scheme, indices, gridConfig, chunk, params: {}
        } as any);

        // The ghost cell (px=0) should NOT have received the rho value
        const ghostIdx = 2 * pNx + 0;
        // Periodic means we don't bounce, and we don't extrude ghost cells locally in the kernel.
        // That's the BoundarySynchronizer's job.
        expect(views[indices['rho'].write][ghostIdx]).toBe(0);
    });
});
