import { describe, it, expect } from 'vitest';
import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { HypercubeConfig, EngineDescriptor } from '../core/types';

describe('Hypercube Neo: LBM Aerodynamics Port', () => {
    const factory = new HypercubeNeoFactory();

    const lbmDescriptor: EngineDescriptor = {
        name: 'Neo-LBM-D2Q9-Test',
        version: '1.0.0',
        faces: [
            { name: 'f0', type: 'scalar', isSynchronized: true },
            { name: 'f1', type: 'scalar', isSynchronized: true },
            { name: 'f2', type: 'scalar', isSynchronized: true },
            { name: 'f3', type: 'scalar', isSynchronized: true },
            { name: 'f4', type: 'scalar', isSynchronized: true },
            { name: 'f5', type: 'scalar', isSynchronized: true },
            { name: 'f6', type: 'scalar', isSynchronized: true },
            { name: 'f7', type: 'scalar', isSynchronized: true },
            { name: 'f8', type: 'scalar', isSynchronized: true },
            { name: 'obstacles', type: 'mask', isSynchronized: true, isReadOnly: true },
            { name: 'vx', type: 'scalar', isSynchronized: true },
            { name: 'vy', type: 'scalar', isSynchronized: true },
            { name: 'density', type: 'scalar', isSynchronized: true },
            { name: 'smoke', type: 'scalar', isSynchronized: true }
        ],
        parameters: {},
        rules: [
            { type: 'lbm-d2q9', method: 'Custom', source: 'f0', params: { omega: 1.0 } }, // Use omega=1.0 (instant equilibrium) for test stability
            { type: 'lbm-macro', method: 'Direct', source: 'f0' },
            { type: 'lbm-smoke', method: 'Semi-Lagrangian', source: 'smoke', params: { dissipation: 1.0, diffusion: 0.0 } }
        ],
        outputs: [],
        requirements: { ghostCells: 1, pingPong: true }
    };

    it('should stream smoke across chunk boundaries in a constant LBM flow', async () => {
        const NX = 16;
        const NY = 32;

        const config: HypercubeConfig = {
            dimensions: { nx: NX, ny: NY, nz: 1 },
            chunks: { x: 1, y: 2 },
            boundaries: { all: { role: 'wall' } },
            engine: 'Neo-LBM-D2Q9-Test',
            params: {},
            objects: [
                {
                    id: 'smoke_source',
                    type: 'rect',
                    position: { x: 4, y: 24 },
                    dimensions: { w: 8, h: 4 },
                    properties: { smoke: 1.0 },
                    rasterMode: 'replace'
                },
                {
                    id: 'velocity_init',
                    type: 'rect',
                    position: { x: 0, y: 0 },
                    dimensions: { w: NX, h: NY },
                    // Manually shifted equilibrium for upward flow (uy = -0.5)
                    // F4 (S) gets more, F2 (N) gets less in PULL architecture
                    properties: {
                        f0: 0.444, f1: 0.111, f2: 0.05, f3: 0.111, f4: 0.20,
                        f5: 0.015, f6: 0.015, f7: 0.04, f8: 0.04
                    },
                    rasterMode: 'replace'
                }
            ],
            mode: 'cpu'
        };

        const engine = await factory.build(config, lbmDescriptor);

        console.log("--- STARTING LBM DIAGNOSTIC TEST ---");

        for (let step = 0; step < 40; step++) {
            await engine.step(step);

            const vyIdx = engine.parityManager.getFaceIndices('vy').read;
            const sIdx = engine.parityManager.getFaceIndices('smoke').read;

            const c0Faces = engine.bridge.getChunkViews('chunk_0_0_0');
            const c1Faces = engine.bridge.getChunkViews('chunk_0_1_0');

            const vyTop = c0Faces[vyIdx];
            const vyBottom = c1Faces[vyIdx];
            const smokeTop = c0Faces[sIdx];

            let maxS0 = 0;
            for (let i = 0; i < smokeTop.length; i++) if (smokeTop[i] > maxS0) maxS0 = smokeTop[i];

            if (step % 5 === 0) {
                console.log(`Step ${step}: MaxSmoke Top=${maxS0.toFixed(3)}, vy[Bottom]=${vyBottom[8 * 18 + 8].toFixed(3)}, vy[Top]=${vyTop[8 * 18 + 8].toFixed(3)}`);
            }
        }

        const sIdxFinal = engine.parityManager.getFaceIndices('smoke').read;
        const smokeTopFinal = engine.bridge.getChunkViews('chunk_0_0_0')[sIdxFinal];

        let found = false;
        for (let i = 0; i < smokeTopFinal.length; i++) {
            if (smokeTopFinal[i] > 0.05) found = true;
        }
        expect(found).toBe(true);
    });
});
