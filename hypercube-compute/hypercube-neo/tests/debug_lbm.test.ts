import { describe, it, expect } from 'vitest';
import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { EngineDescriptor, HypercubeConfig } from '../core/types';

describe('LBM ZERO LEAKAGE DEBUGGER (Proxy)', () => {
    const factory = new HypercubeNeoFactory();

    const lbmDescriptor: EngineDescriptor = {
        name: 'LBM-D2Q9',
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
        rules: [{ type: 'lbm-d2q9', method: 'Custom', source: 'f0' }],
        requirements: { ghostCells: 1, pingPong: true },
        parameters: {}, outputs: []
    };

    it('Traces rho row by row for 5 steps via Factory', async () => {
        const NX = 8;
        const NY = 4;
        const config: HypercubeConfig = {
            dimensions: { nx: NX, ny: NY, nz: 1 },
            chunks: { x: 2, y: 1 },
            boundaries: { all: { role: 'wall' } },
            engine: 'LBM-D2Q9',
            params: { inflowUx: 0.15 },
            objects: [{
                id: "grid_init",
                type: "rect",
                position: { x: 0, y: 0 },
                dimensions: { w: 8, h: 4 },
                properties: {
                    f0: 4 / 9, f1: 1 / 9, f2: 1 / 9, f3: 1 / 9, f4: 1 / 9,
                    f5: 1 / 36, f6: 1 / 36, f7: 1 / 36, f8: 1 / 36
                },
                rasterMode: 'replace'
            }],
            mode: 'cpu',
            executionMode: 'sequential'
        };

        const engine = await factory.build(config, lbmDescriptor);

        const pNx = (NX / 2) + 2;
        const pNy = NY + 2;

        const dIdx = engine.parityManager.getFaceIndices('f0').read;

        for (let s = 0; s < 5; s++) {
            await engine.step(s);
            const chunk0 = engine.bridge.getChunkViews('chunk_0_0_0')[dIdx];
            const sumRow1 = Array.from({ length: pNx }).reduce((acc, _, i) => acc + chunk0[1 * pNx + i], 0);
            console.log(`Step ${s}: Chunk0 Row 1 Heat Sum = ${sumRow1}`);
        }

        const chunk1 = engine.bridge.getChunkViews('chunk_1_0_0')[dIdx];
        const sumRow1_C1 = Array.from({ length: pNx }).reduce((acc, _, i) => acc + chunk1[1 * pNx + i], 0);
        console.log(`Final Chunk1 Row 1 Heat Sum = ${sumRow1_C1}`);

        expect(sumRow1_C1).toBeGreaterThan(0);
    });
});
