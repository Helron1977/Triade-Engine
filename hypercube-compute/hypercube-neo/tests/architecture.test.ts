import { describe, it, expect } from 'vitest';
import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { HypercubeConfig, EngineDescriptor } from '../core/types';

describe('Hypercube Neo: Core Architecture', () => {
    const factory = new HypercubeNeoFactory();

    const testDescriptor: EngineDescriptor = {
        name: 'Architecture-Test',
        version: '1.0.0',
        faces: [
            { name: 'density', type: 'scalar', isSynchronized: true },
            { name: 'vx', type: 'scalar', isSynchronized: true }
        ],
        parameters: {},
        rules: [
            { type: 'neo-heat', method: 'Custom', source: 'density', params: { omega: 1.7 } }
        ],
        outputs: [],
        requirements: { ghostCells: 1, pingPong: true }
    };

    it('should split object rasterization across chunk boundaries', async () => {
        const NX = 32;
        const NY = 16;
        const config: HypercubeConfig = {
            dimensions: { nx: NX, ny: NY, nz: 1 },
            chunks: { x: 2, y: 1 }, // Two 16x16 chunks
            boundaries: { all: { role: 'wall' } },
            engine: 'Architecture-Test',
            params: {},
            objects: [
                {
                    id: 'cross_blob',
                    type: 'rect',
                    position: { x: 12, y: 4 }, // Spans world X: [12, 12+8=20]
                    dimensions: { w: 8, h: 8 }, // Boundary is at world X=16
                    properties: { density: 1.0 },
                    rasterMode: 'replace'
                }
            ],
            mode: 'cpu'
        };

        const engine = await factory.build(config, testDescriptor);

        // Force one step to trigger rasterization and sync
        await engine.step(0);

        // We check the 'read' buffer of density (since rasterization should be an input for the NEXT physics step? 
        // Or is it part of the current step? In current implementation it's part of the step BEFORE compute)
        const dIdx = engine.parityManager.getFaceIndices('density').read;

        // Chunk 0 (Left): range [0, 16). X=12, 13, 14, 15 should have density.
        const chunk0 = engine.bridge.getChunkViews('chunk_0_0_0')[dIdx];
        // World X=12 is local X=12. Padded index for X=12 is 12 + 1 = 13.
        const valLeft = chunk0[8 * 18 + 13];

        // Chunk 1 (Right): range [16, 32). X=16, 17, 18, 19 should have density.
        const chunk1 = engine.bridge.getChunkViews('chunk_1_0_0')[dIdx];
        // World X=16 is local X=0. Padded index for X=0 is 0 + 1 = 1.
        const valRight = chunk1[8 * 18 + 1];

        console.log(`Rasterization Split: Left (X=12)=${valLeft}, Right (X=16)=${valRight}`);

        expect(valLeft).toBe(1.0);
        expect(valRight).toBe(1.0);
    });

    it('should maintain ghost cell consistency after rasterization', async () => {
        const NX = 32;
        const NY = 16;
        const config: HypercubeConfig = {
            dimensions: { nx: NX, ny: NY, nz: 1 },
            chunks: { x: 2, y: 1 },
            boundaries: { all: { role: 'wall' } },
            engine: 'Architecture-Test',
            params: {},
            objects: [
                {
                    id: 'blob',
                    type: 'rect',
                    position: { x: 15, y: 4 }, // Edge at X=15
                    dimensions: { w: 1, h: 8 },
                    properties: { density: 1.0 },
                    rasterMode: 'replace'
                }
            ],
            mode: 'cpu'
        };

        const engine = await factory.build(config, testDescriptor);
        await engine.step(0);

        const dIdx = engine.parityManager.getFaceIndices('density').read;

        // Chunk 1 (Right) has neighbors [16, 32).
        // Its LEFT ghost cell (local X=-1, padding=0) should have value from Chunk 0 (X=15).
        const chunk1 = engine.bridge.getChunkViews('chunk_1_0_0')[dIdx];
        const ghostVal = chunk1[8 * 18 + 0]; // Local X=-1 is index 0

        console.log(`Ghost Cell Consistency: Chunk1 Left Ghost=${ghostVal}`);

        expect(ghostVal).toBe(1.0);
    });
});
