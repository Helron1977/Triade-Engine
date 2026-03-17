import { describe, it, expect } from 'vitest';
import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { EngineDescriptor, HypercubeConfig } from '../core/types';

describe('Hypercube Neo: Declarative Configuration Validation', () => {

    const aeroDescriptor: EngineDescriptor = {
        name: 'Config-Test', version: '1.0.0',
        faces: [
            { name: 'f0', type: 'scalar', isSynchronized: true, isPersistent: false },
            { name: 'f1', type: 'scalar', isSynchronized: true, isPersistent: false },
            { name: 'f2', type: 'scalar', isSynchronized: true, isPersistent: false },
            { name: 'f3', type: 'scalar', isSynchronized: true, isPersistent: false },
            { name: 'f4', type: 'scalar', isSynchronized: true, isPersistent: false },
            { name: 'f5', type: 'scalar', isSynchronized: true, isPersistent: false },
            { name: 'f6', type: 'scalar', isSynchronized: true, isPersistent: false },
            { name: 'f7', type: 'scalar', isSynchronized: true, isPersistent: false },
            { name: 'f8', type: 'scalar', isSynchronized: true, isPersistent: false },
            { name: 'obstacles', type: 'mask', isSynchronized: true, isReadOnly: true },
            { name: 'vx', type: 'scalar', isSynchronized: true },
            { name: 'vy', type: 'scalar', isSynchronized: true },
            { name: 'vorticity', type: 'scalar', isSynchronized: true },
            { name: 'smoke', type: 'scalar', isSynchronized: true }
        ],
        rules: [{ type: 'lbm-aero-fidelity-v1', source: 'f0' }],
        requirements: { ghostCells: 1, pingPong: true },
        parameters: {}, outputs: []
    };

    it('should respect dynamic grid dimensions and chunking', async () => {
        const factory = new HypercubeNeoFactory();
        const config: HypercubeConfig = {
            dimensions: { nx: 128, ny: 256, nz: 1 },
            chunks: { x: 4, y: 1 }, // 128/4 = 32 per chunk width
            boundaries: { all: { role: 'wall' } },
            engine: 'Config-Test',
            params: {},
            mode: 'cpu'
        };

        const neo = await factory.build(config, aeroDescriptor);
        expect(neo.vGrid.chunks.length).toBe(4);

        const chunk0 = neo.vGrid.chunks[0];
        // Total view size per face = (nx/chunks.x + 2 padding) * (ny/chunks.y + 2 padding)
        // (32 + 2) * (256 + 2) = 34 * 258 = 8772
        const pViews = neo.bridge.getChunkViews(chunk0.id);
        expect(pViews[0].length).toBe(34 * 258);
    });

    it('should correctly rasterize VirtualObjects into chunks', async () => {
        const factory = new HypercubeNeoFactory();
        const config: HypercubeConfig = {
            dimensions: { nx: 512, ny: 512, nz: 1 },
            chunks: { x: 2, y: 2 },
            boundaries: { all: { role: 'wall' } },
            engine: 'Config-Test',
            params: {},
            mode: 'cpu',
            objects: [
                {
                    id: 'test-circle',
                    type: 'circle',
                    position: { x: 10, y: 10 },
                    dimensions: { w: 20, h: 20 },
                    influence: { radius: 10, falloff: 'step' },
                    properties: { isObstacle: 1 }
                }
            ]
        };

        const neo = await factory.build(config, aeroDescriptor);

        // Object at (10,10) should be in Chunk(0,0)
        const chunk00 = neo.vGrid.findChunkAt(0, 0)!;
        const objectsIn00 = neo.vGrid.getObjectsInChunk(chunk00);
        expect(objectsIn00.length).toBe(1);
        expect(objectsIn00[0].id).toBe('test-circle');

        // Object at (10,10) should NOT be in Chunk(1,1) (starts at 256, 256)
        const chunk11 = neo.vGrid.findChunkAt(1, 1)!;
        const objectsIn11 = neo.vGrid.getObjectsInChunk(chunk11);
        expect(objectsIn11.length).toBe(0);
    });
});
