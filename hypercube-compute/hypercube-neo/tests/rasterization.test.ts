import { describe, it, expect } from 'vitest';
import { VirtualGrid } from '../core/topology/VirtualGrid';
import { MasterBuffer } from '../core/memory/MasterBuffer';
import { ObjectRasterizer } from '../core/rasterization/ObjectRasterizer';
import { CpuBufferBridge } from '../core/memory/CpuBufferBridge';
import { EngineDescriptor, HypercubeConfig } from '../core/types';

describe('Hypercube Neo: Real-time Rasterization', () => {
    const urbanDescriptor: EngineDescriptor = {
        name: 'UrbanHeat',
        version: '1.0.0',
        faces: [{ name: 'heat', type: 'field', isSynchronized: true }],
        parameters: {},
        rules: [],
        outputs: [],
        requirements: { ghostCells: 1, pingPong: true }
    };

    it('should additively rasterize multiple overlapping influence zones', () => {
        const config: HypercubeConfig = {
            dimensions: { nx: 16, ny: 16, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: { all: { role: 'wall' } },
            engine: 'UrbanHeat',
            params: {},
            objects: [
                {
                    id: 'tree_1',
                    type: 'circle',
                    position: { x: 8, y: 8 },
                    dimensions: { w: 0, h: 0 }, // Point source
                    influence: { falloff: 'step', radius: 4 },
                    properties: { heat: 10 },
                    rasterMode: 'add'
                },
                {
                    id: 'tree_2',
                    type: 'circle',
                    position: { x: 10, y: 8 },
                    dimensions: { w: 0, h: 0 },
                    influence: { falloff: 'step', radius: 4 },
                    properties: { heat: 5 },
                    rasterMode: 'add'
                }
            ],
            mode: 'cpu'
        };

        const vGrid = new VirtualGrid(config, urbanDescriptor);
        const mBuffer = new MasterBuffer(vGrid);
        const bridge = new CpuBufferBridge(mBuffer);
        const rasterizer = new ObjectRasterizer();

        rasterizer.rasterizeChunk(vGrid.chunks[0], vGrid, bridge, 0);

        const views = mBuffer.getChunkViews('chunk_0_0_0');
        const heatView = views.faces[0];
        const pNx = 18;

        // At (9, 8) in world coords -> (10, 9) in padded local coords
        // Both trees cover (9,8). Tree 1 adds 10, Tree 2 adds 5. Total = 15.
        const valAt9_8 = heatView[9 * pNx + 10];
        expect(valAt9_8).toBe(15);

        // At (13, 8) -> Only Tree 2 (10+4=14) should cover it. Tree 1 (8+4=12) doesn't.
        const valAt13_8 = heatView[9 * pNx + 14];
        expect(valAt13_8).toBe(5);
    });

    it('should rasterize dynamic objects at their time-offset position', () => {
        const config: HypercubeConfig = {
            dimensions: { nx: 16, ny: 16, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: { all: { role: 'wall' } },
            engine: 'UrbanHeat',
            params: {},
            objects: [
                {
                    id: 'moving_probe',
                    type: 'circle',
                    position: { x: 0, y: 8 },
                    dimensions: { w: 2, h: 2 },
                    animation: { velocity: { x: 10, y: 0 } }, // Moves 10 units/sec
                    properties: { heat: 100 },
                    rasterMode: 'replace'
                }
            ],
            mode: 'cpu'
        };

        const vGrid = new VirtualGrid(config, urbanDescriptor);
        const mBuffer = new MasterBuffer(vGrid);
        const bridge = new CpuBufferBridge(mBuffer);
        const rasterizer = new ObjectRasterizer();

        // At t=1, position should be x=10
        rasterizer.rasterizeChunk(vGrid.chunks[0], vGrid, bridge, 1.0);

        const views = mBuffer.getChunkViews('chunk_0_0_0');
        const heatView = views.faces[0];
        const pNx = 18;

        // Check at center: x=11 (local 12), y=9 (local 10)
        expect(heatView[10 * pNx + 12]).toBe(100);
        // Check at original x=0 (local 1)
        expect(heatView[9 * pNx + 1]).toBe(0);
    });
});
