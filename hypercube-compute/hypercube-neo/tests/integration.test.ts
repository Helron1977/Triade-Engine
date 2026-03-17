import { describe, it, expect } from 'vitest';
import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { HypercubeConfig, EngineDescriptor } from '../core/types';

describe('Hypercube Neo: Full Integration', () => {
    const factory = new HypercubeNeoFactory();

    const heatDescriptor: EngineDescriptor = {
        name: 'Heat-Neo',
        version: '1.0.0',
        faces: [{ name: 'temperature', type: 'scalar', isSynchronized: true }],
        parameters: { alpha: { name: 'Alpha', type: 'number', default: 0.1 } },
        rules: [
            { type: 'neo-heat', method: 'Custom', source: 'temperature', params: { omega: 1.7 } }
        ],
        outputs: [],
        requirements: { ghostCells: 1, pingPong: true }
    };

    const config: HypercubeConfig = {
        dimensions: { nx: 32, ny: 16, nz: 1 },
        chunks: { x: 2, y: 1 }, // Two 16x16 chunks
        boundaries: { all: { role: 'wall' } },
        engine: 'Heat-Neo',
        params: { alpha: 0.2 },
        objects: [
            {
                id: 'heat_source',
                type: 'circle',
                position: { x: 8, y: 8 }, // Well inside Chunk 0
                dimensions: { w: 4, h: 4 },
                properties: { temperature: 100 },
                rasterMode: 'replace'
            }
        ],
        mode: 'cpu'
    };

    it('should instantiate and run a simulation step correctly', async () => {
        const engine = await factory.build(config, heatDescriptor);

        // Run step 0: Rasterization + Diffusion
        await engine.step(0);

        const c0Views = engine.bridge.getChunkViews('chunk_0_0_0');
        const frameB = c0Views[1]; // Write buffer for temp face

        const pNx = 16 + 2;

        // 1. Check center (where Laplacian is 0)
        // x=10, y=10 -> px=11, py=11
        const centerIdx = 11 * pNx + 11;
        expect(frameB[centerIdx]).toBe(100);

        // 2. Check edge point (where Laplacian is non-zero)
        // x=7, y=10 -> px=8, py=11.
        // Neighbors: (6,10):0, (8,10):100, (7,9):0, (7,11):0.
        // Laplacian = 0 + 100 + 0 + 0 - 4*0 = 100.
        // dst = 0 + 0.2 * 100 = 20.
        const edgeIdx = 11 * pNx + 8;
        expect(frameB[edgeIdx]).toBe(24.75);

        // Run step 1: Buffer swap occurs.
        await engine.step(1);

        const frameA = c0Views[0];
        // Value should have diffused from edge to outer cells
        // x=6, y=10 -> px=7, py=11.
        const outerIdx = 11 * pNx + 7;
        expect(frameA[outerIdx]).toBeGreaterThan(0);
    });
});
