import { describe, it, expect } from 'vitest';
import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { HypercubeConfig, EngineDescriptor } from '../core/types';

describe('Hypercube Neo: Physics & Boundaries', () => {
    const factory = new HypercubeNeoFactory();

    it('should show density rising across chunk boundaries WITH WORLD-WIDE VELOCITY', async () => {
        const NX = 16;
        const NY = 32;

        const constVelDescriptor: EngineDescriptor = {
            name: 'Constant-Velocity-Test',
            version: '1.0.0',
            faces: [
                { name: 'density', type: 'scalar', isSynchronized: true },
                { name: 'vx', type: 'scalar', isSynchronized: true },
                { name: 'vy', type: 'scalar', isSynchronized: true }
            ],
            parameters: {},
            rules: [
                { type: 'advection', method: 'Semi-Lagrangian', source: 'density', field: 'v', params: { dt: 0.5 } }
            ],
            outputs: [],
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: NX, ny: NY, nz: 1 },
            chunks: { x: 1, y: 2 },
            boundaries: { all: { role: 'wall' } },
            engine: 'Constant-Velocity-Test',
            params: {},
            objects: [
                {
                    id: 'background_velocity',
                    type: 'rect',
                    position: { x: 0, y: 0 },
                    dimensions: { w: NX, h: NY },
                    properties: { vx: 0.0, vy: -2.0 }, // Constant upward flow everywhere
                    rasterMode: 'replace'
                },
                {
                    id: 'density_source',
                    type: 'rect',
                    position: { x: 4, y: 24 },
                    dimensions: { w: 8, h: 4 },
                    properties: { density: 1.0 },
                    rasterMode: 'replace'
                }
            ],
            mode: 'cpu'
        };

        const engine = await factory.build(config, constVelDescriptor);

        console.log("--- STARTING WORLD-WIDE VELOCITY CROSSING TEST ---");

        for (let step = 0; step < 40; step++) {
            await engine.step(step);

            const dIdx = engine.parityManager.getFaceIndices('density').read;
            const vyIdx = engine.parityManager.getFaceIndices('vy').read;

            const c0Faces = engine.bridge.getChunkViews('chunk_0_0_0');
            const c1Faces = engine.bridge.getChunkViews('chunk_0_1_0');

            const densityTop = c0Faces[dIdx];
            const vyTop = c0Faces[vyIdx];
            const vyBottom = c1Faces[vyIdx];

            let maxD0 = 0;
            for (let i = 0; i < densityTop.length; i++) if (densityTop[i] > maxD0) maxD0 = densityTop[i];

            if (step % 5 === 0) {
                console.log(`Step ${step}: MaxDensity Top=${maxD0.toFixed(3)}, vyTop[8,8]=${vyTop[8 * 18 + 8].toFixed(1)}, vyBottom[8,8]=${vyBottom[8 * 18 + 8].toFixed(1)}`);
            }
        }

        const dIdxFinal = engine.parityManager.getFaceIndices('density').read;
        const denTop = engine.bridge.getChunkViews('chunk_0_0_0')[dIdxFinal];

        let found = false;
        for (let i = 0; i < denTop.length; i++) {
            if (denTop[i] > 0.1) found = true;
        }
        expect(found).toBe(true);
    });
});
