import { describe, it, expect, beforeEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { NeoEngineProxy } from '../core/NeoEngineProxy';

describe('Showcase Heatmap UI & Rendering Logic', () => {
    let engine: NeoEngineProxy;
    let manifest: any;

    beforeEach(async () => {
        const manifestPath = path.resolve(__dirname, '../showcase/showcase-heat-gpu.json');
        const content = fs.readFileSync(manifestPath, 'utf8');
        manifest = JSON.parse(content);

        // Force sequential mode for Node.js Vitest environment (no window.Worker)
        manifest.config.executionMode = 'sequential';
        manifest.config.mode = 'cpu';

        const factory = new HypercubeNeoFactory();
        engine = await factory.build(manifest.config, manifest.engine);
    });

    it('should correctly resolve face indices through ParityManager for rendering', () => {
        // Simulates the exact logic on line 165 of heatmap.ts
        const faceIdx = engine.getFaceLogicalIndex('temperature');
        expect(faceIdx).toBeDefined();

        const phys = engine.parityManager.getFaceIndices('temperature');
        expect(phys.read).toBeGreaterThanOrEqual(0);
    });

    it('should reliably inject global masks across 2x2 chunks preventing IndexOutOfBounds', () => {
        // Simulate the injectGlobalMask helper
        const injectGlobalMask = (engine: NeoEngineProxy, faceName: string, globalData: Float32Array, size: number) => {
            const vGrid = engine.vGrid;
            const nChunksX = vGrid.config.chunks.x;
            const nChunksY = vGrid.config.chunks.y;
            const vnx = size / nChunksX;
            const vny = size / nChunksY;

            for (const chunk of vGrid.chunks) {
                const nxPhys = vnx + 2;
                const nyPhys = vny + 2;
                const chunkData = new Float32Array(nxPhys * nyPhys);

                const worldXOffset = chunk.x * vnx;
                const worldYOffset = chunk.y * vny;

                for (let ly = 1; ly < nyPhys - 1; ly++) {
                    const worldY = worldYOffset + (ly - 1);
                    const dstRowOffset = ly * nxPhys;
                    const srcRowOffset = worldY * size;

                    for (let lx = 1; lx < nxPhys - 1; lx++) {
                        const worldX = worldXOffset + (lx - 1);
                        chunkData[dstRowOffset + lx] = globalData[srcRowOffset + worldX];
                    }
                }

                // If setFaceData survives, mapping is successful
                expect(() => engine.bridge.setFaceData(chunk.id, faceName, chunkData)).not.toThrow();
            }
        };

        const size = 256;
        const dummyInjectionMask = new Float32Array(size * size).fill(1.5);

        injectGlobalMask(engine, 'temperature', dummyInjectionMask, size);

        // Validate mask made it to the memory buffer array
        const fIdx = engine.parityManager.getFaceIndices('temperature').read;
        const chunk0 = engine.bridge.getChunkViews('chunk_0_0_0')[fIdx];

        // Assert the injection values were placed into the grid
        expect(chunk0).toBeDefined();
        // Since the inner cells were written, find one and assert it is 1.5
        // Inner cell (1,1) -> row_offset = 1*(256+2) = 258. Cell = 258+1 = 259
        expect(chunk0[259]).toBe(1.5);
    });
});
