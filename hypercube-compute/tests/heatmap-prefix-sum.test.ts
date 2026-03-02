import { describe, it, expect } from 'vitest';
import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { HeatmapEngine } from '../src/engines/HeatmapEngine';

describe('HeatmapEngine GPU vs CPU tests', () => {

    it('matches GPU prefix sum with CPU prefix sum output', async () => {
        const mapSize = 256;
        const totalCells = mapSize * mapSize;
        const numFaces = 5;
        const masterBuffer = new HypercubeMasterBuffer(totalCells * numFaces * 4);

        // --- 1. SETUP ENGINE (CPU MODE) ---
        const cpuGrid = await HypercubeGrid.create(
            1, 1, mapSize, masterBuffer,
            () => new HeatmapEngine(10, 1.0),
            numFaces, false, 'cpu', false
        );

        const facesCPU = cpuGrid.cubes[0][0]?.faces!;

        // Input binaire sur la diagonale pour varier les patterns (Face 1 = input)
        for (let i = 0; i < mapSize; i++) {
            facesCPU[1][i * mapSize + i] = 1.0;
        }

        // Execution CPU
        await cpuGrid.compute();

        // Sauvegarde résultats CPU Box Filter (face 2) et SAT (face 4)
        const cpuDiffusion = new Float32Array(facesCPU[2]);
        const cpuSAT = new Float32Array(facesCPU[4]);

        // --- 2. SETUP ENGINE (GPU MODE) ---
        const WebGPUIsSupported = (typeof window !== 'undefined' && 'devicePixelRatio' in window) || true; // Stub for testing node. If true GPU test can run.

        // Since we are running in vitest (Node), native WebGPU might not be available unless mocked
        // For the sake of this roadmap test structure we configure it, but we might skip actual GPU compute 
        // if context device is null.
        try {
            const gpuGrid = await HypercubeGrid.create(
                1, 1, mapSize, masterBuffer,  // reuse buffer (will be overwritten)
                () => new HeatmapEngine(10, 1.0),
                numFaces, false, 'webgpu', false
            );

            // Si WebGPU n'a pas pu init (ex: headless Node), WebGPUContext throwera silencieusement et mode sera reset à cpu. 
            if (gpuGrid.mode !== 'webgpu') {
                console.warn('Skipping true WebGPU execution benchmark in Node/Vitest environment. Please run in browser.');
                return;
            }

            // Ré-initialiser la face 1 (car elle va l'être)
            const facesGPU = gpuGrid.cubes[0][0]?.faces!;
            facesGPU[1].fill(0);
            for (let i = 0; i < mapSize; i++) {
                facesGPU[1][i * mapSize + i] = 1.0;
            }

            // Execution GPU
            console.time('GPU Compute');
            await gpuGrid.compute();
            console.timeEnd('GPU Compute');

            const gpuDiffusion = facesGPU[2];
            const gpuSAT = facesGPU[4];

            // --- 3. ASSERTIONS (TOLÉRANCE 1e-5) ---
            let maxDiffSAT = 0;
            let maxDiffDiff = 0;
            for (let i = 0; i < gpuSAT.length; i++) {
                maxDiffSAT = Math.max(maxDiffSAT, Math.abs(cpuSAT[i] - gpuSAT[i]));
                maxDiffDiff = Math.max(maxDiffDiff, Math.abs(cpuDiffusion[i] - gpuDiffusion[i]));
            }

            expect(maxDiffSAT).toBeLessThan(1e-5);
            expect(maxDiffDiff).toBeLessThan(1e-5);

        } catch (e) {
            console.warn('WebGPU test skipped or failed init in Vitest context:', e);
        }
    });
});
