import { expect, test, describe, beforeAll } from 'vitest';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { VolumeDiffusionEngine } from '../src/engines/VolumeDiffusionEngine';
import { HypercubeGPUContext } from '../src/core/gpu/HypercubeGPUContext';

/**
 * Ce test vérifie que le moteur WebGPU ne lance pas d'erreurs de validation
 * lors de l'initialisation et du compute.
 * Note: Ce test nécessite un environnement avec support WebGPU (ex: Chrome avec --enable-unsafe-webgpu).
 */
describe('WebGPU Engine Validation', () => {
    let masterBuffer: HypercubeMasterBuffer;

    beforeAll(async () => {
        masterBuffer = new HypercubeMasterBuffer(1024 * 1024 * 10); // 10MB
        if (HypercubeGPUContext.isSupported) {
            await HypercubeGPUContext.init();
        }
    });

    test('VolumeDiffusionEngine GPU Initialization & Validation', async () => {
        if (!HypercubeGPUContext.isSupported) {
            console.warn('WebGPU not supported in this environment, skipping test.');
            return;
        }

        const size = 16;
        const grid = await HypercubeGrid.create(
            1, 1, size,
            masterBuffer,
            () => new VolumeDiffusionEngine(0.1, 1.0),
            2,
            false,
            'gpu'
        );

        expect(grid.isGpuReady).toBe(true);

        // Cette étape génère les commandes GPU. 
        // Si il y a une erreur de BindGroup/Pipeline layout, elle sera visible ici 
        // ou lors de la soumission.
        try {
            await grid.compute();
            expect(true).toBe(true);
        } catch (e) {
            console.error('GPU Compute Error:', e);
            throw e;
        }
    });
});
