import type { HypercubeMasterBuffer } from '../HypercubeMasterBuffer';
import { HypercubeChunk } from '../HypercubeChunk';
import type { IHypercubeEngine } from '../../engines/IHypercubeEngine';
// Import statique des moteurs connus (A améliorer via un Registry dynamique plus tard)
import { HeatmapEngine } from '../../engines/HeatmapEngine';
import { FlowFieldEngine } from '../../engines/FlowFieldEngine';
import { FluidEngine } from '../../engines/FluidEngine';
import { AerodynamicsEngine } from '../../engines/AerodynamicsEngine';
import { OceanEngine } from '../../engines/OceanEngine';

/**
 * Script de base exécuté par les instances Web Worker de la HypercubeWorkerPool.
 * N'a pas de DOM, uniquement CPU/Math.
 */

// Simulation d'un faux Master Buffer pour passer la vérification du constructeur HypercubeChunk
class WorkerMasterBufferDummy {
    public buffer: SharedArrayBuffer;
    private _offset: number = 0;
    private _stride: number = 0;

    constructor(sharedBuf: SharedArrayBuffer) {
        this.buffer = sharedBuf;
    }

    // Ne fait rien, car le cube est déjà alloué par le Main Thread
    allocateCube(nx: number, ny: number, nz: number = 1, numFaces: number = 6): { offset: number, stride: number } {
        return { offset: this._offset, stride: this._stride };
    }

    setMockLocation(offset: number, stride: number) {
        this._offset = offset;
        this._stride = stride;
    }
}

self.onmessage = (e: MessageEvent) => {
    const data = e.data;

    if (data.type === 'COMPUTE') {
        const { engineName, engineConfig, sharedBuffer, cubeOffset, stride, numFaces, nx, ny, nz, chunkX, chunkY } = data;

        if (!sharedBuffer) {
            console.error("[Worker] Pas de SharedArrayBuffer reçu.");
            postMessage({ type: 'DONE', success: false });
            return;
        }

        // 1. Recréer l'Engine depuis son nom et sa config
        let engine: IHypercubeEngine | null = null;
        if (engineName === 'Heatmap (O1 Spatial Convolution)') {
            engine = new HeatmapEngine(engineConfig?.radius, engineConfig?.weight);
        } else if (engineName === 'FlowFieldEngine-V12') {
            engine = new FlowFieldEngine();
            if (engineConfig && 'targetX' in engineConfig) {
                (engine as any).targetX = engineConfig.targetX;
                (engine as any).targetY = engineConfig.targetY;
            }
        } else if (engineName === 'Simplified Fluid Dynamics') {
            engine = new FluidEngine(engineConfig?.dt, engineConfig?.buoyancy, engineConfig?.dissipation);
        } else if (engineName === 'Lattice Boltzmann D2Q9 (O(1))') {
            // Configuration AerodynamicsEngine (Step 13)
            engine = new AerodynamicsEngine();
        } else if (engineName === 'OceanEngine') {
            engine = new OceanEngine();
            if (engineConfig && Object.keys(engineConfig).length > 0) {
                (engine as any).params = engineConfig;
            }
        } else {
            // Fallback temporaire pour les autres moteurs non gérés dynamiquement ici
            console.error(`[Worker] Moteur non reconnu ou non supporté par les Web Workers: ${engineName}`);
            postMessage({ type: 'DONE', success: false });
            return;
        }

        if (!engine) {
            console.error(`[Worker CPU] Erreur fatale: engine est null.`);
            postMessage({ type: 'DONE', success: false });
            return;
        }

        // 2. Mock du Master Buffer pour Mapper la VUE (les Float32Array)
        const dummyBuffer = new WorkerMasterBufferDummy(sharedBuffer);
        dummyBuffer.setMockLocation(cubeOffset, stride);

        // 3. Reconstruire le Cube (Zéro-Copie des données, on ne fait que recréer l'objet JS conteneur)
        // Attention: HypercubeChunk va instancier ses Float32Array par dessus l'offset passé
        const cube = new HypercubeChunk(chunkX || 0, chunkY || 0, nx, ny, nz || 1, dummyBuffer as unknown as any, numFaces || 6);
        cube.setEngine(engine);

        // 4. Calcul Lourd O(N) -> O(1)
        try {
            // Note: cube.compute() is async in HypercubeChunk! We must await it!
            Promise.resolve(cube.compute()).then(() => {
                // 5. Libération et notification Main Thread
                postMessage({ type: 'DONE', success: true });
            }).catch((err) => {
                console.error(`[Worker CPU] Crash asynchrone pendant l'exécution du moteur ${engineName}:`, err);
                postMessage({ type: 'DONE', success: false, error: err?.message });
            });
        } catch (error: any) {
            console.error(`[Worker CPU] Crash synchrone pendant l'exécution du moteur ${engineName}:`, error);
            postMessage({ type: 'DONE', success: false, error: error?.message });
        }
    }
};




































