import type { HypercubeMasterBuffer } from '../HypercubeMasterBuffer';
import { HypercubeChunk } from '../HypercubeChunk';
import type { IHypercubeEngine } from '../../engines/IHypercubeEngine';
import { EngineRegistry } from '../EngineRegistry';

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

// Cache pour conserver l'état des moteurs (ex: this.initialized) entre chaque frame
const chunkCache = new Map<number, HypercubeChunk>();

self.onmessage = (e: MessageEvent) => {
    const data = e.data;

    if (data.type === 'COMPUTE') {
        const { engineName, engineConfig, sharedBuffer, cubeOffset, stride, numFaces, nx, ny, nz, chunkX, chunkY } = data;

        if (!sharedBuffer) {
            console.error("[Worker] Pas de SharedArrayBuffer reçu.");
            postMessage({ type: 'DONE', success: false });
            return;
        }

        // --- Validate Header ---
        const header = new Uint32Array(sharedBuffer, 0, 2);
        if (header[0] !== 0x48595045 || header[1] !== 4) {
            console.error("[Worker CPU] Buffer invalide ou version mismatch. Format attendu: HYPE v4.");
            postMessage({ type: 'DONE', success: false });
            return;
        }

        let cube = chunkCache.get(cubeOffset);

        if (!cube) {
            // 1. Instancier l'Engine dynamique via le Registry
            let engine: IHypercubeEngine;
            try {
                engine = EngineRegistry.create(engineName, engineConfig);
            } catch (err: any) {
                console.error(err.message);
                postMessage({ type: 'DONE', success: false, error: err.message });
                return;
            }

            // 2. Mock du Master Buffer pour Mapper la VUE
            const dummyBuffer = new WorkerMasterBufferDummy(sharedBuffer);
            dummyBuffer.setMockLocation(cubeOffset, stride);

            // 3. Reconstruire le Cube (Zéro-Copie des données)
            cube = new HypercubeChunk(chunkX || 0, chunkY || 0, nx, ny, nz || 1, dummyBuffer as unknown as any, numFaces || 6);
            cube.setEngine(engine);
            engine.init(cube.faces, cube.nx, cube.ny, cube.nz, true); // isWorker = true

            chunkCache.set(cubeOffset, cube);
        } else {
            // 4. Mettre à jour la configuration dynamique si elle a changé
            if (cube.engine && engineConfig) {
                EngineRegistry.applyConfig(cube.engine, engineConfig);
            }
        }

        // 5. Calcul Lourd O(N) -> O(1)
        try {
            Promise.resolve(cube.compute()).then(() => {
                // 6. Libération et notification Main Thread
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




































