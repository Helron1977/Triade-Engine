import type { HypercubeChunk } from '../HypercubeChunk';
import type { IHypercubeEngine } from '../../engines/IHypercubeEngine';

/**
 * Interface pour les messages échangés entre le Main Thread et les Workers
 */
export interface HypercubeWorkerMessage {
    type: 'INIT' | 'COMPUTE' | 'DONE';
    cubeOffset?: number;
    stride?: number;
    numFaces?: number;
    mapSize?: number;
    engineName?: string;
    engineConfig?: any;
    sharedBuffer?: SharedArrayBuffer;
    facesToSync?: number[];
}

/**
 * HypercubeWorkerPool
 * Gère une file de Web Workers (N threads) pour dispatcher le calcul des cubes "O(1)".
 * Ne fonctionne que si la mémoire est un SharedArrayBuffer.
 */
export class HypercubeWorkerPool {
    private workers: Worker[] = [];
    private maxThreads: number;

    constructor(maxThreads?: number) {
        // Obtenir le nombre de coeurs logiques (par défaut 4 si indisponible)
        this.maxThreads = maxThreads ?? (typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4);
    }

    /**
     * Initialise la pool en levant N instances du worker script.
     * @param workerScriptPath Le chemin relatif vers le script worker pré-compilé
     */
    async init(workerScriptPath: string = './cpu.worker.js'): Promise<void> {
        return new Promise((resolve) => {
            let initialized = 0;
            const onWorkerInit = () => {
                initialized++;
                if (initialized === this.maxThreads) resolve();
            };

            for (let i = 0; i < this.maxThreads; i++) {
                const worker = new Worker(workerScriptPath, { type: 'module' });
                worker.onmessage = (e: MessageEvent<HypercubeWorkerMessage>) => {
                    if (e.data.type === 'DONE') {
                        // Traité plus loin dans le dispatcher
                    }
                };
                this.workers.push(worker);
                onWorkerInit(); // TODO: Attendre confirmation asynchrone du worker si nécessaire
            }
        });
    }

    /**
     * Dispatch dynamiquement un ensemble de cubes sur les différents threads disponibles.
     * Attend la résolution de tous les workers.
     */
    async computeAll(
        cubesToCompute: HypercubeChunk[],
        sharedBuffer: SharedArrayBuffer,
        engineParams: { name: string, config: any }
    ): Promise<void> {

        if (this.workers.length === 0) {
            console.warn("[HypercubeWorkerPool] Pool vide, fallback sur l'éxecution séquentielle (Main Thread).");
            for (const cube of cubesToCompute) {
                cube.compute();
            }
            return;
        }

        return new Promise((resolve) => {
            let completedCubes = 0;
            const totalCubes = cubesToCompute.length;

            if (totalCubes === 0) return resolve();

            let nextCubeIndex = 0;
            let activeWorkers = 0;

            const assignNextTask = (workerId: number) => {
                const worker = this.workers[workerId];

                // Si toutes les tâches sont finies
                if (completedCubes === totalCubes) {
                    if (activeWorkers === 0) resolve();
                    return;
                }

                // S'il reste des cubes à traiter, assigner au worker
                if (nextCubeIndex < totalCubes) {
                    const cube = cubesToCompute[nextCubeIndex];
                    nextCubeIndex++;
                    activeWorkers++;

                    worker.onmessage = (e: MessageEvent<HypercubeWorkerMessage>) => {
                        if (e.data.type === 'DONE') {
                            completedCubes++;
                            activeWorkers--;
                            assignNextTask(workerId); // Round-Robin
                        }
                    };

                    worker.postMessage({
                        type: 'COMPUTE',
                        cubeOffset: cube.offset,
                        stride: (cube as any).stride || cube.mapSize * cube.mapSize * 4, // Fallback safe
                        numFaces: cube.faces.length,
                        mapSize: cube.mapSize,
                        engineName: engineParams.name,
                        engineConfig: engineParams.config,
                        sharedBuffer: sharedBuffer
                    } as HypercubeWorkerMessage);
                }
            };

            // Démarrage initial
            for (let i = 0; i < this.maxThreads && i < totalCubes; i++) {
                assignNextTask(i);
            }
        });
    }
}




































