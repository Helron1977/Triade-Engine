import { HypercubeMasterBuffer } from './HypercubeMasterBuffer';
import { HypercubeChunk } from './HypercubeChunk';
import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HypercubeWorkerPool } from './cpu/HypercubeWorkerPool';

/**
 * HypercubeGrid gère un assemblage N x M de TriadeCubes adjacents.
 * Il assure la communication "Boundary Exchange" (Ghost Cells) entre les cubes
 * à la fin de chaque étape de calcul pour unifier la simulation.
 */
export class HypercubeGrid {
    public cubes: (HypercubeChunk | null)[][] = [];
    public readonly cols: number;
    public readonly rows: number;
    public readonly nx: number;
    public readonly ny: number;
    public readonly nz: number;
    public isPeriodic: boolean;
    public readonly mode: 'cpu' | 'webgpu';
    public masterBuffer: HypercubeMasterBuffer;

    public stats = {
        computeTimeMs: 0,
        syncTimeMs: 0
    };

    private _engineFactory: () => IHypercubeEngine;
    private workerPool: HypercubeWorkerPool | null = null;

    constructor(
        cols: number,
        rows: number,
        resolution: number | { nx: number, ny: number, nz?: number },
        masterBuffer: HypercubeMasterBuffer,
        engineFactory: () => IHypercubeEngine,
        numFaces: number = 6,
        isPeriodic: boolean = true,
        mode: 'cpu' | 'webgpu' = 'cpu'
    ) {
        this.cols = cols;
        this.rows = rows;

        if (typeof resolution === 'number') {
            this.nx = resolution;
            this.ny = resolution;
            this.nz = 1;
        } else {
            this.nx = resolution.nx;
            this.ny = resolution.ny;
            this.nz = resolution.nz ?? 1;
        }

        this.masterBuffer = masterBuffer;
        this.isPeriodic = isPeriodic;
        this.mode = mode;
        this._engineFactory = engineFactory;

        // Détection automatique du nombre de faces via l'engine (Step 10 Roadmap)
        const tempEngine = engineFactory();
        const requiredFaces = tempEngine.getRequiredFaces();
        const finalNumFaces = Math.max(numFaces, requiredFaces);

        // Allocation de la grille de cubes
        for (let y = 0; y < rows; y++) {
            this.cubes[y] = [];
            for (let x = 0; x < cols; x++) {
                const cube = new HypercubeChunk(x, y, this.nx, this.ny, this.nz, masterBuffer, finalNumFaces);
                const engineInstance = y === 0 && x === 0 ? tempEngine : engineFactory();
                cube.setEngine(engineInstance);
                engineInstance.init(cube.faces, cube.nx, cube.ny, cube.nz, false); // isWorker = false
                this.cubes[y][x] = cube;
            }
        }
    }

    /**
     * Initialise asynchroniquement une grille. Obligatoire si le mode WebGPU est sélectionné
     * afin de préparer les Storage Buffers et de compiler le WGSL via HypercubeGPUContext.
     */
    static async create(
        cols: number,
        rows: number,
        resolution: number | { nx: number, ny: number, nz?: number },
        masterBuffer: HypercubeMasterBuffer,
        engineFactory: () => IHypercubeEngine,
        numFaces: number = 6,
        isPeriodic: boolean = true,
        mode: 'cpu' | 'webgpu' = 'cpu',
        useWorkers: boolean = true
    ): Promise<HypercubeGrid> {
        if (mode === 'webgpu') {
            // Check runtime WebGPU definition in case the user forgets to import Context
            const HypercubeGPUContext = (await import('./gpu/HypercubeGPUContext')).HypercubeGPUContext;
            const success = await HypercubeGPUContext.init();
            if (!success) {
                console.warn("[HypercubeGrid] WebGPU init n'a pas réussi. Fallback implicite vers le mode 'cpu'.");
                mode = 'cpu';
            } else {
                console.info("[HypercubeGrid] Initialisation asynchrone du contexte WebGPU : Succès.");
            }
        }

        const grid = new HypercubeGrid(cols, rows, resolution, masterBuffer, engineFactory, numFaces, isPeriodic, mode);

        // Initialiser la VRAM de tous les cubes
        if (mode === 'webgpu') {
            for (let y = 0; y < rows; y++) {
                for (let x = 0; x < cols; x++) {
                    grid.cubes[y][x]?.initGPU();
                }
            }
        } else if (mode === 'cpu' && useWorkers && typeof SharedArrayBuffer !== 'undefined' && masterBuffer.buffer instanceof SharedArrayBuffer) {
            // Activer l'accélération multicore logicielle
            grid.workerPool = new HypercubeWorkerPool();
            try {
                // Tenter de charger le worker via Vite/Tsup si la compilation exporte le Worker externe
                // Note : il faudra configurer Tsup pour compiler cpu.worker.ts séparément
                await grid.workerPool.init();
                console.info(`[HypercubeGrid] WorkerPool CPU instanciée avec succès (Zero-Copy prêt).`);
            } catch (error) {
                console.warn("[HypercubeGrid] Échec de l'initialisation de la WorkerPool.", error);
                grid.workerPool = null;
            }
        }

        return grid;
    }

    /**
     * Calcule une étape complète de la grille.
     * 1. Exécute "compute()" sur chaque cube (CPU ou WorkerPool) ou déclenche le Compute Shader (GPU)
     * 2. Synchronise les bords (Boundary Exchange) sur les faces demandées ou déduites par le moteur.
     */
    async compute(facesToSynchronize?: number | number[]) {
        if (this.mode === 'webgpu') {
            const HypercubeGPUContext = (await import('./gpu/HypercubeGPUContext')).HypercubeGPUContext;
            const commandEncoder = HypercubeGPUContext.device.createCommandEncoder();

            for (let y = 0; y < this.rows; y++) {
                for (let x = 0; x < this.cols; x++) {
                    const cube = this.cubes[y][x];
                    if (cube && cube.engine && cube.engine.computeGPU) {
                        cube.engine.computeGPU(HypercubeGPUContext.device, commandEncoder, cube.nx, cube.ny, cube.nz);
                    }
                }
            }

            HypercubeGPUContext.device.queue.submit([commandEncoder.finish()]);
            // On pourras remonter les faces en RAM ici avec un async map si l'engine doit afficher
            return;
        }

        // 1. Calcul (Intra-Cube) - Mode CPU Multithread (Workers + SharedMemory)
        const computeStart = performance.now();

        if (this.workerPool && this.masterBuffer.buffer instanceof SharedArrayBuffer) {
            const flatCubes = this.cubes.flat().filter(c => c !== null) as HypercubeChunk[];

            // Note: On assume que l'engine du premier cube est représentatif
            const engineName = flatCubes[0]?.engine?.name || 'Unknown';
            const engineConfigs = flatCubes.map(c => {
                const ce = c.engine;
                if (ce && ce.getConfig) return ce.getConfig();

                // Fallback générique retro-compatible si getConfig n'est pas implémenté
                return {
                    radius: (ce as any)?.radius || 10,
                    weight: (ce as any)?.weight ?? 0.0,
                    targetX: (ce as any)?.targetX ?? 256,
                    targetY: (ce as any)?.targetY ?? 256,
                    isLeftBoundary: (ce as any)?.isLeftBoundary ?? (c.x === 0),
                    isRightBoundary: (ce as any)?.isRightBoundary ?? (c.x === this.cols - 1),
                    ...(ce as any)?.params
                };
            });

            await this.workerPool.computeAll(flatCubes, this.masterBuffer.buffer, { name: engineName, config: engineConfigs });
        }
        // 1bis. Calcul (Intra-Cube) - Mode CPU Séquentiel (Main Thread)
        else {
            for (let y = 0; y < this.rows; y++) {
                for (let x = 0; x < this.cols; x++) {
                    await this.cubes[y][x]?.compute();
                }
            }
        }

        const computeEnd = performance.now();
        this.stats.computeTimeMs = computeEnd - computeStart;

        // 2. Synchronisation des bords O(1) Data Copy
        // S'il n'y a qu'un seul cube, la synchronisation est inutile (et risquerait d'écraser des données sur lui-même)
        if (this.cols === 1 && this.rows === 1) return;

        const syncStart = performance.now();

        let faces: number[];
        if (facesToSynchronize !== undefined) {
            faces = Array.isArray(facesToSynchronize) ? facesToSynchronize : [facesToSynchronize];
        } else {
            // Déduction automatique depuis le moteur du premier cube de la grille
            const engine = this.cubes[0][0]?.engine;
            faces = (engine && engine.getSyncFaces) ? engine.getSyncFaces() : [0];
        }

        for (const f of faces) {
            this.synchronizeBoundaries(f);
        }

        const syncEnd = performance.now();
        this.stats.syncTimeMs = syncEnd - syncStart;
    }

    /**
     * Recopie les vecteurs périphériques (1 pixel de profondeur) vers les bords des voisins.
     */
    private synchronizeBoundaries(f: number) {
        const nx = this.nx;
        const ny = this.ny;
        const nz = this.nz;

        // PASS 1: X-axis (Left/Right exchange of YZ planes)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                const data = cube.faces[f];

                // Right neighbor
                if (x < this.cols - 1 || this.isPeriodic) {
                    const rightCube = this.cubes[y][(x + 1) % this.cols]!;
                    const rightData = rightCube.faces[f];
                    for (let lz = 0; lz < nz; lz++) {
                        for (let ly = 1; ly < ny - 1; ly++) {
                            rightData[rightCube.getIndex(0, ly, lz)] = data[cube.getIndex(nx - 2, ly, lz)];
                        }
                    }
                }

                // Left neighbor
                if (x > 0 || this.isPeriodic) {
                    const leftCube = this.cubes[y][(x - 1 + this.cols) % this.cols]!;
                    const leftData = leftCube.faces[f];
                    for (let lz = 0; lz < nz; lz++) {
                        for (let ly = 1; ly < ny - 1; ly++) {
                            leftData[leftCube.getIndex(nx - 1, ly, lz)] = data[cube.getIndex(1, ly, lz)];
                        }
                    }
                }
            }
        }

        // PASS 2: Y-axis (Top/Bottom exchange of XZ planes)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                const data = cube.faces[f];

                // Bottom neighbor
                if (y < this.rows - 1 || this.isPeriodic) {
                    const botCube = this.cubes[(y + 1) % this.rows][x]!;
                    const botData = botCube.faces[f];
                    for (let lz = 0; lz < nz; lz++) {
                        const srcOffset = cube.getIndex(1, ny - 2, lz);
                        const dstOffset = botCube.getIndex(1, 0, lz);
                        botData.set(data.subarray(srcOffset, srcOffset + nx - 2), dstOffset);
                    }
                }

                // Top neighbor
                if (y > 0 || this.isPeriodic) {
                    const topRow = (y === 0) ? this.rows - 1 : y - 1;
                    const topCube = this.cubes[topRow][x]!;
                    const topData = topCube.faces[f];
                    for (let lz = 0; lz < nz; lz++) {
                        const srcOffset = cube.getIndex(1, 1, lz);
                        const dstOffset = topCube.getIndex(1, ny - 1, lz);
                        topData.set(data.subarray(srcOffset, srcOffset + nx - 2), dstOffset);
                    }
                }
            }
        }

        // PASS 3: Corners (Explicit Diagonal Sync for 3D stacks)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                const data = cube.faces[f];

                const nxP = (x + 1) % this.cols;
                const nxM = (x - 1 + this.cols) % this.cols;
                const nyP = (y + 1) % this.rows;
                const nyM = (y - 1 + this.rows) % this.rows;

                for (let lz = 0; lz < nz; lz++) {
                    // 1. Bottom-Right Neighbor
                    if (this.isPeriodic || (x < this.cols - 1 && y < this.rows - 1)) {
                        this.cubes[nyP][nxP]!.faces[f][this.cubes[nyP][nxP]!.getIndex(0, 0, lz)] = data[cube.getIndex(nx - 2, ny - 2, lz)];
                    }
                    // 2. Bottom-Left Neighbor
                    if (this.isPeriodic || (x > 0 && y < this.rows - 1)) {
                        this.cubes[nyP][nxM]!.faces[f][this.cubes[nyP][nxM]!.getIndex(nx - 1, 0, lz)] = data[cube.getIndex(1, ny - 2, lz)];
                    }
                    // 3. Top-Right Neighbor
                    if (this.isPeriodic || (x < this.cols - 1 && y > 0)) {
                        this.cubes[nyM][nxP]!.faces[f][this.cubes[nyM][nxP]!.getIndex(0, ny - 1, lz)] = data[cube.getIndex(nx - 2, 1, lz)];
                    }
                    // 4. Top-Left Neighbor
                    if (this.isPeriodic || (x > 0 && y > 0)) {
                        this.cubes[nyM][nxM]!.faces[f][this.cubes[nyM][nxM]!.getIndex(nx - 1, ny - 1, lz)] = data[cube.getIndex(1, 1, lz)];
                    }
                }
            }
        }
    }

    /**
     * Libère les ressources asynchrones (ex: Web Workers) associées à la grille.
     */
    public destroy() {
        if (this.workerPool) {
            this.workerPool.terminate();
            this.workerPool = null;
        }
    }
}




































