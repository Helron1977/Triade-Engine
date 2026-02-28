import { TriadeMasterBuffer } from './TriadeMasterBuffer';
import { TriadeCubeV2 } from './TriadeCubeV2';
import type { ITriadeEngine } from '../engines/ITriadeEngine';

/**
 * TriadeGrid gère un assemblage N x M de TriadeCubes adjacents.
 * Il assure la communication "Boundary Exchange" (Ghost Cells) entre les cubes
 * à la fin de chaque étape de calcul pour unifier la simulation.
 */
export class TriadeGrid {
    public cubes: (TriadeCubeV2 | null)[][] = [];
    public readonly cols: number;
    public readonly rows: number;
    public readonly cubeSize: number;
    public isPeriodic: boolean;
    public readonly mode: 'cpu' | 'webgpu';

    private _engineFactory: () => ITriadeEngine;

    // TODO(WebGPU V3): Pipeline layout and buffers references

    constructor(
        cols: number,
        rows: number,
        cubeSize: number,
        masterBuffer: TriadeMasterBuffer,
        engineFactory: () => ITriadeEngine,
        numFaces: number = 6,
        isPeriodic: boolean = true,
        mode: 'cpu' | 'webgpu' = 'cpu'
    ) {
        this.cols = cols;
        this.rows = rows;
        this.cubeSize = cubeSize;
        this.isPeriodic = isPeriodic;
        this.mode = mode;
        this._engineFactory = engineFactory;

        // Allocation de la grille de cubes
        for (let y = 0; y < rows; y++) {
            this.cubes[y] = [];
            for (let x = 0; x < cols; x++) {
                const cube = new TriadeCubeV2(cubeSize, masterBuffer, numFaces);
                cube.setEngine(engineFactory());
                this.cubes[y][x] = cube;
            }
        }
    }

    /**
     * Initialise asynchroniquement une grille. Obligatoire si le mode WebGPU est sélectionné
     * afin de préparer les Storage Buffers et de compiler le WGSL via TriadeGPUContext.
     */
    static async create(
        cols: number,
        rows: number,
        cubeSize: number,
        masterBuffer: TriadeMasterBuffer,
        engineFactory: () => ITriadeEngine,
        numFaces: number = 6,
        isPeriodic: boolean = true,
        mode: 'cpu' | 'webgpu' = 'cpu'
    ): Promise<TriadeGrid> {
        if (mode === 'webgpu') {
            // Check runtime WebGPU definition in case the user forgets to import Context
            const TriadeGPUContext = (await import('./gpu/TriadeGPUContext')).TriadeGPUContext;
            const success = await TriadeGPUContext.init();
            if (!success) {
                console.warn("[TriadeGrid] WebGPU init n'a pas réussi. Fallback implicite vers le mode 'cpu'.");
                mode = 'cpu';
            } else {
                console.info("[TriadeGrid] Initialisation asynchrone du contexte WebGPU : Succès.");
            }
        }

        const grid = new TriadeGrid(cols, rows, cubeSize, masterBuffer, engineFactory, numFaces, isPeriodic, mode);

        // Initialiser la VRAM de tous les cubes
        if (mode === 'webgpu') {
            for (let y = 0; y < rows; y++) {
                for (let x = 0; x < cols; x++) {
                    grid.cubes[y][x]?.initGPU();
                }
            }
        }

        return grid;
    }

    /**
     * Calcule une étape complète de la grille.
     * 1. Exécute "compute()" sur chaque cube (CPU) ou déclenche le Compute Shader (GPU)
     * 2. Synchronise les bords (Boundary Exchange) sur les faces demandées
     */
    async compute(facesToSynchronize: number | number[] = 0) {
        if (this.mode === 'webgpu') {
            const TriadeGPUContext = (await import('./gpu/TriadeGPUContext')).TriadeGPUContext;
            const commandEncoder = TriadeGPUContext.device.createCommandEncoder();

            for (let y = 0; y < this.rows; y++) {
                for (let x = 0; x < this.cols; x++) {
                    const cube = this.cubes[y][x];
                    if (cube && cube.engine && cube.engine.computeGPU) {
                        const passEncoder = commandEncoder.beginComputePass();
                        cube.engine.computeGPU(passEncoder, cube.mapSize);
                        passEncoder.end();
                    }
                }
            }

            TriadeGPUContext.device.queue.submit([commandEncoder.finish()]);
            return;
        }

        // 1. Calcul (Intra-Cube) - Mode CPU
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                this.cubes[y][x]?.compute();
            }
        }

        // 2. Synchronisation des bords O(1) Data Copy
        const faces = Array.isArray(facesToSynchronize) ? facesToSynchronize : [facesToSynchronize];
        for (const f of faces) {
            this.synchronizeBoundaries(f);
        }
    }

    /**
     * Recopie les vecteurs périphériques (1 pixel de profondeur) vers les bords des voisins.
     */
    private synchronizeBoundaries(f: number) {
        const s = this.cubeSize;
        const s_minus_1 = s - 1;
        const s_minus_2 = s - 2;

        // PASS 1: X-axis (Left/Right)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                const data = cube.faces[f];

                // Right neighbor
                if (x < this.cols - 1 || this.isPeriodic) {
                    const rightCube = this.cubes[y][(x + 1) % this.cols]!;
                    const rightData = rightCube.faces[f];
                    for (let row = 1; row < s_minus_1; row++) {
                        rightData[row * s + 0] = data[row * s + s_minus_2];
                    }
                }

                // Left neighbor
                if (x > 0 || this.isPeriodic) {
                    const leftCube = this.cubes[y][(x - 1 + this.cols) % this.cols]!;
                    const leftData = leftCube.faces[f];
                    for (let row = 1; row < s_minus_1; row++) {
                        leftData[row * s + s_minus_1] = data[row * s + 1];
                    }
                }
            }
        }

        // PASS 2: Y-axis (Top/Bottom) - Copying full rows which perfectly transfers the corners (ghost columns from PASS 1) 
        // to diagonal neighbors automatically!
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                const data = cube.faces[f];

                // Bottom neighbor
                if (y < this.rows - 1 || this.isPeriodic) {
                    const bottomCube = this.cubes[(y + 1) % this.rows][x]!;
                    const bottomData = bottomCube.faces[f];
                    bottomData.set(data.subarray(s_minus_2 * s, s_minus_2 * s + s), 0);
                }

                // Top neighbor
                if (y > 0 || this.isPeriodic) {
                    const topCube = this.cubes[(y - 1 + this.rows) % this.rows][x]!;
                    const topData = topCube.faces[f];
                    topData.set(data.subarray(1 * s, 1 * s + s), s_minus_1 * s);
                }
            }
        }
    }
}
