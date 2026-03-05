import { HypercubeMasterBuffer } from './HypercubeMasterBuffer';
import { HypercubeChunk } from './HypercubeChunk';
import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

/**
 * HypercubeGpuGrid - Version V5.4
 * Gestionnaire de grille GPU avec vrai double-buffering + boundary exchange en VRAM.
 */
export class HypercubeGpuGrid {
    public cubes: (HypercubeChunk | null)[][] = [];
    public readonly cols: number;
    public readonly rows: number;
    public readonly nx: number;
    public readonly ny: number;
    public readonly nz: number;
    public masterBuffer: HypercubeMasterBuffer;

    public stats = {
        computeTimeMs: 0,
        syncTimeMs: 0
    };

    public isPeriodic: boolean = true;

    constructor(
        cols: number,
        rows: number,
        resolution: number | { nx: number, ny: number, nz?: number },
        masterBuffer: HypercubeMasterBuffer,
        engineFactory: () => IHypercubeEngine,
        numFaces: number = 6
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

        const tempEngine = engineFactory();
        const requiredFaces = tempEngine.getRequiredFaces();
        const finalNumFaces = Math.max(numFaces, requiredFaces);

        for (let y = 0; y < rows; y++) {
            this.cubes[y] = [];
            for (let x = 0; x < cols; x++) {
                const cube = new HypercubeChunk(x, y, this.nx, this.ny, this.nz, masterBuffer, finalNumFaces);
                const engineInstance = (y === 0 && x === 0) ? tempEngine : engineFactory();
                cube.setEngine(engineInstance);
                this.cubes[y][x] = cube;
            }
        }
    }

    static async create(
        cols: number,
        rows: number,
        resolution: number | { nx: number, ny: number, nz?: number },
        masterBuffer: HypercubeMasterBuffer,
        engineFactory: () => IHypercubeEngine,
        numFaces: number = 6
    ): Promise<HypercubeGpuGrid> {

        const success = await HypercubeGPUContext.init();
        if (!success) throw new Error("WebGPU initialization failed");

        const grid = new HypercubeGpuGrid(cols, rows, resolution, masterBuffer, engineFactory, numFaces);

        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {
                grid.cubes[y][x]?.initGPU();
            }
        }

        return grid;
    }

    // ── GPU REFACTO V5.4 ── Zero-Readback Compute Pipeline
    async compute() {
        const start = performance.now();
        const device = HypercubeGPUContext.device;
        const encoder = device.createCommandEncoder({ label: 'Hypercube Full GPU Pass' });

        // 1. Compute local sur tous les chunks
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x];
                if (cube?.engine?.computeGPU) {
                    cube.engine.computeGPU(
                        device,
                        encoder,
                        cube.nx,
                        cube.ny,
                        cube.nz,
                        cube.gpuReadBuffer!,
                        cube.gpuWriteBuffer!
                    );
                }
            }
        }

        // 2. Synchronisation des frontières directement en VRAM
        if (this.cols > 1 || this.rows > 1) {
            this.synchronizeBoundariesGPU(encoder);
        }

        // 3. Soumission de tout le travail GPU en une seule queue
        device.queue.submit([encoder.finish()]);

        // 4. Swap des buffers (très peu coûteux)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x];
                if (cube) {
                    cube.swapGPUBuffers();
                    // On synchronise la parité logique du moteur (CRITIQUE pour OceanEngine V5.4)
                    if (cube.engine && 'parity' in cube.engine) {
                        (cube.engine as any).parity = cube.gpuParity;
                    }
                }
            }
        }

        this.stats.computeTimeMs = performance.now() - start;
    }

    // ── GPU REFACTO V5.4 ── Boundary Exchange 100% GPU
    private synchronizeBoundariesGPU(encoder: GPUCommandEncoder) {
        const facesToSync = this.cubes[0][0]?.engine?.getSyncFaces?.() ?? [0];

        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                if (!cube.gpuWriteBuffer) continue;

                const fOffset = (f: number) => f * cube.stride;

                // Right neighbor
                if (this.isPeriodic || x < this.cols - 1) {
                    const right = this.cubes[y][(x + 1) % this.cols]!;
                    for (const f of facesToSync) {
                        HypercubeGPUContext.gpuCopyBoundary(
                            encoder,
                            cube.gpuWriteBuffer,
                            right.gpuWriteBuffer!,
                            fOffset(f) + (cube.nx - 2) * 4,   // dernier float valide
                            fOffset(f) + 0,                    // premier float du voisin
                            4,                                 // 1 float
                            cube.ny - 2,                       // nombre de lignes
                            cube.nx * 4,                       // stride source
                            cube.nx * 4                        // stride dest
                        );
                    }
                }

                // Bottom neighbor (similaire)
                if (this.isPeriodic || y < this.rows - 1) {
                    const bottom = this.cubes[(y + 1) % this.rows][x]!;
                    for (const f of facesToSync) {
                        HypercubeGPUContext.gpuCopyBoundary(
                            encoder,
                            cube.gpuWriteBuffer,
                            bottom.gpuWriteBuffer!,
                            fOffset(f) + (cube.ny - 2) * cube.nx * 4,
                            fOffset(f) + 0,
                            cube.nx * 4,
                            1,
                            0,
                            0
                        );
                    }
                }
            }
        }
    }

    public syncAllToHost(faceIndices?: number[]) {
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                this.cubes[y][x]?.syncToHost(faceIndices, false);
            }
        }
    }

    public destroy() {
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                this.cubes[y][x]?.destroy();
            }
        }
        HypercubeGPUContext.destroy();
    }
}