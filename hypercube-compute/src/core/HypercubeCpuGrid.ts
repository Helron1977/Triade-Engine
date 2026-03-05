import { HypercubeMasterBuffer } from './HypercubeMasterBuffer';
import { HypercubeChunk } from './HypercubeChunk';
import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HypercubeWorkerPool } from './cpu/HypercubeWorkerPool';
import { BoundaryConditions, BoundaryConfig, BoundaryType } from './cpu/BoundaryConditions';

/**
 * HypercubeCpuGrid manages an N x M assembly of physical chunks.
 * It is dedicated exclusively to zero-copy CPU ArrayBuffers and Multithreading.
 * It manages the Boundary Exchange (Ghost Cells) between the chunks.
 */
export class HypercubeCpuGrid {
    public cubes: (HypercubeChunk | null)[][] = [];
    public readonly cols: number;
    public readonly rows: number;
    public readonly nx: number;
    public readonly ny: number;
    public readonly nz: number;
    public isPeriodic: boolean;
    public boundaryConfig: BoundaryConfig | null = null;
    public masterBuffer: HypercubeMasterBuffer;

    public stats = {
        computeTimeMs: 0,
        syncTimeMs: 0
    };

    private _engineFactory: () => IHypercubeEngine;
    private workerPool: HypercubeWorkerPool | null = null;
    private useWorkers: boolean;

    constructor(
        cols: number,
        rows: number,
        resolution: number | { nx: number, ny: number, nz?: number },
        masterBuffer: HypercubeMasterBuffer,
        engineFactory: () => IHypercubeEngine,
        numFaces: number = 6,
        isPeriodic: boolean = true,
        useWorkers: boolean = true
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
        this._engineFactory = engineFactory;
        this.useWorkers = useWorkers;

        // Auto-detect faces
        const tempEngine = engineFactory();
        const requiredFaces = tempEngine.getRequiredFaces();
        const finalNumFaces = Math.max(numFaces, requiredFaces);

        // Allocate cubes
        for (let y = 0; y < rows; y++) {
            this.cubes[y] = [];
            for (let x = 0; x < cols; x++) {
                const cube = new HypercubeChunk(x, y, this.nx, this.ny, this.nz, masterBuffer, finalNumFaces);
                const engineInstance = y === 0 && x === 0 ? tempEngine : engineFactory();
                cube.setEngine(engineInstance);
                engineInstance.init(cube.faces, cube.nx, cube.ny, cube.nz, false);
                this.cubes[y][x] = cube;
            }
        }
    }

    /**
     * Initializes a CPU Grid, spinning up the WorkerPool if multithreading is enabled.
     */
    static async create(
        cols: number,
        rows: number,
        resolution: number | { nx: number, ny: number, nz?: number },
        masterBuffer: HypercubeMasterBuffer,
        engineFactory: () => IHypercubeEngine,
        numFaces: number = 6,
        isPeriodic: boolean = true,
        useWorkers: boolean = true,
        workerScriptPath?: string
    ): Promise<HypercubeCpuGrid> {
        const grid = new HypercubeCpuGrid(cols, rows, resolution, masterBuffer, engineFactory, numFaces, isPeriodic, useWorkers);

        if (useWorkers && typeof SharedArrayBuffer !== 'undefined' && masterBuffer.buffer instanceof SharedArrayBuffer) {
            grid.workerPool = new HypercubeWorkerPool();
            try {
                await grid.workerPool.init(masterBuffer.buffer as SharedArrayBuffer, workerScriptPath);
                console.info(`[HypercubeCpuGrid] WorkerPool initialized successfully with script: ${workerScriptPath || './cpu.worker.js'}`);
            } catch (error) {
                console.warn("[HypercubeCpuGrid] WorkerPool initialization failed.", error);
                grid.workerPool = null;
            }
        }

        return grid;
    }

    /**
     * Executes the computational pass, then synchronizes physical memory boundaries.
     */
    async compute(facesToSynchronize?: number | number[]) {
        const computeStart = performance.now();

        if (this.boundaryConfig) {
            const flatCubes = this.cubes.flat().filter(c => c !== null) as HypercubeChunk[];
            for (const cube of flatCubes) {
                if (cube.engine?.setBoundaryConfig) {
                    cube.engine.setBoundaryConfig({
                        ...this.boundaryConfig,
                        isLeftBoundary: cube.x === 0,
                        isRightBoundary: cube.x === this.cols - 1,
                        isTopBoundary: cube.y === 0,
                        isBottomBoundary: cube.y === this.rows - 1,
                    });
                }
            }
        }

        if (this.workerPool && this.masterBuffer.buffer instanceof SharedArrayBuffer) {
            const flatCubes = this.cubes.flat().filter(c => c !== null) as HypercubeChunk[];

            const engineName = flatCubes[0]?.engine?.name || 'Unknown';
            const engineConfigs = flatCubes.map(c => {
                const ce = c.engine;
                if (ce && ce.getConfig) return ce.getConfig();
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
        } else {
            for (let y = 0; y < this.rows; y++) {
                for (let x = 0; x < this.cols; x++) {
                    await this.cubes[y][x]?.compute();
                }
            }
        }

        const computeEnd = performance.now();
        this.stats.computeTimeMs = computeEnd - computeStart;

        if (this.cols === 1 && this.rows === 1) return;

        const syncStart = performance.now();

        let faces: number[];
        if (facesToSynchronize !== undefined) {
            faces = Array.isArray(facesToSynchronize) ? facesToSynchronize : [facesToSynchronize];
        } else {
            const engine = this.cubes[0][0]?.engine;
            faces = (engine && engine.getSyncFaces) ? engine.getSyncFaces() : [0];
        }

        for (const f of faces) {
            this.synchronizeBoundaries(f);
        }

        const syncEnd = performance.now();
        this.stats.syncTimeMs = syncEnd - syncStart;
    }

    private synchronizeBoundaries(f: number) {
        const nx = this.nx;
        const ny = this.ny;
        const nz = this.nz;

        // PASS 1: X-axis (Left/Right)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                const data = cube.faces[f];

                // Send Right
                if (x < this.cols - 1 || this.isPeriodic) {
                    const rightCube = this.cubes[y][(x + 1) % this.cols]!;
                    const rightData = rightCube.faces[f];
                    for (let lz = 0; lz < nz; lz++) {
                        const zOff = lz * ny * nx;
                        for (let ly = 1; ly < ny - 1; ly++) {
                            rightData[zOff + ly * nx] = data[zOff + ly * nx + nx - 2];
                        }
                    }
                }

                // Send Left
                if (x > 0 || this.isPeriodic) {
                    const leftCube = this.cubes[y][(x - 1 + this.cols) % this.cols]!;
                    const leftData = leftCube.faces[f];
                    for (let lz = 0; lz < nz; lz++) {
                        const zOff = lz * ny * nx;
                        for (let ly = 1; ly < ny - 1; ly++) {
                            leftData[zOff + ly * nx + nx - 1] = data[zOff + ly * nx + 1];
                        }
                    }
                }
            }
        }

        // PASS 2: Y-axis (Top/Bottom array.set copy)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                const data = cube.faces[f];

                if (y < this.rows - 1 || this.isPeriodic) {
                    const botCube = this.cubes[(y + 1) % this.rows][x]!;
                    const botData = botCube.faces[f];
                    for (let lz = 0; lz < nz; lz++) {
                        const srcOffset = lz * ny * nx + (ny - 2) * nx + 1;
                        const dstOffset = lz * ny * nx + 1;
                        botData.set(data.subarray(srcOffset, srcOffset + nx - 2), dstOffset);
                    }
                }

                if (y > 0 || this.isPeriodic) {
                    const topRow = (y === 0) ? this.rows - 1 : y - 1;
                    const topCube = this.cubes[topRow][x]!;
                    const topData = topCube.faces[f];
                    for (let lz = 0; lz < nz; lz++) {
                        const srcOffset = lz * ny * nx + nx + 1;
                        const dstOffset = lz * ny * nx + (ny - 1) * nx + 1;
                        topData.set(data.subarray(srcOffset, srcOffset + nx - 2), dstOffset);
                    }
                }
            }
        }

        // PASS 3: Corners (3D diagonal stacks)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                const data = cube.faces[f];

                const hasRight = this.isPeriodic || x < this.cols - 1;
                const hasLeft = this.isPeriodic || x > 0;
                const hasBot = this.isPeriodic || y < this.rows - 1;
                const hasTop = this.isPeriodic || y > 0;

                const rightIdx = (x + 1) % this.cols;
                const leftIdx = (x - 1 + this.cols) % this.cols;
                const botIdx = (y + 1) % this.rows;
                const topIdx = (y - 1 + this.rows) % this.rows;

                const trCubeFace = hasTop && hasRight ? this.cubes[topIdx][rightIdx]!.faces[f] : null;
                const tlCubeFace = hasTop && hasLeft ? this.cubes[topIdx][leftIdx]!.faces[f] : null;
                const brCubeFace = hasBot && hasRight ? this.cubes[botIdx][rightIdx]!.faces[f] : null;
                const blCubeFace = hasBot && hasLeft ? this.cubes[botIdx][leftIdx]!.faces[f] : null;

                const trSrcOffset = (ny - 2) * nx + nx - 2;
                const tlSrcOffset = (ny - 2) * nx + 1;
                const brSrcOffset = nx + nx - 2;
                const blSrcOffset = nx + 1;

                const trDstOffset = 0;
                const tlDstOffset = nx - 1;
                const brDstOffset = (ny - 1) * nx;
                const blDstOffset = (ny - 1) * nx + nx - 1;

                for (let lz = 0; lz < nz; lz++) {
                    const zOff = lz * ny * nx;
                    if (brCubeFace) brCubeFace[zOff + trDstOffset] = data[zOff + trSrcOffset];
                    if (blCubeFace) blCubeFace[zOff + tlDstOffset] = data[zOff + tlSrcOffset];
                    if (trCubeFace) trCubeFace[zOff + brDstOffset] = data[zOff + brSrcOffset];
                    if (tlCubeFace) tlCubeFace[zOff + blDstOffset] = data[zOff + blSrcOffset];
                }
            }
        }
    }

    public destroy() {
        if (this.workerPool) {
            this.workerPool.terminate();
            this.workerPool = null;
        }
    }
}
