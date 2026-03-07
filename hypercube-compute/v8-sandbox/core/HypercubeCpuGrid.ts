/// <reference types="@webgpu/types" />
import { HypercubeMasterBuffer } from './HypercubeMasterBuffer';
import { HypercubeChunk } from './HypercubeChunk';
import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HypercubeWorkerPool } from './HypercubeWorkerPool';
/// <reference types="@webgpu/types" />
import { HypercubeGPUContext } from '../../src/core/gpu/HypercubeGPUContext';
import { V8_METADATA_OFFSET } from './UniformPresets';
import { BoundaryConditions, BoundaryConfig, BoundaryType } from './BoundaryConditions';

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
    public readonly numFaces: number;
    public boundaryConfig: any; // Declarative V8

    // ── V8 OPTIM: Centralized Uniforms ──
    public uniforms: Float32Array = new Float32Array(512); // Padded for safety
    public chunkUniformBuffers: (GPUBuffer | null)[][] = []; // Isolated per-chunk

    public masterBuffer: HypercubeMasterBuffer;
    public mode: 'cpu' | 'gpu';
    public isPeriodic: boolean;

    public stats = {
        computeTimeMs: 0,
        syncTimeMs: 0,
        frameCount: 0,
        gpuWorkMs: 0,
        dispatchCount: 0,
        copyCount: 0
    };

    private frameCounter: number = 0;

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
        useWorkers: boolean = true,
        mode: 'cpu' | 'gpu' = 'cpu'
    ) {
        this.cols = cols;
        this.rows = rows;
        this.mode = mode;
        this.masterBuffer = masterBuffer;
        this.numFaces = numFaces;
        this.isPeriodic = isPeriodic;
        this._engineFactory = engineFactory;
        this.useWorkers = useWorkers;

        if (typeof resolution === 'number') {
            this.nx = resolution;
            this.ny = resolution;
            this.nz = 1;
        } else {
            this.nx = resolution.nx;
            this.ny = resolution.ny;
            this.nz = resolution.nz ?? 1;
        }

        // Auto-detect faces
        const tempEngine = engineFactory();
        const requiredFaces = tempEngine.getRequiredFaces();
        const finalNumFaces = Math.max(numFaces, requiredFaces);

        // Allocate cubes
        for (let y = 0; y < rows; y++) {
            this.cubes[y] = [];
            this.chunkUniformBuffers[y] = [];
            for (let x = 0; x < cols; x++) {
                const cube = new HypercubeChunk(x, y, this.nx, this.ny, this.nz, masterBuffer, finalNumFaces);
                const engineInstance = y === 0 && x === 0 ? tempEngine : engineFactory();
                cube.setEngine(engineInstance);
                engineInstance.init(cube.faces, cube.nx, cube.ny, cube.nz, false);
                this.cubes[y][x] = cube;

                // GPU Initialization
                if (this.mode === 'gpu') {
                    const buffer = HypercubeGPUContext.device.createBuffer({
                        size: this.uniforms.byteLength,
                        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                        label: `Uniform Buffer Chunk ${x},${y}`
                    });
                    this.chunkUniformBuffers[y][x] = buffer;
                    cube.initGPU(buffer);
                }
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
        workerScriptPath?: string,
        mode: 'cpu' | 'gpu' = 'cpu'
    ): Promise<HypercubeCpuGrid> {
        const grid = new HypercubeCpuGrid(cols, rows, resolution, masterBuffer, engineFactory, numFaces, isPeriodic, useWorkers, mode);

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
     * Pushes all CPU data (obstacles, initial states) from all chunks to GPU VRAM.
     */
    public pushToGPU() {
        if (this.mode !== 'gpu') return;

        // 1. Uniforms (Broadcast current state to all chunks)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const buffer = this.chunkUniformBuffers[y][x];
                if (buffer) {
                    HypercubeGPUContext.device.queue.writeBuffer(buffer, 0, new Float32Array(this.uniforms));
                }
            }
        }

        // 2. Chunks data
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                this.cubes[y][x]?.pushToGPU();
            }
        }
    }

    /**
     * Executes the computational pass, then synchronizes physical memory boundaries.
     */
    async compute(facesToSynchronize?: number | number[]) {
        if (this.stats.frameCount === undefined) (this.stats as any).frameCount = 0;
        (this.stats as any).frameCount++;

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

        if (this.mode === 'gpu') {
            const device = HypercubeGPUContext.device;
            const commandEncoder = device.createCommandEncoder({ label: 'Hypercube GPU Pass (CpuGrid)' });
            const flatCubes = this.cubes.flat().filter(c => c !== null) as HypercubeChunk[];

            this.stats.dispatchCount = 0;
            this.stats.copyCount = 0;

            // 1. Dispatch GPU Compute (Isolated Uniforms per Chunk)
            for (let y = 0; y < this.rows; y++) {
                for (let x = 0; x < this.cols; x++) {
                    const cube = this.cubes[y][x];
                    const buffer = this.chunkUniformBuffers[y][x];
                    if (cube && buffer && cube.engine?.computeGPU) {
                        // V8 Metadata Update (Source of Truth)
                        this.uniforms[V8_METADATA_OFFSET + 0] = cube.nx;
                        this.uniforms[V8_METADATA_OFFSET + 1] = cube.ny;
                        this.uniforms[V8_METADATA_OFFSET + 2] = cube.nz;
                        this.uniforms[V8_METADATA_OFFSET + 3] = cube.stride;
                        this.uniforms[V8_METADATA_OFFSET + 4] = cube.gpuResource?.parity ?? 0;
                        this.uniforms[V8_METADATA_OFFSET + 5] = performance.now() / 1000;
                        this.uniforms[V8_METADATA_OFFSET + 6] = x * (cube.nx - 2); // offX
                        this.uniforms[V8_METADATA_OFFSET + 7] = y * (cube.ny - 2); // offY

                        // Global Boundary Roles (Inlet=1, Outlet=2, Wall=3)
                        // V8 FIX: Only apply if chunk is at the Grid edge
                        const roleToNum = (role: string) => role === 'inlet' ? 1 : (role === 'outlet' ? 2 : (role === 'wall' ? 3 : 0));
                        if (this.boundaryConfig) {
                            this.uniforms[V8_METADATA_OFFSET + 8] = (x === 0) ? roleToNum(this.boundaryConfig.left?.role || '') : 0;
                            this.uniforms[V8_METADATA_OFFSET + 9] = (x === this.cols - 1) ? roleToNum(this.boundaryConfig.right?.role || '') : 0;
                            this.uniforms[V8_METADATA_OFFSET + 10] = (y === 0) ? roleToNum(this.boundaryConfig.top?.role || '') : 0;
                            this.uniforms[V8_METADATA_OFFSET + 11] = (y === this.rows - 1) ? roleToNum(this.boundaryConfig.bottom?.role || '') : 0;

                            this.uniforms[V8_METADATA_OFFSET + 12] = (x === 0) ? ((this.boundaryConfig.left as any)?.value || 0) : 0;
                            this.uniforms[V8_METADATA_OFFSET + 13] = (x === this.cols - 1) ? ((this.boundaryConfig.right as any)?.value || 0) : 0;
                            this.uniforms[V8_METADATA_OFFSET + 14] = (y === 0) ? ((this.boundaryConfig.top as any)?.value || 0) : 0;
                            this.uniforms[V8_METADATA_OFFSET + 15] = (y === this.rows - 1) ? ((this.boundaryConfig.bottom as any)?.value || 0) : 0;
                        }

                        device.queue.writeBuffer(buffer, 0, new Float32Array(this.uniforms));

                        cube.engine.computeGPU(
                            device,
                            commandEncoder,
                            cube.nx,
                            cube.ny,
                            cube.nz,
                            cube.gpuReadBuffer!,
                            cube.gpuWriteBuffer!
                        );
                        this.stats.dispatchCount += 2;
                    }
                }
            }

            // 2. Synchronize Boundaries directly in VRAM
            if (this.cols > 1 || this.rows > 1) {
                const syncStart = performance.now();
                const headEngine = flatCubes[0]?.engine;
                const syncFaces = headEngine?.getSyncFaces ? headEngine.getSyncFaces() : [0];
                for (const f of syncFaces) {
                    this.stats.copyCount += this.synchronizeBoundariesGPU(commandEncoder, f);
                }
                this.stats.syncTimeMs = performance.now() - syncStart;
            }

            device.queue.submit([commandEncoder.finish()]);

            // 3. Swap buffers locally
            for (const cube of flatCubes) {
                cube.swapGPUBuffers();
                if (cube.engine && 'parity' in cube.engine) {
                    (cube.engine as any).parity = cube.gpuParity;
                }
            }

            this.stats.computeTimeMs = performance.now() - computeStart;
            this.frameCounter++;
            if (this.frameCounter % 60 === 0) {
                console.log(`%c[Robinet GPU] Frame ${this.frameCounter} | Total: ${this.stats.computeTimeMs.toFixed(2)}ms | Sync: ${this.stats.syncTimeMs.toFixed(2)}ms | Kernels: ${this.stats.dispatchCount} | VRAM Copies: ${this.stats.copyCount}`, "color: #00ff00; font-weight: bold;");
            }
        } else if (this.workerPool && this.masterBuffer.buffer instanceof SharedArrayBuffer) {
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

            await this.workerPool.computeAll(flatCubes, this.masterBuffer.buffer, {
                name: engineName,
                config: engineConfigs,
                descriptor: (this as any)._descriptor
            });
        } else {
            for (let y = 0; y < this.rows; y++) {
                for (let x = 0; x < this.cols; x++) {
                    const cube = this.cubes[y][x];
                    if (cube) {
                        await cube.compute();
                    }
                }
            }
        }

        const computeEnd = performance.now();
        this.stats.computeTimeMs = computeEnd - computeStart;

        // 1x1 grids don't need boundary sync
        if (this.cols > 1 || this.rows > 1) {
            if (this.mode === 'cpu') {
                const syncStart = performance.now();
                let faces: number[];
                if (facesToSynchronize !== undefined) {
                    faces = Array.isArray(facesToSynchronize) ? facesToSynchronize : [facesToSynchronize];
                } else {
                    const engine = this.getEngine();
                    faces = (engine && engine.getSyncFaces) ? engine.getSyncFaces() : [0];
                }

                for (const f of faces) {
                    this.synchronizeBoundaries(f);
                }
                const syncEnd = performance.now();
                this.stats.syncTimeMs = syncEnd - syncStart;
            }
        }

        // --- GLOBAL PARITY TOGGLE ---
        // Must happen AFTER all compute and sync steps to prepare the NEXT frame.
        if (this.mode !== 'gpu') {
            for (let y = 0; y < this.rows; y++) {
                for (let x = 0; x < this.cols; x++) {
                    const cube = this.cubes[y][x];
                    if (cube?.engine && (cube.engine as any).parity !== undefined) {
                        (cube.engine as any).parity = 1 - (cube.engine as any).parity;
                    }
                }
            }
        }
    }

    private getEngine() {
        for (const row of this.cubes) {
            for (const cube of row) {
                if (cube?.engine) return cube.engine;
            }
        }
        return null;
    }

    /**
     * GPU-to-GPU Boundary Exchange using copyBufferToBuffer.
     * Operates directly in VRAM without CPU readbacks.
     */
    private synchronizeBoundariesGPU(commandEncoder: GPUCommandEncoder, faceIndex: number): number {
        const nx = this.nx;
        const ny = this.ny;
        const nz = this.nz;
        const stride = nx * ny * nz;
        const faceOffset = faceIndex * stride * 4; // BYTE OFFSET
        const rowSize = nx * 4;
        const innerRowSize = (nx - 2) * 4;

        const flatCubes = this.cubes.flat().filter(c => c !== null) as HypercubeChunk[];
        if (flatCubes.length === 0) return 0;

        let copies = 0;
        const encoder = commandEncoder;

        // V8 Rule: Sync border of NEW result (writeBuffer) to ghost cells of NEW result (writeBuffer neighbors)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const targetCube = this.cubes[y][x];
                if (!targetCube) continue;

                // IMPORTANT: In GPU mode, compute pass wrote to writeBuffer.
                // We sync the boundaries of that same writeBuffer across chunks.
                const srcBuf = targetCube.gpuWriteBuffer!;

                // 1. X-Axis (Left/Right)
                if (x < this.cols - 1 || this.isPeriodic) {
                    const rightCube = this.cubes[y][(x + 1) % this.cols]!;
                    const dstBuf = rightCube.gpuWriteBuffer!;
                    for (let lz = 0; lz < nz; lz++) {
                        const zOff = lz * ny * nx * 4;
                        copies += HypercubeGPUContext.gpuCopyBoundary(
                            encoder, srcBuf, dstBuf,
                            faceOffset + zOff + 1 * nx * 4 + (nx - 2) * 4, // Source: x=nx-2
                            faceOffset + zOff + 1 * nx * 4,               // Dest: x=0
                            4,
                            ny - 2,
                            nx * 4,
                            nx * 4
                        );
                    }
                }

                if (x > 0 || this.isPeriodic) {
                    const leftCube = this.cubes[y][(x - 1 + this.cols) % this.cols]!;
                    const dstBuf = leftCube.gpuWriteBuffer!;
                    for (let lz = 0; lz < nz; lz++) {
                        const zOff = lz * ny * nx * 4;
                        copies += HypercubeGPUContext.gpuCopyBoundary(
                            encoder, srcBuf, dstBuf,
                            faceOffset + zOff + 1 * nx * 4 + 4,          // Source: x=1
                            faceOffset + zOff + 1 * nx * 4 + (nx - 1) * 4, // Dest: x=nx-1
                            4,
                            ny - 2,
                            nx * 4,
                            nx * 4
                        );
                    }
                }

                // 2. Y-Axis (Top/Bottom)
                if (y < this.rows - 1 || this.isPeriodic) {
                    const botCube = this.cubes[(y + 1) % this.rows][x]!;
                    const dstBuf = botCube.gpuWriteBuffer!;
                    for (let lz = 0; lz < nz; lz++) {
                        const zOff = lz * ny * nx * 4;
                        copies += HypercubeGPUContext.gpuCopyBoundary(
                            encoder, srcBuf, dstBuf,
                            faceOffset + zOff + (ny - 2) * rowSize + 4, // Source: y=ny-2
                            faceOffset + zOff + 4,                      // Dest: y=0
                            innerRowSize
                        );
                    }
                }

                if (y > 0 || this.isPeriodic) {
                    const topCube = this.cubes[(y - 1 + this.rows) % this.rows][x]!;
                    const dstBuf = topCube.gpuWriteBuffer!;
                    for (let lz = 0; lz < nz; lz++) {
                        const zOff = lz * ny * nx * 4; // BYTE OFFSET
                        copies += HypercubeGPUContext.gpuCopyBoundary(
                            encoder, srcBuf, dstBuf,
                            faceOffset + zOff + rowSize + 4,              // Source: y=1
                            faceOffset + zOff + (ny - 1) * rowSize + 4,   // Dest: y=ny-1
                            innerRowSize
                        );
                    }
                }

                // 3. Corners - Single float copies
                const hasRight = this.isPeriodic || x < this.cols - 1;
                const hasLeft = this.isPeriodic || x > 0;
                const hasBot = this.isPeriodic || y < this.rows - 1;
                const hasTop = this.isPeriodic || y > 0;

                const rIdx = (x + 1) % this.cols;
                const lIdx = (x - 1 + this.cols) % this.cols;
                const bIdx = (y + 1) % this.rows;
                const tIdx = (y - 1 + this.rows) % this.rows;

                for (let lz = 0; lz < nz; lz++) {
                    const zOff = lz * ny * nx * 4; // BYTE OFFSET
                    // Bottom-Right -> Neighbor's Top-Left Ghost
                    if (hasBot && hasRight) {
                        const target = this.cubes[bIdx][rIdx]!.gpuWriteBuffer!;
                        encoder.copyBufferToBuffer(srcBuf, faceOffset + zOff + (ny - 2) * rowSize + (nx - 2) * 4, target, faceOffset + zOff, 4);
                        copies++;
                    }
                    // Bottom-Left -> Neighbor's Top-Right Ghost
                    if (hasBot && hasLeft) {
                        const target = this.cubes[bIdx][lIdx]!.gpuWriteBuffer!;
                        encoder.copyBufferToBuffer(srcBuf, faceOffset + zOff + (ny - 2) * rowSize + 4, target, faceOffset + zOff + (nx - 1) * 4, 4);
                        copies++;
                    }
                    // Top-Right -> Neighbor's Bottom-Left Ghost
                    if (hasTop && hasRight) {
                        const target = this.cubes[tIdx][rIdx]!.gpuWriteBuffer!;
                        encoder.copyBufferToBuffer(srcBuf, faceOffset + zOff + rowSize + (nx - 2) * 4, target, faceOffset + zOff + (ny - 1) * rowSize, 4);
                        copies++;
                    }
                    // Top-Left -> Neighbor's BR Ghost
                    if (hasTop && hasLeft) {
                        const target = this.cubes[tIdx][lIdx]!.gpuWriteBuffer!;
                        encoder.copyBufferToBuffer(srcBuf, faceOffset + zOff + rowSize + 4, target, faceOffset + zOff + (ny - 1) * rowSize + (nx - 1) * 4, 4);
                        copies++;
                    }
                }
            }
        }
        return copies;
    }

    public synchronizeBoundaries(f: number) {
        const nx = this.nx;
        const ny = this.ny;
        const nz = this.nz;

        // Verify face existence across grid
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                if (!this.cubes[y][x]?.faces[f]) return;
            }
        }

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
                        const srcOffset = lz * ny * nx + (ny - 2) * nx;
                        const dstOffset = lz * ny * nx;
                        // Sync full row coverage (0 to nx-1)
                        botData.set(data.subarray(srcOffset, srcOffset + nx), dstOffset);
                    }
                }

                if (y > 0 || this.isPeriodic) {
                    const topRow = (y === 0) ? this.rows - 1 : y - 1;
                    const topCube = this.cubes[topRow][x]!;
                    const topData = topCube.faces[f];
                    for (let lz = 0; lz < nz; lz++) {
                        const srcOffset = lz * ny * nx + nx;
                        const dstOffset = lz * ny * nx + (ny - 1) * nx;
                        topData.set(data.subarray(srcOffset, srcOffset + nx), dstOffset);
                    }
                }
            }
        }

        // PASS 3: Corners (Explicitly mapped cells)
        for (let y = 0; y < this.rows; y++) {
            for (let x = 0; x < this.cols; x++) {
                const cube = this.cubes[y][x]!;
                const data = cube.faces[f];

                const hasR = this.isPeriodic || x < this.cols - 1;
                const hasL = this.isPeriodic || x > 0;
                const hasB = this.isPeriodic || y < this.rows - 1;
                const hasT = this.isPeriodic || y > 0;

                const rI = (x + 1) % this.cols;
                const lI = (x - 1 + this.cols) % this.cols;
                const bI = (y + 1) % this.rows;
                const tI = (y - 1 + this.rows) % this.rows;

                for (let lz = 0; lz < nz; lz++) {
                    const zOff = lz * ny * nx;

                    // BR (nx-2, ny-2) -> Bottom-Right Neighbor's TL Ghost (0,0)
                    if (hasB && hasR) this.cubes[bI][rI]!.faces[f][zOff] = data[zOff + (ny - 2) * nx + nx - 2];

                    // BL (1, ny-2) -> Bottom-Left Neighbor's TR Ghost (nx-1, 0)
                    if (hasB && hasL) this.cubes[bI][lI]!.faces[f][zOff + nx - 1] = data[zOff + (ny - 2) * nx + 1];

                    // TR (nx-2, 1) -> Top-Right Neighbor's BL Ghost (0, ny-1)
                    if (hasT && hasR) this.cubes[tI][rI]!.faces[f][zOff + (ny - 1) * nx] = data[zOff + nx + nx - 2];

                    // TL (1, 1) -> Top-Left Neighbor's BR Ghost (nx-1, ny-1)
                    if (hasT && hasL) this.cubes[tI][lI]!.faces[f][zOff + (ny - 1) * nx + nx - 1] = data[zOff + nx + 1];
                }
            }
        }
    }

    /**
     * Resolves global coordinates to a specific chunk and local index.
     */
    public getChunkAt(gx: number, gy: number, gz: number = 0): { cube: HypercubeChunk, lx: number, ly: number, lz: number } | null {
        const vnx = this.nx - 2;
        const vny = this.ny - 2;
        const vnz = this.nz > 1 ? this.nz - 2 : 1;

        const cx = Math.floor(gx / vnx);
        const cy = Math.floor(gy / vny);
        const cz = this.nz > 1 ? Math.floor(gz / vnz) : 0;

        if (cx < 0 || cx >= this.cols || cy < 0 || cy >= this.rows) return null;

        const cube = this.cubes[cy][cx];
        if (!cube) return null;

        return {
            cube,
            lx: (gx % vnx) + 1,
            ly: (gy % vny) + 1,
            lz: this.nz > 1 ? (gz % vnz) + 1 : 0
        };
    }

    /**
     * Sets a value at global (world) coordinates across any chunk.
     */
    public setAt(gx: number, gy: number, gz: number, face: number, value: number) {
        const res = this.getChunkAt(gx, gy, gz);
        if (res) {
            const { cube, lx, ly, lz } = res;
            const idx = lz * cube.ny * cube.nx + ly * cube.nx + lx;
            cube.faces[face][idx] = value;
        }
    }

    /**
     * Paints a circle (or sphere in 3D) at global coordinates.
     */
    public paintCircle(gx: number, gy: number, gz: number, face: number, radius: number, value: number) {
        const vnx = this.nx - 2;
        const vny = this.ny - 2;

        // Iterate through all affected chunks
        const minCX = Math.max(0, Math.floor((gx - radius) / vnx));
        const maxCX = Math.min(this.cols - 1, Math.floor((gx + radius) / vnx));
        const minCY = Math.max(0, Math.floor((gy - radius) / vny));
        const maxCY = Math.min(this.rows - 1, Math.floor((gy + radius) / vny));

        for (let cy = minCY; cy <= maxCY; cy++) {
            for (let cx = minCX; cx <= maxCX; cx++) {
                const cube = this.cubes[cy][cx]!;
                for (let lz = 0; lz < cube.nz; lz++) {
                    const worldZ = this.nz > 1 ? lz - 1 : 0;
                    for (let ly = 1; ly < cube.ny - 1; ly++) {
                        const worldY = cy * vny + (ly - 1);
                        for (let lx = 1; lx < cube.nx - 1; lx++) {
                            const worldX = cx * vnx + (lx - 1);

                            const dz = this.nz > 1 ? (worldZ - gz) : 0;
                            const distSq = (worldX - gx) ** 2 + (worldY - gy) ** 2 + dz ** 2;
                            if (distSq < radius * radius) {
                                const idx = lz * cube.ny * cube.nx + ly * cube.nx + lx;
                                cube.faces[face][idx] = value;
                            }
                        }
                    }
                }
            }
        }

        // Sync boundaries after global paint
        this.synchronizeBoundaries(face);
    }

    /**
     * Specialized LBM helper to apply an equilibrium state globally.
     * Use this for "splashes" or initial conditions in LBM engines.
     */
    public applyEquilibrium(gx: number, gy: number, gz: number, radius: number, rho: number, ux: number, uy: number) {
        const vnx = this.nx - 2;
        const vny = this.ny - 2;

        for (let cy = 0; cy < this.rows; cy++) {
            for (let cx = 0; cx < this.cols; cx++) {
                const cube = this.cubes[cy][cx]!;
                const engine = cube.engine;
                if (!engine?.getEquilibrium) continue;

                // Precompute equilibrium
                const fEq = engine.getEquilibrium(rho, ux, uy);

                for (let ly = 1; ly < cube.ny - 1; ly++) {
                    const worldY = cy * vny + (ly - 1);
                    for (let lx = 1; lx < cube.nx - 1; lx++) {
                        const worldX = cx * vnx + (lx - 1);
                        const distSq = (worldX - gx) ** 2 + (worldY - gy) ** 2;
                        if (distSq < radius * radius) {
                            const idx = ly * cube.nx + lx;
                            // Apply to fi (0-8) and f_post (9-17)
                            for (let k = 0; k < 9; k++) {
                                cube.faces[k][idx] = fEq[k];
                                cube.faces[k + 9][idx] = fEq[k];
                            }
                            // Also set macro rho/u if faces exist
                            if (cube.faces[22]) cube.faces[22][idx] = rho;
                            if (cube.faces[19]) cube.faces[19][idx] = ux;
                            if (cube.faces[20]) cube.faces[20][idx] = uy;
                        }
                    }
                }
            }
        }

        // Sync boundaries for ALL populations and macros
        const engine = this.cubes[0][0]?.engine; // Assuming getEngine() is a typo and keeping original way to get engine
        if (engine) {
            // We must sync BOTH sets of populations and all macro faces
            // because we just initialized them all.
            const facesToSync = [
                0, 1, 2, 3, 4, 5, 6, 7, 8,    // Populations Set A
                9, 10, 11, 12, 13, 14, 15, 16, 17, // Populations Set B
                19, 20, 22, 23, 24             // Macros + Bio
            ];
            for (const f of facesToSync) {
                this.synchronizeBoundaries(f);
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
