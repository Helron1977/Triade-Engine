import { HypercubeMasterBuffer } from './HypercubeMasterBuffer';
import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

export class HypercubeChunk {
    public readonly nx: number;
    public readonly ny: number;
    public readonly nz: number;
    public readonly faces: Float32Array[] = [];

    // ── GPU REFACTO V5.4 ── Double Buffering propre
    public gpuReadBuffer: GPUBuffer | null = null;   // Buffer actuellement lu par le renderer + prochain compute
    public gpuWriteBuffer: GPUBuffer | null = null;  // Buffer dans lequel le compute écrit
    public gpuParity: number = 0;                    // 0 = read, 1 = write

    /** @deprecated Use gpuReadBuffer instead */
    public get gpuBuffer(): GPUBuffer | null {
        return this.gpuReadBuffer;
    }

    public readonly offset: number;
    public readonly stride: number;
    public engine: IHypercubeEngine | null = null;
    public readonly x: number;
    public readonly y: number;
    public readonly z: number;
    private masterBuffer: HypercubeMasterBuffer;

    // ── GPU REFACTO V5.4 ── Staging buffer partagé (un seul par chunk, réutilisé)
    private stagingBuffer: GPUBuffer | null = null;

    constructor(
        x: number, y: number, nx: number, ny: number, nz: number = 1,
        masterBuffer: HypercubeMasterBuffer, numFaces: number = 6, z: number = 0
    ) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.masterBuffer = masterBuffer;
        this.nx = nx;
        this.ny = ny;
        this.nz = nz;

        const allocation = masterBuffer.allocateCube(nx, ny, nz, numFaces);
        this.offset = allocation.offset;
        this.stride = allocation.stride;

        const floatCount = nx * ny * nz;
        for (let i = 0; i < numFaces; i++) {
            this.faces.push(
                new Float32Array(masterBuffer.buffer, this.offset + (i * this.stride), floatCount)
            );
        }
    }

    public getIndex(lx: number, ly: number, lz: number = 0): number {
        return (lz * this.ny * this.nx) + (ly * this.nx) + lx;
    }

    public getSlice(faceIndex: number, lz: number): Float32Array {
        const sliceSize = this.nx * this.ny;
        const offset = lz * sliceSize;
        return this.faces[faceIndex].slice(offset, offset + sliceSize);
    }

    setEngine(engine: IHypercubeEngine) {
        this.engine = engine;
    }

    // ── GPU REFACTO V5.4 ──
    initGPU() {
        if (!this.engine) return;

        const totalSize = this.faces.length * this.stride;
        const device = HypercubeGPUContext.device;

        this.gpuReadBuffer = HypercubeGPUContext.createStorageBuffer(totalSize);
        this.gpuWriteBuffer = HypercubeGPUContext.createStorageBuffer(totalSize);

        // Upload initial des données CPU vers les deux buffers
        device.queue.writeBuffer(this.gpuReadBuffer, 0, this.masterBuffer.buffer, this.offset, totalSize);
        device.queue.writeBuffer(this.gpuWriteBuffer, 0, this.masterBuffer.buffer, this.offset, totalSize);

        if (this.engine.initGPU) {
            this.engine.initGPU(
                device,
                this.gpuReadBuffer,
                this.gpuWriteBuffer,
                this.stride,
                this.nx, this.ny, this.nz
            );
        }
    }

    // ── GPU REFACTO V5.4 ──
    swapGPUBuffers() {
        if (this.gpuReadBuffer && this.gpuWriteBuffer) {
            [this.gpuReadBuffer, this.gpuWriteBuffer] = [this.gpuWriteBuffer, this.gpuReadBuffer];
            this.gpuParity = 1 - this.gpuParity;
        }
    }

    /** Méthode CPU – inchangée */
    async compute() {
        if (!this.engine) return;
        await (this.engine.compute as any)(this.faces, this.nx, this.ny, this.nz, this.x, this.y, this.z);
    }

    clearFace(faceIndex: number) {
        this.faces[faceIndex].fill(0);
    }

    // ── GPU REFACTO V5.4 ── Lecture asynchrone non-bloquante
    async syncToHost(faceIndices?: number[], block: boolean = false): Promise<void> {
        if (!this.gpuReadBuffer) return;

        const device = HypercubeGPUContext.device;
        const totalSize = this.faces.length * this.stride;

        if (!this.stagingBuffer) {
            this.stagingBuffer = device.createBuffer({
                size: totalSize,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                label: `Staging Buffer Chunk ${this.x},${this.y}`
            });
        }

        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.gpuReadBuffer, 0, this.stagingBuffer, 0, totalSize);
        device.queue.submit([encoder.finish()]);

        if (block) {
            await this.stagingBuffer.mapAsync(GPUMapMode.READ);
            const mapped = this.stagingBuffer.getMappedRange();
            new Uint8Array(this.masterBuffer.buffer, this.offset, totalSize)
                .set(new Uint8Array(mapped));
            this.stagingBuffer.unmap();
        } else {
            // ── GPU REFACTO V5.4 ── Mapping asynchrone (Non-Bloquant)
            // On ne tente le mapping que si le buffer est libre (évite les erreurs d'overlap)
            if (this.stagingBuffer.mapState === 'unmapped') {
                this.stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
                    if (this.stagingBuffer && this.stagingBuffer.mapState === 'mapped') {
                        const mapped = this.stagingBuffer.getMappedRange();
                        new Uint8Array(this.masterBuffer.buffer, this.offset, totalSize)
                            .set(new Uint8Array(mapped));
                        this.stagingBuffer.unmap();
                    }
                }).catch(() => { /* On ignore les échecs de lecture différée */ });
            }
        }
    }

    // ── GPU REFACTO V5.4 ──
    syncFromHost(faceIndices?: number[]) {
        if (!this.gpuReadBuffer) return;
        const device = HypercubeGPUContext.device;
        const totalSize = this.faces.length * this.stride;

        if (!faceIndices || faceIndices.length === 0) {
            device.queue.writeBuffer(this.gpuReadBuffer, 0, this.masterBuffer.buffer, this.offset, totalSize);
            if (this.gpuWriteBuffer) {
                device.queue.writeBuffer(this.gpuWriteBuffer, 0, this.masterBuffer.buffer, this.offset, totalSize);
            }
        } else {
            for (const idx of faceIndices) {
                const offset = idx * this.stride;
                device.queue.writeBuffer(this.gpuReadBuffer, offset, this.masterBuffer.buffer, this.offset + offset, this.stride);
                if (this.gpuWriteBuffer) {
                    device.queue.writeBuffer(this.gpuWriteBuffer, offset, this.masterBuffer.buffer, this.offset + offset, this.stride);
                }
            }
        }
    }

    // ── GPU REFACTO V5.4 ── Nettoyage propre
    destroy() {
        this.gpuReadBuffer?.destroy();
        this.gpuWriteBuffer?.destroy();
        this.stagingBuffer?.destroy();

        this.gpuReadBuffer = null;
        this.gpuWriteBuffer = null;
        this.stagingBuffer = null;
    }
}
