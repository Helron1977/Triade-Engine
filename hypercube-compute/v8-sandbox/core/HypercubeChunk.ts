/// <reference types="@webgpu/types" />
import { HypercubeMasterBuffer } from './HypercubeMasterBuffer';
import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HypercubeGPUContext } from '../../src/core/gpu/HypercubeGPUContext';
import { HypercubeGpuResource } from './HypercubeGpuResource';

export class HypercubeChunk {
    public readonly nx: number;
    public readonly ny: number;
    public readonly nz: number;
    public readonly faces: Float32Array[] = [];

    // ── GPU REFACTO V5.4 ── Architecture déléguee
    public gpuResource: HypercubeGpuResource | null = null;

    public get gpuReadBuffer(): GPUBuffer | null { return this.gpuResource?.readBuffer ?? null; }
    public get gpuWriteBuffer(): GPUBuffer | null { return this.gpuResource?.writeBuffer ?? null; }
    public get gpuParity(): number { return this.gpuResource?.parity ?? 0; }

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

    initGPU(uniformBuffer?: GPUBuffer) {
        if (!this.engine) return;

        const totalSize = this.faces.length * this.stride;
        this.gpuResource = new HypercubeGpuResource(totalSize, `Chunk ${this.x},${this.y}`);
        this.gpuResource.init();

        this.pushToGPU();

    }

    /**
     * Uploads the current CPU faces (from masterBuffer) to the GPU buffers.
     * Useful for manual initialization of obstacles/states on CPU before starting GPU compute.
     */
    public pushToGPU() {
        if (!this.gpuResource || !this.gpuReadBuffer || !this.gpuWriteBuffer) return;
        const device = HypercubeGPUContext.device;
        const totalSize = this.faces.length * this.stride;
        device.queue.writeBuffer(this.gpuReadBuffer, 0, this.masterBuffer.buffer, this.offset, totalSize);
        device.queue.writeBuffer(this.gpuWriteBuffer, 0, this.masterBuffer.buffer, this.offset, totalSize);
    }

    swapGPUBuffers() {
        this.gpuResource?.swap();
    }

    async compute() {
        if (!this.engine) return;
        await (this.engine.compute as any)(this.faces, this.nx, this.ny, this.nz, this.x, this.y, this.z);
    }

    clearFace(faceIndex: number) {
        this.faces[faceIndex].fill(0);
    }

    async syncToHost(faceIndices?: number[], block: boolean = false): Promise<void> {
        if (!this.gpuResource || !this.gpuResource.readBuffer) return;

        const res = this.gpuResource;
        const totalSize = this.faces.length * this.stride;
        const device = HypercubeGPUContext.device;

        const readBuffer = res.readBuffer;
        const staging = res.stagingBuffer;
        if (!readBuffer || !staging || staging.mapState !== 'unmapped') return;

        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(readBuffer, 0, staging, 0, totalSize);
        device.queue.submit([encoder.finish()]);

        if (block) {
            await staging.mapAsync(GPUMapMode.READ);
            this.applyMappedRange(staging, totalSize);
        } else {
            staging.mapAsync(GPUMapMode.READ).then(() => {
                if (staging && staging.mapState === 'mapped') {
                    this.applyMappedRange(staging, totalSize);
                }
            }).catch(() => { });
        }
    }

    private applyMappedRange(staging: GPUBuffer, size: number) {
        const mapped = staging.getMappedRange();
        new Uint8Array(this.masterBuffer.buffer, this.offset, size).set(new Uint8Array(mapped));
        staging.unmap();
    }

    syncFromHost(faceIndices?: number[]) {
        if (!this.gpuReadBuffer) return;
        const device = HypercubeGPUContext.device;
        const totalSize = this.faces.length * this.stride;

        const doWrite = (buf: GPUBuffer) => {
            if (!faceIndices || faceIndices.length === 0) {
                device.queue.writeBuffer(buf, 0, this.masterBuffer.buffer, this.offset, totalSize);
            } else {
                for (const idx of faceIndices) {
                    const offset = idx * this.stride;
                    device.queue.writeBuffer(buf, offset, this.masterBuffer.buffer, this.offset + offset, this.stride);
                }
            }
        };

        doWrite(this.gpuReadBuffer);
        if (this.gpuWriteBuffer) doWrite(this.gpuWriteBuffer);
    }

    destroy() {
        this.gpuResource?.destroy();
        this.gpuResource = null;
    }
}

