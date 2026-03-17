import { IBufferBridge } from './IBufferBridge';
import { IMasterBuffer } from './topology/GridAbstractions';

/**
 * CpuBufferBridge: High-performance passthrough bridge for CPU computation.
 * Directly maps to MasterBuffer's SharedArrayBuffer views.
 */
export class CpuBufferBridge implements IBufferBridge {
    constructor(private mBuffer: IMasterBuffer) {}

    /**
     * Resolves the physical face views for a given chunk.
     */
    public getChunkViews(chunkId: string): Float32Array[] {
        return this.mBuffer.getChunkViews(chunkId).faces;
    }

    /**
     * Shared memory buffer between Main Thread and Workers.
     */
    public get rawBuffer(): SharedArrayBuffer | ArrayBuffer {
        return this.mBuffer.rawBuffer;
    }

    /** Stride in elements between face start and end. */
    public get strideFace(): number { return this.mBuffer.strideFace; }
    /** Total number of elements (all faces) per chunk. */
    public get totalSlotsPerChunk(): number { return this.mBuffer.totalSlotsPerChunk; }
    /** Always undefined for CPU Mode. */
    public get gpuBuffer(): any { return undefined; }

    public initializeEquilibrium(): void {
        this.mBuffer.initializeEquilibrium();
    }

    public setFaceData(chunkId: string, faceName: string, data: Float32Array, fillAllPingPong: boolean = false): void {
        this.mBuffer.setFaceData(chunkId, faceName, data, fillAllPingPong);
    }

    public syncToDevice(): void {
        // No-op for pure CPU mode
    }

    public async syncToHost(faceIndices?: number[]): Promise<void> {
        // No-op for pure CPU mode (data is already in SharedArrayBuffer)
    }

    public commit(): void {
        // No-op for synchronous CPU execution
    }
}
