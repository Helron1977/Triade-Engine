import { IBufferBridge } from './IBufferBridge';
import { IMasterBuffer } from '../topology/GridAbstractions';

/**
 * GpuBufferBridge: Orchestrates GPU-CPU memory synchronization.
 * Wraps MasterBuffer's WebGPU buffer logic with an agnostic API.
 */
export class GpuBufferBridge implements IBufferBridge {
    constructor(private mBuffer: IMasterBuffer) {}

    /**
     * Resolves the physical face views for a given chunk.
     * In GPU mode, these are local CPU mirrors used for injection or read-back.
     */
    public getChunkViews(chunkId: string): Float32Array[] {
        return this.mBuffer.getChunkViews(chunkId).faces;
    }

    /**
     * Shared memory buffer between Main Thread and Workers (CPU mirror).
     */
    public get rawBuffer(): SharedArrayBuffer | ArrayBuffer {
        return this.mBuffer.rawBuffer;
    }

    /** Stride in elements between face start and end in VRAM. */
    public get strideFace(): number { return this.mBuffer.strideFace; }
    /** Total number of elements (all faces) per chunk in VRAM. */
    public get totalSlotsPerChunk(): number { return this.mBuffer.totalSlotsPerChunk; }
    /** Reference to the underlying IDeviceBuffer (WebGPU). */
    public get gpuBuffer(): any { return this.mBuffer.gpuBuffer; }

    public initializeEquilibrium(): void {
        this.mBuffer.initializeEquilibrium();
    }

    public setFaceData(chunkId: string, faceName: string, data: Float32Array, fillAllPingPong: boolean = false): void {
        this.mBuffer.setFaceData(chunkId, faceName, data, fillAllPingPong);
    }

    public syncToDevice(): void {
        this.mBuffer.syncToDevice();
    }

    public async syncToHost(faceIndices?: number[]): Promise<void> {
        if (faceIndices && faceIndices.length > 0) {
            await this.mBuffer.syncFacesToHost(faceIndices);
        } else {
            await this.mBuffer.syncToHost();
        }
    }

    /**
     * Finalizes the memory state. 
     * In GPU mode, we avoid automatic full-buffer syncs in the main loop to prevent overwriting 
     * compute shader results with the CPU shadow buffer (Zero-Stall philosophy).
     */
    public commit(): void {
        // No-op. Call syncToDevice() explicitly if CPU-side changes (like rasterization) 
        // must be pushed to the GPU.
    }
}
