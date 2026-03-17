/**
 * IBufferBridge: The bridge between numerical logic and physical memory.
 * It abstracts synchronization (GPU/CPU/Worker) and chunk-specific memory access.
 */
export interface IBufferBridge {
    /**
     * Retrieves the raw memory views (Float32Array) for a specific chunk.
     * These views are used by kernels to read and write grid data.
     */
    getChunkViews(chunkId: string): Float32Array[];

    /**
     * The underlying memory buffer (SharedArrayBuffer or ArrayBuffer).
     * Used for Worker initialization and cross-thread sharing.
     */
    readonly rawBuffer: SharedArrayBuffer | ArrayBuffer;

    /**
     * GPU-specific properties (optional for CPU-only bridges).
     */
    readonly strideFace: number;
    readonly totalSlotsPerChunk: number;
    readonly gpuBuffer?: any;

    /**
     * Initializes the buffer state to equilibrium (zeroes or baseline values).
     */
    initializeEquilibrium(): void;

    /**
     * Directly injects data into a specific face.
     * @param chunkId ID of the target chunk
     * @param faceName Name of the target face
     * @param data The data to inject
     * @param fillAllPingPong Whether to fill both ping-pong buffers (Mode-aware)
     */
    setFaceData(chunkId: string, faceName: string, data: Float32Array, fillAllPingPong?: boolean): void;

    /**
     * Synchronizes host memory (CPU) with device memory (GPU).
     * Used before computation starts to inject dynamic changes.
     */
    syncToDevice(): void;

    /**
     * Synchronizes device memory (GPU) back to host memory (CPU).
     * Used after computation to allow visualization or CPU-side inspection.
     * @param faceIndices Optional list of specific face indices to sync (optimization).
     */
    syncToHost(faceIndices?: number[]): Promise<void>;

    /**
     * Logical commit of all pending computations.
     * In some implementations, this might be a no-op; in others, it triggers pipeline execution.
     */
    commit(): void;
}
