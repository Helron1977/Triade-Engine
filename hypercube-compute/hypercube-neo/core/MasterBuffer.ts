import { IMasterBuffer, IPhysicalChunk, IVirtualGrid } from './topology/GridAbstractions';
import { DataContract } from './DataContract';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

/**
 * The MasterBuffer is the physical memory anchor for Hypercube Neo.
 * It allocates a single contiguous buffer and creates zero-copy views for each chunk.
 */
export class MasterBuffer implements IMasterBuffer {
    /** Raw underlying memory (SharedArrayBuffer for CPU, ArrayBuffer mirror for GPU). */
    public readonly rawBuffer: SharedArrayBuffer | ArrayBuffer;
    /** Total length of the buffer in bytes. */
    public readonly byteLength: number;
    /** Number of face slots (including ping-pong) allocated per chunk. */
    public readonly totalSlotsPerChunk: number;
    /** WebGPU Storage Buffer reference (GPU mode only). */
    public gpuBuffer?: GPUBuffer;
    public strideFace: number = 0; // Number of float32 elements per face (including alignment padding)
    private chunkViews: Map<string, IPhysicalChunk> = new Map();

    constructor(private vGrid: IVirtualGrid) {
        // First pass: calculate alignment-safe total length
        const grid = this.vGrid as any;
        const dataContract = grid.dataContract as DataContract;
        const faceMappings = dataContract.getFaceMappings();

        // Find the maximum local dimensions to ensure uniform stride across all chunks
        let maxNx = 0, maxNy = 0, maxNz = 0;
        for (const chunk of this.vGrid.chunks) {
            maxNx = Math.max(maxNx, chunk.localDimensions.nx);
            maxNy = Math.max(maxNy, chunk.localDimensions.ny);
            maxNz = Math.max(maxNz, chunk.localDimensions.nz ?? 1);
        }

        const ghosts = dataContract.descriptor.requirements.ghostCells;
        const cellsPerFaceRaw = (maxNx + 2 * ghosts) * (maxNy + 2 * ghosts) * (grid.config.dimensions.nz > 1 ? (maxNz + 2 * ghosts) : 1);

        // WebGPU Alignment: Storage buffer offsets must be multiples of 256 bytes.
        const bytesPerFaceRaw = cellsPerFaceRaw * 4;
        const bytesPerFaceAligned = Math.ceil(bytesPerFaceRaw / 256) * 256;
        this.strideFace = bytesPerFaceAligned / 4;

        const facesPerChunk = faceMappings.reduce((acc, f) => acc + (f.isPingPong ? 2 : 1), 0);
        this.totalSlotsPerChunk = facesPerChunk;

        this.byteLength = grid.chunks.length * this.totalSlotsPerChunk * bytesPerFaceAligned;

        const mode = grid.config.mode;
        if (mode === 'gpu') {
            if (!HypercubeGPUContext.isInitialized) {
                throw new Error("MasterBuffer: GPU Mode requested but HypercubeGPUContext not initialized.");
            }
            this.gpuBuffer = HypercubeGPUContext.createStorageBuffer(this.byteLength);
            this.rawBuffer = new ArrayBuffer(this.byteLength);
        } else {
            try {
                this.rawBuffer = new SharedArrayBuffer(this.byteLength);
            } catch (e) {
                this.rawBuffer = new ArrayBuffer(this.byteLength);
            }
        }

        this.partitionMemory(bytesPerFaceAligned, cellsPerFaceRaw, faceMappings);
    }

    private partitionMemory(bytesPerFaceAligned: number, cellsPerFaceRaw: number, faceMappings: any[]) {
        let offset = 0;
        const grid = this.vGrid as any;

        for (const vChunk of grid.chunks) {
            const physicalFaces: Float32Array[] = [];
            const chunkStartOffset = offset;

            for (const face of faceMappings) {
                const numBuffers = face.isPingPong ? 2 : 1;
                for (let i = 0; i < numBuffers; i++) {
                    const view = new Float32Array(this.rawBuffer, offset, cellsPerFaceRaw);
                    physicalFaces.push(view);
                    offset += bytesPerFaceAligned;
                }
            }

            // Correctly advance offset to the start of the next chunk based on ALL allocated slots
            offset = chunkStartOffset + this.totalSlotsPerChunk * bytesPerFaceAligned;

            this.chunkViews.set(vChunk.id, {
                id: vChunk.id,
                faces: physicalFaces
            });
        }
    }

    /**
     * Copy data from GPU back to the CPU ArrayBuffer.
     * WARNING: This is slow and should be used sparingly (e.g. for inspection or CPU visualization).
     */
    public async syncToHost(): Promise<void> {
        if (!this.gpuBuffer) return;
        const size = this.byteLength;
        const staging = HypercubeGPUContext.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            label: 'MasterBuffer Readback Staging'
        });

        const encoder = HypercubeGPUContext.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.gpuBuffer, 0, staging, 0, size);
        HypercubeGPUContext.device.queue.submit([encoder.finish()]);

        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange());
        new Float32Array(this.rawBuffer).set(data);
        staging.unmap();
        staging.destroy();
    }

    /**
     * Read back only specific faces from the GPU.
     * Dramatically reduces VRAM->RAM bandwidth stall.
     */
    public async syncFacesToHost(faceIndices: number[]): Promise<void> {
        if (!this.gpuBuffer) return;
        const copySize = this.strideFace * 4;

        // Create staging buffers for the required faces
        const stagingBuffers = faceIndices.map(() => HypercubeGPUContext.device.createBuffer({
            size: copySize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }));

        const encoder = HypercubeGPUContext.device.createCommandEncoder();

        // Command the GPU to copy each face into its respective staging buffer
        faceIndices.forEach((fIdx, i) => {
            // Note: Since chunks are contiguous, if we had multiple chunks we'd need multiple copies or a sparse readback.
            // For GPU zero-stall, we typically operate on 1 chunk. We'll copy the face block directly.
            // A more robust implementation would iterate chunks. Assuming 1 chunk.
            encoder.copyBufferToBuffer(this.gpuBuffer!, fIdx * copySize, stagingBuffers[i], 0, copySize);
        });

        HypercubeGPUContext.device.queue.submit([encoder.finish()]);

        await Promise.all(stagingBuffers.map(b => b.mapAsync(GPUMapMode.READ)));

        faceIndices.forEach((fIdx, i) => {
            const data = new Float32Array(stagingBuffers[i].getMappedRange());
            const cpuView = new Float32Array(this.rawBuffer, fIdx * copySize, this.strideFace);
            cpuView.set(data);
            stagingBuffers[i].unmap();
            stagingBuffers[i].destroy();
        });
    }

    /**
     * Copy data from the CPU ArrayBuffer to the GPU buffer.
     * Used for initial rasterization or dynamic object injection in GPU mode.
     */
    public syncToDevice(): void {
        if (!this.gpuBuffer) return;
        HypercubeGPUContext.device.queue.writeBuffer(this.gpuBuffer, 0, new Uint8Array(this.rawBuffer as any));
    }

    /**
     * Resolves the physical chunk views for a given ID.
     */
    public getChunkViews(chunkId: string): IPhysicalChunk {
        const views = this.chunkViews.get(chunkId);
        if (!views) {
            throw new Error(`MasterBuffer: Chunk ${chunkId} not partitioned.`);
        }
        return views;
    }

    /**
     * Injects data into a specific face.
     * @param chunkId Target chunk
     * @param faceName Logical face name
     * @param data Typed data to set
     * @param fillAllPingPong If true and face is ping-pong, fills both buffers.
     */
    public setFaceData(chunkId: string, faceName: string, data: Float32Array | number[], fillAllPingPong: boolean = false): void {
        const views = this.getChunkViews(chunkId);
        const dataContract = (this.vGrid as any).dataContract as DataContract;
        const descriptor = dataContract.descriptor;
        const faceIdx = descriptor.faces.findIndex((f: any) => f.name === faceName);

        if (faceIdx === -1) {
            console.error(`MasterBuffer: Face '${faceName}' not found in descriptor.`);
            return;
        }

        const faceMappings = dataContract.getFaceMappings();
        let bufIdx = 0;
        for (let i = 0; i < faceIdx; i++) {
            bufIdx += faceMappings[i].isPingPong ? 2 : 1;
        }

        views.faces[bufIdx].set(data as any);
        if (fillAllPingPong && faceMappings[faceIdx].isPingPong) {
            views.faces[bufIdx + 1].set(data as any);
        }
    }

    /**
     * Initialize all cell populations to equilibrium (rho=1.0, u=0.0).
     * Prevents NaN when compute pass reads ghost cells before first sync.
     */
    public initializeEquilibrium(): void {
        const w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];
        const dataContract = (this.vGrid as any).dataContract as DataContract;
        const faceMappings = dataContract.getFaceMappings();

        for (const chunk of this.vGrid.chunks) {
            const views = this.getChunkViews(chunk.id);
            for (let d = 0; d < 9; d++) {
                const faceName = `f${d}`;
                const faceIdx = faceMappings.findIndex(m => m.name === faceName);
                if (faceIdx === -1) continue;

                let bufIdx = 0;
                for (let i = 0; i < faceIdx; i++) {
                    bufIdx += faceMappings[i].isPingPong ? 2 : 1;
                }

                const faceView = views.faces[bufIdx];
                faceView.fill(w[d]);
                if (faceMappings[faceIdx].isPingPong) {
                    views.faces[bufIdx + 1].fill(w[d]);
                }
            }
        }
        console.log("MasterBuffer: Initialized all LBM faces to equilibrium (rho=1.0).");
    }
}
