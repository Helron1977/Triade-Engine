import { IMasterBuffer, IPhysicalChunk, IVirtualGrid } from './GridAbstractions';
import { DataContract } from './DataContract';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

/**
 * The MasterBuffer is the physical memory anchor for Hypercube Neo.
 * It allocates a single contiguous buffer and creates zero-copy views for each chunk.
 */
export class MasterBuffer implements IMasterBuffer {
    public readonly rawBuffer: SharedArrayBuffer | ArrayBuffer;
    public readonly byteLength: number;
    public readonly totalSlotsPerChunk: number;
    public gpuBuffer?: GPUBuffer;
    public strideFace: number = 0; // Number of float32 elements per face (including alignment padding)
    private chunkViews: Map<string, IPhysicalChunk> = new Map();

    constructor(private vGrid: IVirtualGrid) {
        // First pass: calculate alignment-safe total length
        const grid = this.vGrid as any;
        const dataContract = grid.dataContract as DataContract;
        const faceMappings = dataContract.getFaceMappings();

        const nx = Math.floor(grid.config.dimensions.nx / grid.config.chunks.x);
        const ny = Math.floor(grid.config.dimensions.ny / grid.config.chunks.y);
        const nz = Math.floor((grid.config.dimensions.nz || 1) / (grid.config.chunks.z || 1));
        const cellsPerFaceRaw = (nx + 2) * (ny + 2) * (grid.config.dimensions.nz > 1 ? (nz + 2) : 1);

        // WebGPU Alignment: Storage buffer offsets in bind groups MUST be multiples of 256
        const bytesPerFaceRaw = cellsPerFaceRaw * 4;
        const bytesPerFaceAligned = Math.ceil(bytesPerFaceRaw / 256) * 256;
        this.strideFace = bytesPerFaceAligned / 4;

        const facesPerChunk = faceMappings.reduce((acc, f) => acc + (f.isPingPong ? 2 : 1), 0);
        // Standard Neo Slot Allocation:
        // 0-17: f0..f8 (Ping-Pong)
        // 18: obstacles
        // 19: vx
        // 20: vy
        // 21: vorticity
        // 22: smoke
        // We ensure at least 23 slots for NeoAero.wgsl compatibility, but allow more if needed.
        this.totalSlotsPerChunk = Math.max(facesPerChunk, 23);

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
     * Copy data from the CPU ArrayBuffer to the GPU buffer.
     * Used for initial rasterization or dynamic object injection in some modes.
     */
    public syncToDevice(): void {
        if (!this.gpuBuffer) return;
        HypercubeGPUContext.device.queue.writeBuffer(this.gpuBuffer, 0, new Uint8Array(this.rawBuffer as any));
    }

    getChunkViews(chunkId: string): IPhysicalChunk {
        const views = this.chunkViews.get(chunkId);
        if (!views) {
            throw new Error(`MasterBuffer: Chunk ${chunkId} not partitioned.`);
        }
        return views;
    }
}
