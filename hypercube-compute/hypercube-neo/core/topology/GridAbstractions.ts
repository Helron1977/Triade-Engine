import { BoundaryRole, GridBoundaries, Dimension3D, HypercubeConfig, VirtualObject } from '../types';
import { IBufferBridge } from '../IBufferBridge';

/**
 * Descriptor for a joint between two chunks or a world boundary.
 */
export interface JointDescriptor {
    role: BoundaryRole;
    neighborId?: string; // ID of the connected chunk, if internal
    face: 'left' | 'right' | 'top' | 'bottom' | 'front' | 'back';
}

/**
 * A Virtual Chunk is a declarative slice of the global grid.
 */
export interface VirtualChunk {
    x: number;
    y: number;
    z: number;
    id: string;
    joints: JointDescriptor[];
    localDimensions: Dimension3D; // The actual size of this chunk (handles remainders)
}

/**
 * The MapConstructor is responsible for arranging chunks and deducing their joints.
 */
export interface IMapConstructor {
    /**
     * Generate the virtual layout of the grid.
     * Automatically sets roles to 'joint' for internal connections.
     * External boundaries are mapped from the global config.
     */
    buildMap(
        dims: Dimension3D,
        chunks: { x: number; y: number; z?: number },
        globalBoundaries: GridBoundaries
    ): VirtualChunk[];
}

/**
 * Holds the virtual representation of the entire grid.
 */
export interface IVirtualGrid {
    readonly dimensions: Dimension3D;
    readonly chunkLayout: { x: number; y: number; z: number };
    readonly chunks: VirtualChunk[];
    findChunkAt(x: number, y: number, z?: number): VirtualChunk | undefined;
    getObjectsInChunk(chunk: VirtualChunk, t?: number): VirtualObject[];
    getTotalMemoryRequirement(): number;
}

/**
 * A Physical Chunk is a set of memory views (Float32Array) for a chunk.
 */
export interface IPhysicalChunk {
    readonly id: string;
    readonly faces: Float32Array[]; // One view per EngineFace
}

/**
 * The MasterBuffer manages the physical allocation and segmenting of memory.
 */
export interface IMasterBuffer {
    readonly byteLength: number;
    readonly totalSlotsPerChunk: number;
    readonly strideFace: number;
    readonly rawBuffer: SharedArrayBuffer | ArrayBuffer;
    readonly gpuBuffer?: any; // GPUBuffer if mode is 'gpu'

    /**
     * Get the set of memory views for a specific chunk.
     */
    getChunkViews(chunkId: string): IPhysicalChunk;

    /**
     * Copy data from CPU to GPU.
     */
    syncToDevice(): void;

    /**
     * Direct buffer injection for a specific face.
     * Useful for importing high-resolution topology maps (heightmaps, obstacles).
     */
    setFaceData(chunkId: string, faceName: string, data: Float32Array | number[], fillAllPingPong?: boolean): void;

    /**
     * Copy data from GPU back to the CPU ArrayBuffer.
     */
    syncToHost(): Promise<void>;

    /**
     * Copy specific faces from GPU back to the CPU ArrayBuffer.
     */
    syncFacesToHost(faceIndices: number[]): Promise<void>;

    /**
     * Initialize all cell populations to equilibrium.
     */
    initializeEquilibrium(): void;
}

/**
 * Handles the synchronization of ghost cells between chunks.
 */
export interface IBoundarySynchronizer {
    /**
     * Must handle faces, edges, and corners for 2D/3D.
     */
    syncAll(vGrid: IVirtualGrid, bridge: IBufferBridge, parityManager?: any, mode?: 'read' | 'write'): void;
}

/**
 * Handles baking VirtualObjects into the physical memory.
 */
export interface IRasterizer {
    /**
     * Rasterize relevant objects into a specific chunk's memory.
     * @param t Current simulation time (for path/animation evaluation)
     */
    rasterizeChunk(
        vChunk: VirtualChunk,
        vGrid: IVirtualGrid,
        bridge: IBufferBridge,
        t: number,
        parityTarget?: 'read' | 'write'
    ): void;
}
