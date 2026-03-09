import { describe, it, expect } from 'vitest';
import { VirtualGrid } from '../core/VirtualGrid';
import { MasterBuffer } from '../core/MasterBuffer';
import { ParityManager } from '../core/ParityManager';
import { GpuDispatcher } from '../core/GpuDispatcher';
import { EngineDescriptor, HypercubeConfig } from '../core/types';
import { DataContract } from '../core/DataContract';
import { HypercubeGPUContext } from '../core/gpu/HypercubeGPUContext';

// Mock Global WebGPU Enums for NodeJS test environment
if (typeof (global as any).GPUBufferUsage === 'undefined') {
    (global as any).GPUBufferUsage = {
        STORAGE: 0x0040,
        UNIFORM: 0x0010,
        COPY_SRC: 0x0004,
        COPY_DST: 0x0008,
        MAP_READ: 0x0001,
        MAP_WRITE: 0x0002
    };
}
if (typeof (global as any).GPUMapMode === 'undefined') {
    (global as any).GPUMapMode = {
        READ: 0x0001,
        WRITE: 0x0002
    };
}

describe('Hypercube Neo: GPU Orchestration Logic', () => {
    // Mock GPU Device for orchestration testing (Neo mode)
    (HypercubeGPUContext as any)._device = {
        createBuffer: () => ({ destroy: () => { } }),
        createBindGroup: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        queue: { writeBuffer: () => { }, submit: () => { } }
    };

    const descriptor: EngineDescriptor = {
        name: 'GPU-Orchestration-Test',
        version: '1.0.0',
        faces: [
            { name: 'f1', type: 'scalar', isSynchronized: true },
            { name: 'obs', type: 'mask', isSynchronized: true, isReadOnly: true }
        ],
        parameters: {},
        rules: [],
        outputs: [],
        requirements: { ghostCells: 1, pingPong: true }
    };

    const config: HypercubeConfig = {
        dimensions: { nx: 16, ny: 16, nz: 1 },
        chunks: { x: 2, y: 1 },
        boundaries: { all: { role: 'wall' } },
        engine: 'GPU-Orchestration-Test',
        params: {},
        mode: 'cpu' // CPU mode allows MasterBuffer to init without a real GPU, but we test GpuDispatcher logic
    };

    it('should calculate correct chunk offsets in GpuDispatcher', () => {
        const vGrid = new VirtualGrid(config, descriptor);
        const mBuffer = new MasterBuffer(vGrid);
        const parityManager = new ParityManager(vGrid.dataContract);
        const dispatcher = new GpuDispatcher(vGrid, mBuffer, parityManager);

        // Access public method for testing bind group params
        const bg0 = dispatcher.getChunkBufferParams(0); // chunk 0
        const bg1 = dispatcher.getChunkBufferParams(1); // chunk 1

        const strideFace = mBuffer.strideFace;
        const totalSlots = mBuffer.totalSlotsPerChunk;

        // Chunk 0 offset should be 0
        expect(bg0.offset).toBe(0);
        expect(bg0.size).toBe(totalSlots * strideFace * 4);

        // Chunk 1 offset should be chunk0's size
        expect(bg1.offset).toBe(totalSlots * strideFace * 4);
        expect(bg1.size).toBe(totalSlots * strideFace * 4);
    });

    it('should correctly handle parity in ParityManager', () => {
        const dataContract = new DataContract(descriptor);
        const parityManager = new ParityManager(dataContract);

        const f1Indices = parityManager.getFaceIndices('f1');
        const obsIndices = parityManager.getFaceIndices('obs');

        // Initial setup (tick 0)
        expect(parityManager.currentTick).toBe(0);
        expect(f1Indices.read).toBe(0);
        expect(f1Indices.write).toBe(1);

        // Static faces (obs) should have same read/write indices
        // f1_0=0, f1_1=1, obs=2
        expect(obsIndices.read).toBe(2);
        expect(obsIndices.write).toBe(2);

        // Advance tick
        parityManager.nextTick();
        const f1IndicesUpdated = parityManager.getFaceIndices('f1');

        expect(parityManager.currentTick).toBe(1);
        expect(f1IndicesUpdated.read).toBe(1);
        expect(f1IndicesUpdated.write).toBe(0);

        // Static face remains the same
        expect(parityManager.getFaceIndices('obs').read).toBe(2);
    });
});
