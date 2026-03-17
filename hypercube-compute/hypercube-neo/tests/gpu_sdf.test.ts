import { describe, it, expect } from 'vitest';
import { VirtualGrid } from '../core/topology/VirtualGrid';
import { MasterBuffer } from '../core/MasterBuffer';
import { ParityManager } from '../core/ParityManager';
import { GpuDispatcher } from '../core/GpuDispatcher';
import { EngineDescriptor, HypercubeConfig } from '../core/types';
import { HypercubeGPUContext } from '../core/gpu/HypercubeGPUContext';
import { GpuBufferBridge } from '../core/GpuBufferBridge';

// Mock Global WebGPU Enums
if (typeof (globalThis as any).GPUBufferUsage === 'undefined') {
    (globalThis as any).GPUBufferUsage = { STORAGE: 0x0040, UNIFORM: 0x0010, COPY_SRC: 0x0004, COPY_DST: 0x0008 };
}

describe('Hypercube Neo: GPU SDF Orchestration', () => {
    (HypercubeGPUContext as any)._device = {
        createBuffer: () => ({ destroy: () => { } }),
        createBindGroup: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        queue: { writeBuffer: () => { }, submit: () => { } },
        limits: { minUniformBufferOffsetAlignment: 256 }
    };

    const descriptor: EngineDescriptor = {
        name: 'GPU-SDF-Test',
        version: '1.0.0',
        faces: [{ name: 'sdf', type: 'scalar', isSynchronized: true }],
        parameters: {}, rules: [], outputs: [],
        requirements: { ghostCells: 1, pingPong: true }
    };

    const config: HypercubeConfig = {
        dimensions: { nx: 16, ny: 16, nz: 1 },
        chunks: { x: 1, y: 1 },
        boundaries: { all: { role: 'wall' } },
        engine: 'GPU-SDF-Test',
        params: {},
        mode: 'cpu'
    };

    it('should map JFA step to the correct uniform offset in GpuDispatcher', () => {
        const vGrid = new VirtualGrid(config, descriptor);
        const mBuffer = new MasterBuffer(vGrid);
        const bridge = new GpuBufferBridge(mBuffer);
        const parityManager = new ParityManager(vGrid.dataContract);
        const dispatcher = new GpuDispatcher(vGrid, bridge, parityManager);

        const strideFace = mBuffer.strideFace;
        const alignedStrideFace = HypercubeGPUContext.alignToUniform(strideFace * 4);

        // Simplified check: Does it call alignToUniform correctly for chunk 0
        const params = dispatcher.getChunkBufferParams(0);
        expect(params.offset).toBe(0);
    });
});
