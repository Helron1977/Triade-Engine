import { describe, it, expect, vi } from 'vitest';
import { GpuDispatcher } from '../core/dispatchers/GpuDispatcher';
import { HypercubeGPUContext } from '../core/gpu/HypercubeGPUContext';
import { GpuKernelRegistry } from '../core/kernels/GpuKernelRegistry';

// Mock WebGPU Globals
(globalThis as any).GPUBufferUsage = {
    UNIFORM: 0x01,
    STORAGE: 0x02,
    COPY_DST: 0x04,
    COPY_SRC: 0x08
};

describe('GpuDispatcher Layout Validation', () => {
    it('should align with NeoTensor.wgsl Uniforms struct', async () => {
        // 1. Mock minimal environment
        const mockDevice = {
            createBuffer: vi.fn().mockReturnValue({ size: 1024, usage: 7, label: 'mock' }),
            createShaderModule: vi.fn(),
            createComputePipeline: vi.fn().mockReturnValue({ 
                getBindGroupLayout: vi.fn().mockReturnValue({}),
                label: 'mock-pipeline'
            }),
            createBindGroup: vi.fn().mockReturnValue({}),
            createCommandEncoder: vi.fn().mockReturnValue({
                beginComputePass: vi.fn().mockReturnValue({
                    setPipeline: vi.fn(),
                    setBindGroup: vi.fn(),
                    dispatchWorkgroups: vi.fn(),
                    end: vi.fn()
                }),
                finish: vi.fn()
            }),
            queue: {
                writeBuffer: vi.fn(),
                submit: vi.fn()
            },
            limits: { minUniformBufferOffsetAlignment: 256 }
        };

        vi.spyOn(HypercubeGPUContext, 'device', 'get').mockReturnValue(mockDevice as any);
        vi.spyOn(HypercubeGPUContext, 'isInitialized', 'get').mockReturnValue(true);
        vi.spyOn(GpuKernelRegistry, 'getSource').mockReturnValue('@compute @workgroup_size(1,1) fn main() {}');
        vi.spyOn(GpuKernelRegistry, 'getMetadata').mockReturnValue({ roles: {}, uniformObjectOffset: 32 } as any);

        const mockDataContract = {
            descriptor: {
                faces: [
                    { name: 'mode_a', isSynchronized: true },
                    { name: 'mode_b', isSynchronized: true },
                    { name: 'mode_c', isSynchronized: true },
                    { name: 'target', isSynchronized: true },
                    { name: 'reconstruction', isSynchronized: true }
                ],
                rules: [
                    { type: 'neo-tensor-cp-v1', params: { rank: 12, regularization: 0.07 } }
                ]
            },
            getFaceMappings: () => [
                { name: 'mode_a', isPingPong: false },
                { name: 'mode_b', isPingPong: false },
                { name: 'mode_c', isPingPong: false },
                { name: 'target', isPingPong: false },
                { name: 'reconstruction', isPingPong: false }
            ]
        };

        const mockGrid = {
            dimensions: { nx: 32, ny: 32, nz: 8 },
            chunkLayout: { x: 1, y: 1 },
            chunks: [{ id: 'c0', x: 0, y: 0, z: 0, localDimensions: { nx: 32, ny: 32, nz: 8 }, joints: [] }],
            dataContract: mockDataContract,
            config: { boundaries: { all: { role: 'wall' } }, objects: [] }
        };

        const mockBridge = {
            gpuBuffer: {},
            strideFace: 1024,
            totalSlotsPerChunk: 5,
            commit: vi.fn()
        };

        const mockParity = {
            getFaceIndices: (name: string) => ({ read: 0, write: 0 }),
            currentTick: 42
        };

        const dispatcher = new GpuDispatcher(mockGrid as any, mockBridge as any, mockParity as any);
        
        // Exercise dispatch to trigger buffer filling
        await dispatcher.dispatch(1.0);

        // 2. Capture the data written to the uniform buffer
        const lastWrite = mockDevice.queue.writeBuffer.mock.calls[0];
        const writtenData = lastWrite[2] as Uint8Array;
        const u32 = new Uint32Array(writtenData.buffer);
        const f32 = new Float32Array(writtenData.buffer);

        // 3. SECURE THE LOCKS (Matches NeoTensor.wgsl)
        
        // Dimensions (u32 in GpuDispatcher/WGSL now)
        expect(u32[0]).toBe(32); // nx
        expect(u32[1]).toBe(32); // ny
        expect(u32[2]).toBe(8);  // nz
        
        // Tensor Parameters
        expect(f32[3]).toBe(12); // rank (mock had 12)
        expect(f32[4]).toBeCloseTo(0.07, 6); // regularization
        
        // Face Indices (5..9)
        expect(u32[5]).toBe(0); // mode_a
        expect(u32[8]).toBe(0); // target
        expect(u32[9]).toBe(0); // reconstruction
        
        // Hardware/System info (10, 11)
        expect(u32[10]).toBe(1024); // strideFace
        expect(u32[11]).toBe(42);   // currentTick
    });
});
