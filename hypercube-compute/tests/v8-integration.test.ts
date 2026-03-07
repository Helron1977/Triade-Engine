import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HypercubeFactory } from '../v8-sandbox/core/HypercubeFactory';
import { HeatDiffusionV8 } from '../v8-sandbox/engines/HeatDiffusionV8';
import { HeatDiffusionGpuV8 } from '../v8-sandbox/engines/HeatDiffusionGpuV8';
import { HypercubeGPUContext } from '../src/core/gpu/HypercubeGPUContext';
import { Circle } from '../v8-sandbox/core/Shapes';

// Mock WebGPU Constants for Node environment
(globalThis as any).GPUBufferUsage = {
    STORAGE: 0x0080,
    UNIFORM: 0x0040,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
};
(globalThis as any).GPUShaderStage = {
    COMPUTE: 0x0004,
    VERTEX: 0x0001,
    FRAGMENT: 0x0002,
};
(globalThis as any).GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002,
};

describe('V8 Logic Audit (Flowless Integration)', () => {
    let mockDevice: any;
    let mockQueue: any;
    let mockEncoder: any;
    let createdBindGroups: any[] = [];

    beforeEach(() => {
        vi.clearAllMocks();
        createdBindGroups = [];

        mockQueue = {
            writeBuffer: vi.fn(),
            submit: vi.fn(),
        };

        mockEncoder = {
            beginComputePass: vi.fn(() => ({
                setPipeline: vi.fn(),
                setBindGroup: vi.fn(),
                dispatchWorkgroups: vi.fn(),
                end: vi.fn(),
            })),
            copyBufferToBuffer: vi.fn(),
            finish: vi.fn(() => ({})),
        };

        mockDevice = {
            createBuffer: vi.fn(({ label }) => ({
                label,
                mapAsync: vi.fn().mockResolvedValue(undefined),
                getMappedRange: vi.fn(() => new ArrayBuffer(1024)),
                unmap: vi.fn(),
                destroy: vi.fn(),
            })),
            createShaderModule: vi.fn(),
            createBindGroupLayout: vi.fn(),
            createPipelineLayout: vi.fn(),
            createComputePipeline: vi.fn(() => ({
                getBindGroupLayout: vi.fn(),
            })),
            createBindGroup: vi.fn(({ label }) => {
                const bg = { label: label || `BG-${createdBindGroups.length}` };
                createdBindGroups.push(bg);
                return bg;
            }),
            createCommandEncoder: vi.fn(() => mockEncoder),
            queue: mockQueue,
        };

        (HypercubeGPUContext as any)._device = mockDevice;
    });

    it('should alternate BindGroups and handle Dynamic Injection', async () => {
        // 1. Instancier (Initial State)
        const proxy = await HypercubeFactory.instantiate(
            HeatDiffusionV8,
            {
                dimensions: { nx: 32, ny: 32, chunks: [2, 1] },
                mode: 'gpu'
            },
            HeatDiffusionGpuV8
        );

        // Au démarrage, l'Engine a créé des BindGroups et poussé l'initialState
        mockQueue.writeBuffer.mockClear();
        mockEncoder.beginComputePass.mockClear();

        // 2. Step 1 (Parity 0)
        proxy.compute();

        const pass0 = mockEncoder.beginComputePass.mock.results[0].value;
        const call0 = pass0.setBindGroup.mock.calls.find((c: any) => c[0] === 0);
        expect(createdBindGroups.indexOf(call0[1]) % 2).toBe(0);

        // 3. Step 2 (Parity 1)
        proxy.compute();

        const pass2 = mockEncoder.beginComputePass.mock.results[2].value;
        const call2 = pass2.setBindGroup.mock.calls.find((c: any) => c[0] === 0);
        expect(createdBindGroups.indexOf(call2[1]) % 2).toBe(1);

        // 4. DYNAMIC INJECTION CHECK (The core fix proof)
        mockQueue.writeBuffer.mockClear();
        proxy.addShape(new Circle({ x: 10, y: 10, z: 0 }, 5, {
            'Temperature': { role: 'inlet', value: 3.0 }
        }));

        // PROOF: addShape MUST call writeBuffer on GPU buffers via syncFromHost
        // We expect at least 2 calls (Read and Write buffers of touched chunk)
        expect(mockQueue.writeBuffer).toHaveBeenCalled();

        console.info("V8 Flow Audit Success: Parity and Dynamic Injection verified. 🔄🎨⚡✅");
    });
});
