import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ParameterMapper } from '../v8-sandbox/core/ParameterMapper';
import { V8EngineProxy } from '../v8-sandbox/core/V8EngineProxy';
import { EngineDescriptor } from '../v8-sandbox/engines/EngineManifest';
import { V8_METADATA_OFFSET, V8_PARAMS_OFFSET, V8_PARAM_STRIDE } from '../v8-sandbox/core/UniformPresets';
import { HypercubeFactory } from '../v8-sandbox/core/HypercubeFactory';
import { HypercubeCpuGrid } from '../v8-sandbox/core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from '../v8-sandbox/core/HypercubeMasterBuffer';
import { HypercubeGPUContext } from '../src/core/gpu/HypercubeGPUContext';

// Mock WebGPU Constants for Node environment
(globalThis as any).GPUBufferUsage = {
    STORAGE: 0x0080,
    UNIFORM: 0x0040,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
};
(globalThis as any).GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002,
};

describe('V8 Core Tests', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('V8 ParameterMapper', () => {
        const mockDescriptor: EngineDescriptor = {
            name: 'TestEngine',
            faces: [],
            parameters: [
                { name: 'viscosity', label: 'Viscosity', defaultValue: 0.1 },
                { name: 'diffusion', label: 'Diffusion', defaultValue: 0.5 },
            ],
            rules: []
        };

        it('should resolve semantic names to correct offsets', () => {
            const mapper = new ParameterMapper(mockDescriptor);
            expect(mapper.getOffset('viscosity')).toBe(0);
            expect(mapper.getOffset('diffusion')).toBe(1);
        });

        it('should return correct default values array', () => {
            const mapper = new ParameterMapper(mockDescriptor);
            const defaults = mapper.getDefaults();
            expect(defaults[0]).toBeCloseTo(0.1);
            expect(defaults[1]).toBeCloseTo(0.5);
        });
    });

    describe('V8EngineProxy', () => {
        const mockDescriptor: EngineDescriptor = {
            name: 'ComplexEngine',
            faces: [],
            parameters: [
                { name: 'p0', label: 'P0', defaultValue: 10 },
                { name: 'p1', label: 'P1', defaultValue: 20 },
                { name: 'p2', label: 'P2', defaultValue: 30 },
            ],
            rules: []
        };

        it('should update grid.uniforms correctly with base-8 offset and multi-params', () => {
            const mockGrid = {
                uniforms: new Float32Array(512),
                compute: vi.fn()
            };
            const mockEngine = { parity: 0 };
            const proxy = new V8EngineProxy(mockGrid as any, mockDescriptor, mockEngine as any);

            // Defaults should be injected at base 8
            expect(mockGrid.uniforms[V8_PARAMS_OFFSET + 0]).toBeCloseTo(10);
            expect(mockGrid.uniforms[V8_PARAMS_OFFSET + 4]).toBeCloseTo(20);
            expect(mockGrid.uniforms[V8_PARAMS_OFFSET + 8]).toBeCloseTo(30);

            proxy.setParam('p1', 99);
            expect(mockGrid.uniforms[V8_PARAMS_OFFSET + 4]).toBe(99);
        });
    });

    describe('HypercubeFactory', () => {
        it('should allocate memory without throwing OOM', async () => {
            const mockDescriptor: EngineDescriptor = {
                name: 'OOM-Test',
                faces: [{ name: 'F1', type: 'scalar' }, { name: 'F2', type: 'scalar' }],
                parameters: [],
                rules: []
            };

            const proxy = await HypercubeFactory.instantiate(mockDescriptor, {
                dimensions: { nx: 32, ny: 32, chunks: [2, 2] },
                mode: 'cpu'
            });

            expect(proxy).toBeDefined();
            expect(proxy.grid.cols).toBe(2);
        });

        it('should throw explicit error if GPU requested but not initialized', async () => {
            const mockDescriptor: EngineDescriptor = {
                name: 'GPU-Safety-Test',
                faces: [], parameters: [], rules: []
            };

            vi.spyOn(HypercubeGPUContext, 'isInitialized', 'get').mockReturnValue(false);

            await expect(HypercubeFactory.instantiate(mockDescriptor, {
                dimensions: { nx: 16, ny: 16, chunks: [1, 1] },
                mode: 'gpu'
            })).rejects.toThrow();
        });

        it('should call grid.pushToGPU if mode is gpu (Regression V8)', async () => {
            vi.spyOn(HypercubeGPUContext, 'isInitialized', 'get').mockReturnValue(true);

            const mockGrid = {
                pushToGPU: vi.fn(),
                cubes: [[{ engine: { name: 'Test' } }]],
                nx: 16, ny: 16,
                boundaryConfig: {},
                uniforms: new Float32Array(512)
            };

            vi.spyOn(HypercubeCpuGrid, 'create').mockResolvedValue(mockGrid as any);

            const mockDescriptor: EngineDescriptor = {
                name: 'PushGPU-Test',
                faces: [], parameters: [], rules: []
            };

            await HypercubeFactory.instantiate(mockDescriptor, {
                dimensions: { nx: 16, ny: 16, chunks: [1, 1] },
                mode: 'gpu'
            });

            expect(mockGrid.pushToGPU).toHaveBeenCalled();
        });
    });

    describe('HypercubeCpuGrid (V8 Specifics)', () => {
        it('should correctly set metadata uniforms (nx, ny, offX, offY)', async () => {
            const writeSpy = vi.fn();
            vi.spyOn(HypercubeGPUContext, 'device', 'get').mockReturnValue({
                queue: { writeBuffer: writeSpy, submit: vi.fn() },
                createCommandEncoder: () => ({
                    beginComputePass: () => ({
                        setPipeline: vi.fn(), setBindGroup: vi.fn(),
                        dispatchWorkgroups: vi.fn(), end: vi.fn()
                    }),
                    finish: () => ({}),
                    copyBufferToBuffer: vi.fn()
                }),
                createBuffer: () => ({ size: 1024, unmap: vi.fn() })
            } as any);

            const grid = new HypercubeCpuGrid(
                2, 1, { nx: 256, ny: 128 },
                new HypercubeMasterBuffer(1024 * 1024),
                () => ({ getRequiredFaces: () => 1, init: () => { }, computeGPU: vi.fn(), name: 'Test' } as any),
                1, false, false, 'gpu'
            );

            await grid.compute();

            // Check metadata for last chunk (1,0)
            expect(grid.uniforms[0]).toBe(256); // NX
            expect(grid.uniforms[1]).toBe(128); // NY
            expect(grid.uniforms[6]).toBe(254); // offX (256-2 * 1)
        });
    });
});
