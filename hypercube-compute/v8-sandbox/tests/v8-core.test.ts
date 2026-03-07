import { describe, it, expect, vi } from 'vitest';
import { ParameterMapper } from '../core/ParameterMapper';
import { V8EngineProxy } from '../core/V8EngineProxy';
import { EngineDescriptor } from '../engines/EngineManifest';

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

    it('should throw error for unknown parameters', () => {
        const mapper = new ParameterMapper(mockDescriptor);
        expect(() => mapper.getOffset('unknown')).toThrow();
    });

    it('should return correct default values array', () => {
        const mapper = new ParameterMapper(mockDescriptor);
        const defaults = mapper.getDefaults();
        expect(defaults[0]).toBe(0.1);
        expect(defaults[1]).toBe(0.5);
    });
});

describe('V8EngineProxy', () => {
    const mockDescriptor: EngineDescriptor = {
        name: 'TestEngine',
        faces: [],
        parameters: [
            { name: 'heat', label: 'Heat', defaultValue: 0.0 },
        ],
        rules: []
    };

    it('should update grid.uniforms correctly', () => {
        const mockGrid = {
            uniforms: new Float32Array(10),
            compute: vi.fn(),
            nx: 16, ny: 16, nz: 1
        };

        const proxy = new V8EngineProxy(mockGrid as any, mockDescriptor);

        // Initial value check
        expect(mockGrid.uniforms[0]).toBe(0.0);

        // Update via semantic name
        proxy.setParam('heat', 1.0);
        expect(mockGrid.uniforms[0]).toBe(1.0);
    });

    it('should call grid.compute when step is called', () => {
        const mockGrid = {
            uniforms: new Float32Array(10),
            compute: vi.fn()
        };
        const proxy = new V8EngineProxy(mockGrid as any, mockDescriptor);
        proxy.compute();
        expect(mockGrid.compute).toHaveBeenCalled();
    });
});
