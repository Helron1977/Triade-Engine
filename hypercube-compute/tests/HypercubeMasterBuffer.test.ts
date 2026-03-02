import { describe, it, expect } from 'vitest';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';

describe('HypercubeMasterBuffer', () => {
    it('should allocate without throwing if within memory limits', () => {
        const master = new HypercubeMasterBuffer(10 * 1024 * 1024); // 10MB
        expect(() => master.allocateCube(100, 6)).not.toThrow();
    });

    it('should throw out of memory if limits exceeded', () => {
        const master = new HypercubeMasterBuffer(1024 * 1024); // 1MB
        // 1000x1000 * 4 bytes * 6 faces = 24MB, should throw
        expect(() => master.allocateCube(1000, 6)).toThrow(/Out Of Memory/);
    });

    it('should correctly align allocations to 256 bytes per face for optimal WebGPU stride', () => {
        const master = new HypercubeMasterBuffer(10 * 1024 * 1024);

        // 10x10 map = 100 floats = 400 bytes per face raw.
        // It should be padded to next multiple of 256 -> 512 bytes per face.
        const alloc = master.allocateCube(10, 1);

        expect(alloc.stride).toBe(512);

        // The first offset should be 0 or aligned to 256.
        expect(alloc.offset % 256).toBe(0);

        // Next allocation
        const alloc2 = master.allocateCube(10, 1);
        expect(alloc2.offset).toBe(alloc.offset + 512);
        expect(alloc2.offset % 256).toBe(0);
    });

    it('should report used memory correctly', () => {
        const master = new HypercubeMasterBuffer(10 * 1024 * 1024);
        master.allocateCube(100, 6);
        // 100x100 * 4 = 40,000 bytes raw -> 40,192 bytes aligned (157 * 256) per face.
        // 6 faces * 40,192 = 241,152 bytes.
        // 241,152 / (1024*1024) = 0.23 MB
        expect(master.getUsedMemoryInMB()).toBe('0.23 MB');
    });
});
