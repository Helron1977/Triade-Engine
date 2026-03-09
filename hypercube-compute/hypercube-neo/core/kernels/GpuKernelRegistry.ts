import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';

/**
 * Registry for WebGPU compute kernels in Neo.
 * Maps scheme types to WGSL source or ready-to-use pipelines.
 */
export class GpuKernelRegistry {
    private static cache: Map<string, string> = new Map();

    public static async register(type: string, wgslUrl: string) {
        const response = await fetch(wgslUrl);
        const source = await response.text();
        this.cache.set(type, source);
    }

    public static setSource(type: string, source: string) {
        this.cache.set(type, source);
    }

    public static getSource(type: string): string {
        const source = this.cache.get(type);
        if (!source) throw new Error(`GpuKernelRegistry: No WGSL source for type "${type}"`);
        return source;
    }
}
