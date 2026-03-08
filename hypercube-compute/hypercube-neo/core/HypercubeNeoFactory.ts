import { IFactory } from './IFactory';
import { HypercubeConfig, EngineDescriptor, HypercubeManifest } from './types';
import { VirtualGrid } from './VirtualGrid';
import { MasterBuffer } from './MasterBuffer';
import { NeoEngineProxy } from './NeoEngineProxy';
import { ObjectRasterizer } from './ObjectRasterizer';
import { BoundarySynchronizer } from './BoundarySynchronizer';
import { ParityManager } from './ParityManager';
import { KernelRegistry } from './kernels/KernelRegistry';
import { initializeKernels } from './kernels/KernelInitializer';

// Auto-register default kernels
initializeKernels();

/**
 * High-level factory for Hypercube Neo.
 * Simplifies the orchestration of virtual and physical layers.
 */
export class HypercubeNeoFactory implements IFactory {

    /**
     * Create a virtual representation of the grid.
     */
    public createVirtualGrid(config: HypercubeConfig, descriptor: EngineDescriptor): VirtualGrid {
        return new VirtualGrid(config, descriptor);
    }

    /**
     * Load a self-contained manifest (V4).
     */
    public async fromManifest(url: string): Promise<HypercubeManifest> {
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Failed to load manifest: ${response.statusText}`);
            return await response.json() as HypercubeManifest;
        } catch (e) {
            console.error("Factory: Error loading manifest", e);
            throw e;
        }
    }

    /**
     * Build the full Neo stack and return an orchestration proxy.
     * Respects the 'mode' (cpu/gpu) defined in the configuration.
     */
    public async build(config: HypercubeConfig, descriptor: EngineDescriptor): Promise<NeoEngineProxy> {
        console.log(`Factory: Building engine in ${config.mode.toUpperCase()} mode...`);

        if (config.mode === 'gpu') {
            console.warn("Factory: GPU mode requested but not yet fully implemented. Falling back to CPU.");
            // config.mode = 'cpu'; // Temporary fallback for stability
        }

        // 1. Domain Decomposition (Virtual Layout)
        const vGrid = this.createVirtualGrid(config, descriptor);

        // 2. Memory Allocation (Physical Layout)
        // Note: In the future, MasterBuffer will handle GPU buffer allocation if mode === 'gpu'
        const mBuffer = new MasterBuffer(vGrid);

        // 3. Parity Management
        const parityManager = new ParityManager(vGrid.dataContract);

        // 4. Orchestration Layer
        const rasterizer = new ObjectRasterizer(parityManager);
        const synchronizer = new BoundarySynchronizer();

        return new NeoEngineProxy(
            vGrid,
            mBuffer,
            parityManager,
            rasterizer,
            synchronizer
        );
    }
}
