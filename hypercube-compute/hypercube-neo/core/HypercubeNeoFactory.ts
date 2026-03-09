import { IFactory } from './IFactory';
import { IBoundarySynchronizer } from './GridAbstractions';
import { HypercubeConfig, EngineDescriptor, HypercubeManifest } from './types';
import { VirtualGrid } from './VirtualGrid';
import { MasterBuffer } from './MasterBuffer';
import { NeoEngineProxy } from './NeoEngineProxy';
import { IDispatcher } from './IDispatcher';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';
import { ObjectRasterizer } from './ObjectRasterizer';
import { BoundarySynchronizer } from './BoundarySynchronizer';
import { ParityManager } from './ParityManager';
import { KernelRegistry } from './kernels/KernelRegistry';
import { initializeKernels } from './kernels/KernelInitializer';
import { initializeGpuKernels } from './kernels/GpuKernelInitializer';
import { GpuDispatcher } from './GpuDispatcher';
import { GpuBoundarySynchronizer } from './GpuBoundarySynchronizer';

// Auto-register default kernels
initializeKernels();
initializeGpuKernels();

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

        // 1. Domain Decomposition (Virtual Layout)
        const vGrid = this.createVirtualGrid(config, descriptor);

        // 2. GPU Initialization (if needed)
        if (config.mode === 'gpu') {
            await HypercubeGPUContext.init();
        }

        // 3. Memory Allocation (Physical Layout)
        const mBuffer = new MasterBuffer(vGrid);

        // 4. Parity Management
        const parityManager = new ParityManager(vGrid.dataContract);

        // 5. Orchestration Layer
        const rasterizer = new ObjectRasterizer(parityManager);
        let synchronizer: IBoundarySynchronizer;
        let dispatcher: IDispatcher;

        if (config.mode === 'gpu') {
            synchronizer = new GpuBoundarySynchronizer();
            dispatcher = new GpuDispatcher(vGrid, mBuffer, parityManager);
        } else {
            synchronizer = new BoundarySynchronizer();
            dispatcher = config.executionMode === 'parallel'
                ? new (await import('./ParallelDispatcher')).ParallelDispatcher(vGrid, mBuffer, parityManager)
                : new (await import('./NumericalDispatcher')).NumericalDispatcher(vGrid, mBuffer, parityManager);
        }

        const engine = new NeoEngineProxy(
            vGrid,
            mBuffer,
            parityManager,
            rasterizer,
            synchronizer,
            dispatcher
        );

        await engine.init();
        return engine;
    }
}
