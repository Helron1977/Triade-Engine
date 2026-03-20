import { IFactory } from './IFactory';
import { IBoundarySynchronizer, IMasterBuffer } from './topology/GridAbstractions';
import { HypercubeConfig, EngineDescriptor, HypercubeManifest } from './types';
import { VirtualGrid } from './topology/VirtualGrid';
import { MasterBuffer } from './memory/MasterBuffer';
import { NeoEngineProxy } from './NeoEngineProxy';
import { IDispatcher } from './dispatchers/IDispatcher';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';
import { ObjectRasterizer } from './rasterization/ObjectRasterizer';
import { CpuBoundarySynchronizer } from './topology/CpuBoundarySynchronizer';
import { ParityManager } from './ParityManager';
import { KernelRegistry } from './kernels/KernelRegistry';
import { initializeKernels } from './kernels/KernelInitializer';
import { initializeGpuKernels } from './kernels/GpuKernelInitializer';
import { GpuDispatcher } from './dispatchers/GpuDispatcher';
import { GpuBoundarySynchronizer } from './topology/GpuBoundarySynchronizer';
import { CpuBufferBridge } from './memory/CpuBufferBridge';
import { GpuBufferBridge } from './memory/GpuBufferBridge';
import { IBufferBridge } from './memory/IBufferBridge';

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
     * @throws {Error} If manifest validation fails or initialization occurs out of order.
     */
    public async build(config: HypercubeConfig, descriptor: EngineDescriptor): Promise<NeoEngineProxy> {
        this.validateManifest(config, descriptor);

        console.log(`Factory: Building engine in ${config.mode.toUpperCase()} mode...`);

        // 1. Virtual & Physical Layout
        const vGrid = this.createVirtualGrid(config, descriptor);

        if (config.mode === 'gpu') {
            await HypercubeGPUContext.init();
        }

        const mBuffer = new MasterBuffer(vGrid);
        const parityManager = new ParityManager(vGrid.dataContract);

        // 2. Abstraction Layer (Bridge)
        const bridge = this.createBridge(config, mBuffer);

        // 3. Orchestration Components
        const rasterizer = new ObjectRasterizer(parityManager);
        const synchronizer = this.createSynchronizer(config);
        const dispatcher = await this.createDispatcher(config, vGrid, bridge, parityManager);

        const engine = new NeoEngineProxy(
            vGrid,
            bridge,
            parityManager,
            rasterizer,
            synchronizer,
            dispatcher
        );

        await engine.init();
        return engine;
    }

    /**
     * Creates the appropriate memory bridge based on the execution mode.
     */
    private createBridge(config: HypercubeConfig, mBuffer: IMasterBuffer): IBufferBridge {
        return config.mode === 'gpu'
            ? new GpuBufferBridge(mBuffer)
            : new CpuBufferBridge(mBuffer);
    }

    /**
     * Creates the appropriate topological synchronizer.
     */
    private createSynchronizer(config: HypercubeConfig): IBoundarySynchronizer {
        return config.mode === 'gpu'
            ? new GpuBoundarySynchronizer()
            : new CpuBoundarySynchronizer();
    }

    /**
     * Creates the appropriate numerical dispatcher (async for dynamic imports).
     */
    private async createDispatcher(
        config: HypercubeConfig,
        vGrid: VirtualGrid,
        bridge: IBufferBridge,
        parityManager: ParityManager
    ): Promise<IDispatcher> {
        if (config.mode === 'gpu') {
            return new GpuDispatcher(vGrid, bridge as GpuBufferBridge, parityManager);
        }

        if (config.executionMode === 'parallel') {
            const { ParallelDispatcher } = await import('./dispatchers/ParallelDispatcher');
            return new ParallelDispatcher(vGrid, bridge, parityManager);
        }

        const { NumericalDispatcher } = await import('./dispatchers/NumericalDispatcher');
        return new NumericalDispatcher(vGrid, bridge, parityManager);
    }

    /**
     * Rigorous validation of the engine configuration.
     */
    private validateManifest(config: HypercubeConfig, descriptor: EngineDescriptor) {
        if (!config.dimensions || config.dimensions.nx <= 0 || config.dimensions.ny <= 0) {
            throw new Error("Validation Error: Dimensions nx and ny must be positive integers.");
        }

        if (config.mode === 'gpu') {
            const isPow2 = (n: number) => (n & (n - 1)) === 0;
            if (!isPow2(config.dimensions.nx) || !isPow2(config.dimensions.ny)) {
                console.warn("GPU Performance Warning: Dimensions are not powers of 2. This may cause Bank Conflicts.");
            }
        }

        if (!descriptor.rules || descriptor.rules.length === 0) {
            throw new Error("Validation Error: Engine must define at least one compute rule (rules[]).");
        }

        if (!descriptor.faces || descriptor.faces.length === 0) {
            throw new Error("Validation Error: Engine must define at least one data face.");
        }

        // Kernel-specific rules validation
        for (const rule of (descriptor.rules || [])) {
            if (rule.type === 'lbm-d2q9' || rule.type === 'neo-ocean-v1') {
                // Check if it has a source face and if it's marked as synchronized
                const populations = descriptor.faces.find(f => f.name === rule.source);
                if (!populations) {
                    throw new Error(`Validation Error: Kernel '${rule.type}' requires source face '${rule.source}' which is missing.`);
                }
            }
        }

        console.log("Factory: Manifest validated successfully.");
    }
}
