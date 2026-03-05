import { HypercubeCpuGrid } from './core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from './core/HypercubeMasterBuffer';
import { EngineRegistry } from './core/EngineRegistry';
import { CanvasAdapter } from './io/CanvasAdapter';
import { WebGpuRenderer } from './io/WebGpuRenderer';
import { HypercubeIsoRenderer } from './utils/HypercubeIsoRenderer';
import { HypercubeGPUContext } from './core/gpu/HypercubeGPUContext';

export interface HypercubeConfig {
    engine: string;
    mode?: 'auto' | 'cpu' | 'gpu';
    resolution: number | { nx: number, ny: number, nz?: number };
    cols?: number;
    rows?: number;
    workers?: boolean;
    workerScript?: string;
    periodic?: boolean;
    params?: Record<string, any>;
}

/**
 * Hypercube V5 - High Level Facade
 * The easiest way to start a simulation.
 */
export class Hypercube {
    /**
     * Creates and initializes a complete simulation grid.
     */
    public static async create(config: HypercubeConfig): Promise<HypercubeCpuGrid> {
        const cols = config.cols ?? 1;
        const rows = config.rows ?? 1;
        const res = typeof config.resolution === 'number' ? config.resolution : config.resolution.nx;

        // Auto-instantiate engine to get metadata
        const tempEngine = EngineRegistry.create(config.engine);
        const numFaces = tempEngine.getRequiredFaces();

        // Calculate dimensions
        let nx = 0, ny = 0, nz = 1;
        if (typeof config.resolution === 'number') {
            nx = ny = config.resolution;
        } else {
            nx = config.resolution.nx;
            ny = config.resolution.ny;
            nz = config.resolution.nz ?? 1;
        }

        // Auto-allocate MasterBuffer
        const totalCellsPerChunk = nx * ny * nz;
        const bytesNeeded = totalCellsPerChunk * numFaces * 4 * cols * rows + 4096;
        const masterBuffer = new HypercubeMasterBuffer(bytesNeeded);

        // Resolve Mode (Auto / CPU / GPU)
        let resolvedMode: 'cpu' | 'gpu' = config.mode === 'gpu' ? 'gpu' : 'cpu';
        if (config.mode === 'auto' || config.mode === 'gpu') {
            const gpuAvailable = await HypercubeGPUContext.init();
            if (gpuAvailable) {
                resolvedMode = 'gpu';
            } else if (config.mode === 'gpu') {
                throw new Error("WebGPU is not supported on this device/browser, but mode was forced to 'gpu'.");
            } else {
                console.info("[Hypercube] WebGPU unavailable. Falling back to CPU mode.");
                resolvedMode = 'cpu';
            }
        }

        // Bootstrap Grid
        const grid = await HypercubeCpuGrid.create(
            cols, rows,
            config.resolution,
            masterBuffer,
            () => EngineRegistry.create(config.engine, config.params),
            numFaces,
            config.periodic ?? true,
            config.workers ?? true,
            config.workerScript,
            resolvedMode
        );

        // Store engine name for auto-rendering
        (grid as any)._engineName = config.engine;
        return grid;
    }

    /**
     * High-level rendering helper. 
     * Automatically chooses the best renderer (2D or Iso) for the given grid.
     */
    public static autoRender(grid: HypercubeCpuGrid, canvas: HTMLCanvasElement, options: any = {}) {
        const engineName = (grid as any)._engineName;
        const tags = EngineRegistry.getTags(engineName);

        if (tags.includes('iso') || tags.includes('2.5d')) {
            // Iso Render
            if (!(grid as any)._renderer) {
                (grid as any)._renderer = new HypercubeIsoRenderer(canvas, undefined, options.scale || 4.0);
            }
            const renderer = (grid as any)._renderer as HypercubeIsoRenderer;
            renderer.clearAndSetup(options.r ?? 10, options.g ?? 20, options.b ?? 30);
            renderer.renderMultiChunkVolume(
                grid.cubes.map(r => r.map(c => c!.faces)),
                grid.nx, grid.ny, grid.cols, grid.rows,
                {
                    densityFaceIndex: options.faceIndex ?? 22,
                    obstacleFaceIndex: options.obstaclesFace ?? 18
                }
            );
        } else if ((grid as any).mode === 'gpu') {
            // Native GPU Direct Render
            if (!(grid as any)._renderer) {
                (grid as any)._renderer = new WebGpuRenderer(canvas);
            }
            const renderer = (grid as any)._renderer as WebGpuRenderer;
            renderer.render(grid, {
                faceIndex: options.faceIndex ?? 0,
                colormap: options.colormap || 'heatmap',
                minVal: options.minVal ?? 0,
                maxVal: options.maxVal ?? 1,
                sliceZ: options.sliceZ ?? Math.floor(grid.nz / 2),
                obstaclesFace: options.obstaclesFace
            });
        } else {
            // CPU Canvas Adapter (2D or 3D slice)
            if (!(grid as any)._renderer) {
                (grid as any)._renderer = new CanvasAdapter(canvas);
            }
            const renderer = (grid as any)._renderer as CanvasAdapter;
            renderer.renderFromFaces(
                grid.cubes.map(r => r.map(c => c!.faces)),
                grid.nx, grid.ny, grid.cols, grid.rows,
                {
                    faceIndex: options.faceIndex ?? 0,
                    colormap: options.colormap || 'heatmap',
                    minVal: options.minVal ?? 0,
                    maxVal: options.maxVal ?? 1,
                    sliceZ: options.sliceZ ?? Math.floor(grid.nz / 2),
                    obstaclesFace: options.obstaclesFace
                }
            );
        }
    }
}
