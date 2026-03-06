import { HypercubeCpuGrid } from './core/HypercubeCpuGrid';
import { HypercubeMasterBuffer } from './core/HypercubeMasterBuffer';
import { EngineRegistry } from './core/EngineRegistry';
import { CanvasAdapter } from './io/CanvasAdapter';
import { WebGpuRenderer } from './io/WebGpuRenderer';
import { HypercubeIsoRenderer } from './utils/HypercubeIsoRenderer';
import { HypercubeGPUContext } from './core/gpu/HypercubeGPUContext';
import { HypercubeChunk } from './core/HypercubeChunk';
import { HypercubeGpuVolumeRenderer } from './io/HypercubeGpuVolumeRenderer';
import type { IHypercubeEngine, VisualProfile } from './engines/IHypercubeEngine';
import { VisualRegistry } from './io/VisualRegistry';

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
    private masterBuffer: HypercubeMasterBuffer;

    constructor(initialMemoryMB: number = 100) {
        this.masterBuffer = new HypercubeMasterBuffer(initialMemoryMB * 1024 * 1024);
    }

    /**
     * Legacy helper for tests. Creates a single chunk grid.
     */
    public createCube(
        id: string,
        res: { nx: number, ny: number, nz?: number },
        engine: IHypercubeEngine,
        numFaces: number = 6
    ): HypercubeChunk {
        const nx = res.nx;
        const ny = res.ny;
        const nz = res.nz ?? 1;
        const chunk = new HypercubeChunk(0, 0, nx, ny, nz, this.masterBuffer, numFaces);
        chunk.setEngine(engine);
        engine.init(chunk.faces, nx, ny, nz, false);
        return chunk;
    }

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
     * @description Orchestre le rendu automatique en se basant sur le contrat visuel de l'Engine.
     */
    public static autoRender(grid: HypercubeCpuGrid, canvas: HTMLCanvasElement, options: any = {}) {
        const firstChunk = grid.cubes[0][0];
        const engine = firstChunk?.engine;
        if (!engine) return;

        // 1. Récupération du profil (Nouveau contrat V7) ou Fallback (Legacy tags)
        const rawProfile = (engine as any).getVisualProfile ? (engine as any).getVisualProfile() : (Hypercube as any).getFallbackProfile(engine);
        const profile = VisualRegistry.resolve(rawProfile as VisualProfile);
        const schema = (engine as any).getSchema ? (engine as any).getSchema() : { faces: [] };

        const resolveFace = (layer: any) => {
            if (layer.faceIndex !== undefined) return layer.faceIndex;
            if (layer.faceLabel) {
                const found = schema.faces.find((f: any) => f.label === layer.faceLabel);
                if (found) return found.index;
            }
            return 0;
        };

        const layers = profile.layers || [];
        const primaryLayer = layers.find((l: any) => l.role === 'primary') || layers[0] || { role: 'primary' };
        const obstacleLayer = layers.find((l: any) => l.role === 'obstacle');
        const vorticityLayer = layers.find((l: any) => l.role === 'vorticity');

        const isIso = profile.defaultMode === 'isometric' || profile.defaultMode === '2.5d' || options.mode === 'isometric';

        if (isIso && grid.mode === 'cpu') {
            // Iso Render
            if (!(grid as any)._renderer || !((grid as any)._renderer instanceof HypercubeIsoRenderer)) {
                (grid as any)._renderer = new HypercubeIsoRenderer(canvas, undefined, options.scale || 4.0);
            }
            const renderer = (grid as any)._renderer as HypercubeIsoRenderer;
            renderer.clearAndSetup(options.r ?? 10, options.g ?? 20, options.b ?? 35);
            renderer.renderMultiChunkVolume(
                grid.cubes.map(r => r.map(c => c!.faces)),
                grid.nx, grid.ny, grid.cols, grid.rows,
                {
                    densityFaceIndex: options.faceIndex ?? resolveFace(primaryLayer),
                    obstacleFaceIndex: options.obstaclesFace ?? (obstacleLayer ? resolveFace(obstacleLayer) : undefined)
                }
            );
        } else if (grid.mode === 'gpu') {
            // Native GPU Direct Render
            if (!(grid as any)._gpuRenderer) {
                (grid as any)._gpuRenderer = new HypercubeGpuVolumeRenderer(canvas);
            }
            const renderer = (grid as any)._gpuRenderer as HypercubeGpuVolumeRenderer;
            renderer.render(grid, {
                faceIndex: options.faceIndex ?? resolveFace(primaryLayer),
                obstacleFaceIndex: options.obstaclesFace ?? (obstacleLayer ? resolveFace(obstacleLayer) : undefined),
                vorticityFace: options.vorticityFace ?? (vorticityLayer ? resolveFace(vorticityLayer) : undefined),
                colormap: options.colormap || primaryLayer.colormap || 'heatmap',
                minVal: options.minVal ?? primaryLayer.range?.[0] ?? 0,
                maxVal: options.maxVal ?? primaryLayer.range?.[1] ?? 1,
                mode: isIso ? 'isometric' : 'topdown'
            });
        } else {
            // CPU Canvas Adapter
            if (!(grid as any)._renderer) {
                (grid as any)._renderer = new CanvasAdapter(canvas);
            }
            const adapter = (grid as any)._renderer as CanvasAdapter;
            adapter.renderFromFaces(
                grid.cubes.map(r => r.map(c => c!.faces)),
                grid.nx, grid.ny, grid.cols, grid.rows,
                {
                    faceIndex: options.faceIndex ?? resolveFace(primaryLayer),
                    obstaclesFace: options.obstaclesFace ?? (obstacleLayer ? resolveFace(obstacleLayer) : undefined),
                    vorticityFace: options.vorticityFace ?? (vorticityLayer ? resolveFace(vorticityLayer) : undefined),
                    colormap: options.colormap ?? primaryLayer.colormap ?? 'heatmap',
                    minVal: options.minVal ?? primaryLayer.range?.[0] ?? 0,
                    maxVal: options.maxVal ?? primaryLayer.range?.[1] ?? 1,
                    sliceZ: options.sliceZ ?? 0
                }
            );
        }
    }

    /**
     * @description Créé un profil visuel de secours basé sur les anciens tags.
     */
    private static getFallbackProfile(engine: any): any {
        const tags = engine.getTags ? engine.getTags() : [];
        const isArctic = tags.includes('arctic');
        const isOcean = tags.includes('ocean');
        const isLbm = tags.includes('lbm');
        const p = engine.parity ?? 0;

        return {
            layers: [
                {
                    faceIndex: isArctic ? (22 + p) : 0,
                    role: 'primary',
                    colormap: isArctic ? 'arctic' : (isOcean ? 'ocean' : 'heatmap'),
                    range: isOcean ? [0, 1.5] : [0, 1]
                },
                { faceIndex: 18, role: 'obstacle' },
                { faceIndex: 21, role: 'vorticity' }
            ],
            defaultMode: (isOcean || tags.includes('2.5d')) ? '2.5d' : 'topdown'
        };
    }
}
