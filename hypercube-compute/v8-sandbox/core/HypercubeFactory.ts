import { HypercubeMasterBuffer } from './HypercubeMasterBuffer';
import { HypercubeGPUContext } from '../../src/core/gpu/HypercubeGPUContext';
import { HypercubeCpuGrid } from './HypercubeCpuGrid';
import { EngineDescriptor, BoundaryManifest } from '../engines/EngineManifest';
import { Rasterizer } from './Rasterizer';
import { Shape } from './Shapes';
import { V8EngineProxy } from './V8EngineProxy';
import { V8EngineShim } from './V8EngineShim';
import { IHypercubeEngine } from '../engines/IHypercubeEngine';

export interface GridConfig {
    dimensions: {
        nx: number;
        ny: number;
        nz?: number;
        chunks: [number, number];
    };
    boundaries?: BoundaryManifest;
    initialState?: Shape[]; // Tableau de formes "acteurs"
    params?: Record<string, any>;
    mode?: 'cpu' | 'gpu';
    useWorkers?: boolean;
}

export class HypercubeFactory {
    /**
     * @description L'entrée unique pour créer un univers Hypercube.
     * Valide l'alignement GPU et alloue la mémoire de manière optimale.
     */
    static async instantiate(
        descriptor: EngineDescriptor,
        config: GridConfig,
        gpuEngineClass?: new () => any
    ): Promise<V8EngineProxy> {
        // 0. Contrôle de garde GPU (V8 Safety)
        if (config.mode === 'gpu' && !HypercubeGPUContext.isInitialized) {
            throw new Error(`[HypercubeFactory] Impossible d'instancier un moteur GPU car HypercubeGPUContext.init() n'a pas été appelé (ou a échoué).`);
        }

        // 1. Validation de l'alignement GPU (Multiple de 16 requis pour les Workgroups)
        const { nx, ny, nz = 1, chunks } = config.dimensions;

        if (nx % 16 !== 0 || ny % 16 !== 0) {
            throw new Error(`[HypercubeFactory] Dimensions invalides (${nx}x${ny}). Les dimensions doivent être des multiples de 16 pour l'alignement GPU.`);
        }

        // 2. Calcul des ressources nécessaires (Basé sur le Manifest et l'alignement WebGPU)
        const numFaces = descriptor.faces.length;
        const ALIGNMENT = 256;
        const stridePerFace = Math.ceil((nx * ny * nz * 4) / ALIGNMENT) * ALIGNMENT;
        const bytesPerChunk = stridePerFace * numFaces;
        const totalChunks = chunks[0] * chunks[1];

        // Taille totale : (Bytes par Chunk + Alignement inter-chunk 256) * Nombre de chunks
        const totalSize = (Math.ceil(bytesPerChunk / ALIGNMENT) * ALIGNMENT) * totalChunks;

        // 3. Contrôle de garde (V8 Safety)
        const MAX_SAFE_ALLOCATION = 1024 * 1024 * 1024; // 1GB
        if (totalSize > MAX_SAFE_ALLOCATION) {
            console.warn(`[HypercubeFactory] Allocation massive détectée (${(totalSize / (1024 * 1024)).toFixed(2)} MB). Vérifiez vos dimensions.`);
        }

        // 4. Allocation du MasterBuffer (Automatique)
        const masterBuffer = new HypercubeMasterBuffer(totalSize + 2048); // + Small Header Margin

        // 4. Instantiation de la Grille
        // Détection de la périodicité : si le rôle est 'joint' sur des bords opposés du monde.
        const isPeriodic = config.boundaries?.all?.role === 'joint' ||
            (config.boundaries?.left?.role === 'joint' && config.boundaries?.right?.role === 'joint');

        // ...
        const grid = await HypercubeCpuGrid.create(
            chunks[0],
            chunks[1],
            { nx, ny, nz },
            masterBuffer,
            () => {
                const shim = new V8EngineShim(descriptor);
                if (gpuEngineClass) {
                    const gpuInst = new gpuEngineClass();
                    (shim as any).initGPU = gpuInst.initGPU.bind(gpuInst);
                    (shim as any).computeGPU = gpuInst.computeGPU.bind(gpuInst);
                    (shim as any).wgslSource = gpuInst.wgslSource;
                    (shim as any).parity = (gpuInst as any).parity;
                }
                return shim;
            },
            numFaces,
            isPeriodic,
            config.useWorkers ?? true,
            undefined,
            config.mode ?? 'gpu'
        );

        // 5. Injection des réglages (Conditions aux limites avec rôles et facteurs)
        if (config.boundaries) {
            grid.boundaryConfig = config.boundaries as any;
        }
        (grid as any)._descriptor = descriptor;

        // 6. Initialisation de l'état (Rasterization des formes)
        if (config.initialState) {
            for (const shape of config.initialState) {
                Rasterizer.paint(grid, shape);
            }
        }

        // 7. Synchronisation matérielle initiale (V8 Ready)
        if (config.mode === 'gpu') {
            grid.pushToGPU();
        }

        // 8. Création du Handle de contrôle (V8 Proxy)
        const engine = grid.cubes[0][0]?.engine as IHypercubeEngine;
        const proxy = new V8EngineProxy(grid, descriptor, engine);

        // 8. Injection des paramètres initiaux
        if (config.params) {
            for (const [name, value] of Object.entries(config.params)) {
                proxy.setParam(name, value);
            }
        }

        console.info(`[HypercubeFactory] Engine '${descriptor.name}' instancié via Proxy V8.`);
        return proxy;
    }
}
