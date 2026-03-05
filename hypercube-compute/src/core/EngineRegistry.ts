import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HeatmapEngine } from '../engines/HeatmapEngine';
import { FlowFieldEngine } from '../engines/FlowFieldEngine';
import { FluidEngine } from '../engines/FluidEngine';
import { AerodynamicsEngine } from '../engines/AerodynamicsEngine';
import { OceanEngine } from '../engines/OceanEngine';
import { HeatDiffusionEngine3D } from '../engines/HeatDiffusionEngine3D';
import { VolumeDiffusionEngine } from '../engines/VolumeDiffusionEngine';
import { GameOfLifeEngine } from '../engines/GameOfLifeEngine';
import { GrayScottEngine } from '../engines/GrayScottEngine';

export type EngineConstructor = new (...args: any[]) => IHypercubeEngine;

export class EngineRegistry {
    private static engines = new Map<string, { constructor: EngineConstructor, tags: string[] }>();

    /**
     * Enregistre dynamiquement un nouveau moteur mathématique.
     */
    public static register(name: string, constructor: EngineConstructor, tags: string[] = []) {
        this.engines.set(name, { constructor, tags });
    }

    /**
     * Fabrique une instance de moteur à partir de son nom et d'une configuration.
     * Appliquera intelligemment la configuration si présente.
     */
    public static create(name: string, config?: any): IHypercubeEngine {
        const entry = this.engines.get(name);
        if (!entry) {
            throw new Error(`[EngineRegistry] Moteur non reconnu ou non supporté : "${name}". Avez-vous oublié de l'enregistrer ?`);
        }

        const engine = new entry.constructor();

        // Application générique de la configuration
        if (config) {
            EngineRegistry.applyConfig(engine, config);
        }

        return engine;
    }

    public static getTags(name: string): string[] {
        return this.engines.get(name)?.tags || [];
    }

    /**
     * Applique un dictionnaire de configuration à l'instance du moteur.
     */
    public static applyConfig(engine: IHypercubeEngine, config: any) {
        if (!config || typeof config !== 'object') return;

        // Priority 1: Engine-specific applyConfig
        if (typeof engine.applyConfig === 'function') {
            engine.applyConfig(config);
            return;
        }

        // Priority 2: Generic parity handling (must be top-level)
        if (config.parity !== undefined) {
            (engine as any).parity = config.parity;
        }

        if (config.boundaryConfig !== undefined && typeof engine.setBoundaryConfig === 'function') {
            engine.setBoundaryConfig(config.boundaryConfig);
        }

        // Priority 3: Generic merge (fallback)
        for (const key of Object.keys(config)) {
            if (key !== 'boundaryConfig' && key !== 'parity') {
                (engine as any)[key] = config[key];
            }
        }
    }
}

// ---- Auto-inscription des moteurs officiels ----

EngineRegistry.register('Heatmap (O1 Spatial Convolution)', HeatmapEngine, ['2d']);
EngineRegistry.register('FlowFieldEngine-V12', FlowFieldEngine, ['2d']);
EngineRegistry.register('Fluid Engine (O1 Tensor Nav-Stokes)', FluidEngine, ['2d']);
EngineRegistry.register('Aerodynamics LBM D2Q9', AerodynamicsEngine, ['2d', 'lbm']);
EngineRegistry.register('OceanEngine 2.5D', OceanEngine, ['2.5d', 'iso', 'lbm']);
EngineRegistry.register('HeatDiffusionEngine3D', HeatDiffusionEngine3D, ['3d', 'slice']);
EngineRegistry.register('Volume Diffusion (3D Stencil)', VolumeDiffusionEngine, ['3d', 'slice']);
EngineRegistry.register('GameOfLifeEngine', GameOfLifeEngine, ['2d', 'cellular']);
EngineRegistry.register('GrayScottEngine', GrayScottEngine, ['2d', 'organic']);

