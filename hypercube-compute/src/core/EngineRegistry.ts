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

        if (config.boundaryConfig !== undefined && typeof engine.setBoundaryConfig === 'function') {
            engine.setBoundaryConfig(config.boundaryConfig);
        }

        // Cas particulier OceanEngine qui encapsule sa config dans 'params'
        if (engine.name === 'OceanEngine') {
            (engine as any).params = { ...(engine as any).params, ...config };
            return;
        }

        // Cas général : fusion directe (ex: Aerodynamics, Heatmap)
        for (const key of Object.keys(config)) {
            if (key !== 'boundaryConfig') {
                (engine as any)[key] = config[key];
            }
        }
    }
}

// ---- Auto-inscription des moteurs officiels ----

EngineRegistry.register('Heatmap (O1 Spatial Convolution)', HeatmapEngine, ['2d']);
EngineRegistry.register('FlowFieldEngine-V12', FlowFieldEngine, ['2d']);
EngineRegistry.register('Simplified Fluid Dynamics', FluidEngine, ['2d']);
EngineRegistry.register('Aerodynamics LBM D2Q9', AerodynamicsEngine, ['2d', 'lbm']);
EngineRegistry.register('OceanEngine', OceanEngine, ['2.5d', 'iso', 'lbm']);
EngineRegistry.register('HeatDiffusionEngine3D', HeatDiffusionEngine3D, ['3d', 'slice']);
EngineRegistry.register('Volume Diffusion (3D Stencil)', VolumeDiffusionEngine, ['3d', 'slice']);
EngineRegistry.register('GameOfLifeEngine', GameOfLifeEngine, ['2d', 'cellular']);
EngineRegistry.register('GrayScottEngine', GrayScottEngine, ['2d', 'organic']);

