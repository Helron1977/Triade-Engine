import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HeatmapEngine } from '../engines/HeatmapEngine';
import { FlowFieldEngine } from '../engines/FlowFieldEngine';
import { FluidEngine } from '../engines/FluidEngine';
import { AerodynamicsEngine } from '../engines/AerodynamicsEngine';
import { OceanEngine } from '../engines/OceanEngine';

export type EngineConstructor = new (...args: any[]) => IHypercubeEngine;

export class EngineRegistry {
    private static engines = new Map<string, EngineConstructor>();

    /**
     * Enregistre dynamiquement un nouveau moteur mathématique.
     */
    public static register(name: string, constructor: EngineConstructor) {
        this.engines.set(name, constructor);
    }

    /**
     * Fabrique une instance de moteur à partir de son nom et d'une configuration.
     * Appliquera intelligemment la configuration si présente.
     */
    public static create(name: string, config?: any): IHypercubeEngine {
        const ctor = this.engines.get(name);
        if (!ctor) {
            throw new Error(`[EngineRegistry] Moteur non reconnu ou non supporté : "${name}". Avez-vous oublié de l'enregistrer ?`);
        }

        const engine = new ctor();

        // Application générique de la configuration
        if (config) {
            EngineRegistry.applyConfig(engine, config);
        }

        return engine;
    }

    /**
     * Applique un dictionnaire de configuration à l'instance du moteur.
     */
    public static applyConfig(engine: IHypercubeEngine, config: any) {
        if (!config || typeof config !== 'object') return;

        // Cas particulier OceanEngine qui encapsule sa config dans 'params'
        if (engine.name === 'OceanEngine') {
            (engine as any).params = { ...(engine as any).params, ...config };
            return;
        }

        // Cas général : fusion directe (ex: Aerodynamics, Heatmap)
        for (const key of Object.keys(config)) {
            (engine as any)[key] = config[key];
        }
    }
}

// ---- Auto-inscription des moteurs officiels ----

EngineRegistry.register('Heatmap (O1 Spatial Convolution)', HeatmapEngine);
EngineRegistry.register('FlowFieldEngine-V12', FlowFieldEngine);
EngineRegistry.register('Simplified Fluid Dynamics', FluidEngine);
EngineRegistry.register('Lattice Boltzmann D2Q9 (O(1))', AerodynamicsEngine);
EngineRegistry.register('Aerodynamics LBM D2Q9', AerodynamicsEngine); // Alias de compatibilité
EngineRegistry.register('OceanEngine', OceanEngine);

