import { EngineDescriptor } from './EngineManifest';

/**
 * HeatDiffusionV8 - First Proof of Concept for the Manifest-based Engine.
 */
export const HeatDiffusionV8: EngineDescriptor = {
    name: 'HeatDiffusion3D',
    description: '3D Heat Diffusion using Finite Difference (Laplacian)',

    // 1. Memory Layout (Semantic Names)
    faces: [
        { name: 'Temperature', type: 'scalar', isSynchronized: true, defaultValue: 0 },
        { name: 'TemperatureNext', type: 'scalar', isSynchronized: true, isReadOnly: false },
        { name: 'Obstacles', type: 'mask', isReadOnly: true, defaultValue: 0 }
    ],

    // 2. Control Contract (Semantic Names)
    parameters: [
        {
            name: 'diffusionRate',
            label: 'Diffusion Rate',
            description: 'Rate of heat spread through the volume',
            defaultValue: 0.1,
            min: 0,
            max: 0.25
        }
    ],

    // 3. Mathematical Rules (The "What")
    rules: [
        {
            type: 'diffusion',
            method: 'Explict-Euler',
            source: 'Temperature',
            params: {
                diffusionRate: 0.1
            }
        }
    ],

    // 3. Visualization Default
    visualProfile: {
        primary: 'Temperature',
        overlay: 'Obstacles',
        colormap: 'heatmap'
    }
};
