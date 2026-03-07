import { EngineDescriptor } from './EngineManifest';

/**
 * OceanV8 - Manifest for the 2.5D Ocean Engine (LBM D2Q9 + Biology).
 * This manifest acts as a semantic bridge for the legacy OceanEngine (02).
 */
export const OceanV8: EngineDescriptor = {
    name: 'OceanEngine 2.5D',
    description: '2.5D Ocean Simulation with Thermal/Biological currents',

    // 1. Data Contract (25 Faces matching Physical Layout of OceanEngine)
    faces: [
        // LBM Populations P0 (0-8)
        { name: 'P0_0', type: 'scalar', isSynchronized: true },
        { name: 'P0_1', type: 'scalar', isSynchronized: true },
        { name: 'P0_2', type: 'scalar', isSynchronized: true },
        { name: 'P0_3', type: 'scalar', isSynchronized: true },
        { name: 'P0_4', type: 'scalar', isSynchronized: true },
        { name: 'P0_5', type: 'scalar', isSynchronized: true },
        { name: 'P0_6', type: 'scalar', isSynchronized: true },
        { name: 'P0_7', type: 'scalar', isSynchronized: true },
        { name: 'P0_8', type: 'scalar', isSynchronized: true },

        // LBM Populations P1 (9-17)
        { name: 'P1_0', type: 'scalar', isSynchronized: true },
        { name: 'P1_1', type: 'scalar', isSynchronized: true },
        { name: 'P1_2', type: 'scalar', isSynchronized: true },
        { name: 'P1_3', type: 'scalar', isSynchronized: true },
        { name: 'P1_4', type: 'scalar', isSynchronized: true },
        { name: 'P1_5', type: 'scalar', isSynchronized: true },
        { name: 'P1_6', type: 'scalar', isSynchronized: true },
        { name: 'P1_7', type: 'scalar', isSynchronized: true },
        { name: 'P1_8', type: 'scalar', isSynchronized: true },

        // Masks & Results (18-24)
        { name: 'Obstacles', type: 'mask', isReadOnly: true, defaultValue: 0 },   // 18
        { name: 'Velocity_X', type: 'scalar', isSynchronized: true },            // 19
        { name: 'Velocity_Y', type: 'scalar', isSynchronized: true },            // 20
        { name: 'Pressure', type: 'scalar', isOptional: true },                  // 21 (Reserved/Unused)
        { name: 'Water_Height', type: 'scalar', isSynchronized: true },          // 22
        { name: 'Biology', type: 'scalar', isSynchronized: true },               // 23 (P0)
        { name: 'BiologyNext', type: 'scalar', isSynchronized: true }            // 24 (P1)
    ],

    // 2. Control Contract (Parameters)
    parameters: [
        { name: 'tau_0', label: 'Relaxation Time', defaultValue: 0.8, min: 0.5, max: 2.0 },
        { name: 'smagorinsky', label: 'Smagorinsky', defaultValue: 0.2, min: 0, max: 0.5 },
        { name: 'bioDiffusion', label: 'Bio Diffusion', defaultValue: 0.05, min: 0, max: 0.2 },
        { name: 'bioGrowth', label: 'Bio Growth', defaultValue: 0.0005, min: 0, max: 0.01 }
    ],

    // 3. Compute Contract (Abstract representation)
    rules: [
        { type: 'advection', method: 'Upwind', source: 'Biology', field: 'Velocity' }
    ],

    // 4. Default Visualization Profile
    visualProfile: {
        primary: 'Biology',
        overlay: 'Obstacles',
        colormap: 'ocean'
    }
};
