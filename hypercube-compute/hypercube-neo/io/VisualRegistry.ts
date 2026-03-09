import { VisualProfile } from "../core/types";

/**
 * VisualRegistry
 * 
 * Central repository for all rendering presets and compositions for Hypercube Neo.
 * Decouples physical semantic hints from specific visual implementation.
 */
export class VisualRegistry {
    private static profiles: Map<string, VisualProfile> = new Map();

    /**
     * Initialize default framework profiles.
     */
    static {
        // Arctic Profile (LBM / Aerodynamics)
        this.register('arctic', {
            layers: [
                { faceLabel: 'Obstacles', role: 'obstacle' },
                { faceLabel: 'Smoke', role: 'primary', colormap: 'arctic' },
                { faceLabel: 'Vorticity', role: 'vorticity', alpha: 0.5 }
            ],
            defaultMode: 'topdown'
        });

        // Ocean Profile (Waves / Bio)
        this.register('ocean', {
            layers: [
                { faceLabel: 'Water_Height', role: 'primary', colormap: 'ocean', range: [0.0, 1.5] },
                { faceLabel: 'Biology', role: 'secondary', colormap: 'ocean', alpha: 0.3 }
            ],
            defaultMode: '2.5d'
        });

        // Standard Profile (Heatmap)
        this.register('standard', {
            layers: [
                { role: 'primary', colormap: 'heatmap', range: [0, 1] }
            ],
            defaultMode: 'topdown'
        });
    }

    /**
     * Register a new visual profile.
     */
    public static register(id: string, profile: VisualProfile) {
        this.profiles.set(id, profile);
    }

    /**
     * Retrieve a profile by ID.
     */
    public static get(id: string): VisualProfile | undefined {
        return this.profiles.get(id);
    }

    /**
     * Resolve a profile (either direct or via ID).
     */
    public static resolve(profile: VisualProfile | string): VisualProfile {
        if (typeof profile === 'string') {
            return this.get(profile) || this.get('standard')!;
        }
        if (profile.styleId) {
            const base = this.get(profile.styleId) || this.get('standard')!;
            return { ...base, ...profile };
        }
        return profile;
    }
}
