import { VisualProfile } from "../engines/IHypercubeEngine";

/**
 * VisualRegistry
 * 
 * Central repository for all rendering presets and compositions.
 * Decouples physical semantic hints from specific visual implementation.
 */
export class VisualRegistry {
    private static profiles: Map<string, VisualProfile> = new Map();

    /**
     * Initialise les profils par défaut du framework.
     */
    static {
        // Profil Arctique (LBM / Aérodynamique)
        this.register('arctic', {
            layers: [
                { faceLabel: 'Obstacles', role: 'obstacle' },
                { faceLabel: 'Smoke', role: 'primary', colormap: 'arctic' },
                { faceLabel: 'Vorticity', role: 'vorticity', alpha: 0.5 }
            ],
            defaultMode: 'topdown'
        });

        // Profil Océan (Vagues / Bio)
        this.register('ocean', {
            layers: [
                { faceLabel: 'Water_Height', role: 'primary', colormap: 'ocean', range: [0.0, 1.5] },
                { faceLabel: 'Biology', role: 'secondary', colormap: 'ocean', alpha: 0.3 }
            ],
            defaultMode: '2.5d'
        });

        // Profil Standard (Heatmap)
        this.register('standard', {
            layers: [
                { role: 'primary', colormap: 'heatmap', range: [0, 1] }
            ],
            defaultMode: 'topdown'
        });
    }

    /**
     * Enregistre un nouveau profil visuel.
     */
    public static register(id: string, profile: VisualProfile) {
        this.profiles.set(id, profile);
    }

    /**
     * Récupère un profil par son ID.
     */
    public static get(id: string): VisualProfile | undefined {
        return this.profiles.get(id);
    }

    /**
     * Résout un profil (soit direct, soit via ID).
     */
    public static resolve(profile: VisualProfile | string): VisualProfile {
        if (typeof profile === 'string') {
            return this.get(profile) || this.get('standard')!;
        }
        if (profile.styleId) {
            const base = this.get(profile.styleId) || this.get('standard')!;
            // Override possible
            return { ...base, ...profile };
        }
        return profile;
    }
}
