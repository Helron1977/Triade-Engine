/**
 * TriadeGPUContext est un singleton utilitaire pour initialiser l'environnement WebGPU.
 * Il récupère l'Adapter physique et le Device logique pour Hypercube-Compute.
 */
export class TriadeGPUContext {
    private static _device: GPUDevice | null = null;
    private static _adapter: GPUAdapter | null = null;
    private static _isSupported: boolean | null = null;

    /**
     * Vérifie si le navigateur supporte WebGPU.
     */
    static get isSupported(): boolean {
        if (this._isSupported === null) {
            this._isSupported = typeof navigator !== 'undefined' && 'gpu' in navigator;
        }
        return this._isSupported;
    }

    /**
     * Retourne le GPUDevice actif. Renvoie une erreur si non initialisé.
     */
    static get device(): GPUDevice {
        if (!this._device) {
            throw new Error("[Triade GPU] GPUDevice non initialisé. Appelez TriadeGPUContext.init() d'abord.");
        }
        return this._device;
    }

    /**
     * Initialise l'interface WebGPU (Adapter + Device).
     * @returns boolean true si succès, false si non supporté.
     */
    static async init(options?: GPURequestAdapterOptions): Promise<boolean> {
        if (!this.isSupported) {
            console.warn("[Triade GPU] WebGPU n'est pas supporté par ce navigateur.");
            return false;
        }

        if (this._device) {
            return true; // Déjà initialisé
        }

        try {
            this._adapter = await navigator.gpu.requestAdapter(options);
            if (!this._adapter) {
                console.warn("[Triade GPU] Aucun adaptateur WebGPU trouvé.");
                return false;
            }

            this._device = await this._adapter.requestDevice();

            // Gérer la perte du device
            this._device.lost.then((info) => {
                console.error(`[Triade GPU] WebGPU Device perdu: ${info.message}`);
                this._device = null;
                this._adapter = null;
            });

            return true;
        } catch (error) {
            console.error("[Triade GPU] Erreur d'initialisation WebGPU:", error);
            return false;
        }
    }
}
