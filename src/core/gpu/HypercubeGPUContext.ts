/**
 * HypercubeGPUContext est un singleton utilitaire pour initialiser l'environnement WebGPU.
 * Il récupère l'Adapter physique et le Device logique pour Hypercube-Compute.
 */
export class HypercubeGPUContext {
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
            throw new Error("[Hypercube GPU] GPUDevice non initialisé. Appelez HypercubeGPUContext.init() d'abord.");
        }
        return this._device;
    }

    /**
     * Initialise l'interface WebGPU (Adapter + Device).
     * @returns boolean true si succès, false si non supporté.
     */
    static async init(options?: GPURequestAdapterOptions): Promise<boolean> {
        if (!this.isSupported) {
            console.warn("[Hypercube GPU] WebGPU n'est pas supporté par ce navigateur.");
            return false;
        }

        if (this._device) {
            return true; // Déjà initialisé
        }

        try {
            this._adapter = await navigator.gpu.requestAdapter(options);
            if (!this._adapter) {
                console.warn("[Hypercube GPU] Aucun adaptateur WebGPU trouvé.");
                return false;
            }

            this._device = await this._adapter.requestDevice();

            // Gérer la perte du device
            this._device.lost.then((info) => {
                console.error(`[Hypercube GPU] WebGPU Device perdu: ${info.message}`);
                this._device = null;
                this._adapter = null;
            });

            return true;
        } catch (error) {
            console.error("[Hypercube GPU] Erreur d'initialisation WebGPU:", error);
            return false;
        }
    }

    /**
     * Compile un code source WGSL en Compute Pipeline WebGPU.
     * @param wgslCode Code source WGSL
     * @param entryPoint Nom de la fonction d'entrée (ex: 'compute_main')
     */
    static createComputePipeline(wgslCode: string, entryPoint: string): GPUComputePipeline {
        const shaderModule = this.device.createShaderModule({ code: wgslCode });
        return this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: entryPoint
            }
        });
    }

    /**
     * Crée et initialise un Storage Buffer WebGPU à partir d'un Float32Array.
     */
    static createStorageBuffer(data: Float32Array): GPUBuffer {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            // STORAGE pour la lecture/écriture dans le shader
            // COPY_SRC pour relire le résultat depuis le CPU
            // COPY_DST pour mettre à jour les données depuis le CPU
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        });

        new Float32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }
}




































