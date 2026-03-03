/**
 * HypercubeGPUContext – Singleton pour initialiser WebGPU.
 * Gère Adapter, Device, buffers et cleanup.
 */
export class HypercubeGPUContext {
    private static _device: GPUDevice | null = null;
    private static _adapter: GPUAdapter | null = null;
    private static _isSupported: boolean | null = null;
    private static _preferredFormat: GPUTextureFormat = 'bgra8unorm'; // fallback

    static get isSupported(): boolean {
        if (this._isSupported === null) {
            this._isSupported = typeof navigator !== 'undefined' && 'gpu' in navigator;
            if (this.isSupported) {
                this._preferredFormat = navigator.gpu.getPreferredCanvasFormat?.() ?? 'bgra8unorm';
            }
        }
        return this._isSupported;
    }

    static get device(): GPUDevice {
        if (!this._device) {
            throw new Error("[Hypercube GPU] GPUDevice non initialisé. Appelez init() d'abord.");
        }
        return this._device;
    }

    static get preferredFormat(): GPUTextureFormat {
        return this._preferredFormat;
    }

    /**
     * Initialise WebGPU (Adapter + Device).
     * @param options Options pour requestAdapter / requestDevice
     */
    static async init(options: GPURequestAdapterOptions = {}): Promise<boolean> {
        if (!this.isSupported) {
            console.warn("[Hypercube GPU] WebGPU non supporté.");
            return false;
        }

        if (this._device) return true;

        try {
            this._adapter = await navigator.gpu.requestAdapter(options);
            if (!this._adapter) {
                console.warn("[Hypercube GPU] Aucun adaptateur trouvé.");
                return false;
            }

            // Required limits/features (exemple extensible)
            const requiredFeatures: GPUFeatureName[] = [];
            const requiredLimits: Record<string, number> = {
                maxComputeInvocationsPerWorkgroup: this._adapter.limits.maxComputeInvocationsPerWorkgroup,
                maxComputeWorkgroupSizeX: this._adapter.limits.maxComputeWorkgroupSizeX,
                maxComputeWorkgroupSizeY: this._adapter.limits.maxComputeWorkgroupSizeY,
                maxComputeWorkgroupSizeZ: this._adapter.limits.maxComputeWorkgroupSizeZ,
            };

            this._device = await this._adapter.requestDevice({
                requiredFeatures,
                requiredLimits,
            });

            // Gestion perte device
            this._device.lost.then((info) => {
                console.error(`[Hypercube GPU] Device perdu: ${info.message} (${info.reason})`);
                this._device = null;
                this._adapter = null;
            });

            return true;
        } catch (err) {
            console.error("[Hypercube GPU] Init échouée:", err);
            return false;
        }
    }

    /**
     * Crée un storage buffer à partir de Float32Array.
     * @param data Données initiales
     * @param mappedAtCreation Utiliser mappedAtCreation (true) ou writeBuffer (false)
     */
    static createStorageBuffer(data: Float32Array, mappedAtCreation: boolean = true): GPUBuffer {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation,
        });

        if (mappedAtCreation) {
            new Float32Array(buffer.getMappedRange()).set(data as any);
            buffer.unmap();
        } else {
            this.device.queue.writeBuffer(buffer, 0, data as any);
        }

        return buffer;
    }

    /**
     * Crée un uniform buffer dynamique.
     * @param data ArrayBuffer ou une vue typée
     */
    static createUniformBuffer(data: ArrayBuffer | { byteLength: number }): GPUBuffer {
        const buffer = this.device.createBuffer({
            size: Math.ceil(data.byteLength / 16) * 16, // align 16 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const actualData = 'buffer' in data ? (data as any) : data;
        this.device.queue.writeBuffer(buffer, 0, actualData);
        return buffer;
    }

    /**
     * Compile un WGSL en Compute Pipeline.
     */
    static createComputePipeline(wgslCode: string, entryPoint: string): GPUComputePipeline {
        const module = this.device.createShaderModule({ code: wgslCode });
        return this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint },
        });
    }

    /**
     * Nettoyage complet (pour hot-reload ou tab close).
     */
    static destroy(): void {
        if (this._device) {
            this._device.destroy();
            this._device = null;
        }
        this._adapter = null;
    }
}




































