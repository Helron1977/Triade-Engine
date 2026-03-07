import { IHypercubeEngine, FlatTensorView } from '../engines/IHypercubeEngine';
import { EngineDescriptor } from '../engines/EngineManifest';

/**
 * V8EngineShim - Bridges the gap between a Manifest and the Core.
 * This is the object that actually runs inside each HypercubeChunk.
 */
export class V8EngineShim implements IHypercubeEngine {
    public parity: number = 0; // Support pour le ping-pong CPU

    constructor(public descriptor: EngineDescriptor) { }

    get name() { return this.descriptor.name; }
    getRequiredFaces() { return this.descriptor.faces.length; }

    /**
     * @description Initialisation CPU (Draft - Utilise les valeurs par défaut du manifeste)
     */
    init(faces: FlatTensorView[], nx: number, ny: number, nz: number): void {
        this.descriptor.faces.forEach((face, idx) => {
            if (face.defaultValue !== undefined) {
                faces[idx].fill(face.defaultValue);
            }
        });
    }

    /**
     * @description Boucle de calcul CPU (Rendu déterministe depuis les règles du manifeste)
     */
    compute(faces: FlatTensorView[], nx: number, ny: number, nz: number): void {
        const diffusionRule = this.descriptor.rules.find(r => r.type === 'diffusion');
        if (!diffusionRule) return;

        // --- PING-PONG LOGIC ---
        // On identifie les deux faces de température pour le ping-pong
        const tempIdx = this.descriptor.faces.findIndex(f => f.name === (diffusionRule.source || 'Temperature'));
        const nextIdx = this.descriptor.faces.findIndex(f => f.name === 'TemperatureNext');

        if (tempIdx === -1 || nextIdx === -1) return;

        // Swapping based on parity
        const isOdd = (this.parity % 2) !== 0;
        const srcFaceIdx = isOdd ? nextIdx : tempIdx;
        const dstFaceIdx = isOdd ? tempIdx : nextIdx;

        const src = faces[srcFaceIdx];
        const dst = faces[dstFaceIdx];

        // Obstacles (Face fixe)
        const obsFaceIdx = this.descriptor.faces.findIndex(f => f.type === 'mask');
        const obs = obsFaceIdx !== -1 ? faces[obsFaceIdx] : null;

        // Diffusion rate
        const rate = (this as any).diffusionRate ?? diffusionRule.params?.diffusionRate ?? 0.1;

        // COMPUTATION 2D
        for (let y = 1; y < ny - 1; y++) {
            const offset = y * nx;
            for (let x = 1; x < nx - 1; x++) {
                const idx = offset + x;

                if (obs && obs[idx] > 0.5) {
                    dst[idx] = 0.0;
                    continue;
                }

                const val = src[idx];
                const laplacian = (src[idx - 1] + src[idx + 1] + src[idx - nx] + src[idx + nx]) - 4.0 * val;

                dst[idx] = val + rate * laplacian;
            }
        }

        this.parity++;
    }
    /**
     * @description Mapping GPU (Délégué à l'AbstractGpuEngine correspondant)
     */
    initGPU(device: GPUDevice, readBuffer: GPUBuffer, writeBuffer: GPUBuffer, uniformBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void {
        // Cette méthode sera surchargée ou gérée par le Factory qui injectera 
        // l'implémentation GPU réelle (ex: HeatDiffusionGpuV8)
    }

    getSyncFaces(): number[] {
        const result: number[] = [];
        this.descriptor.faces.forEach((f, idx) => {
            if (f.isSynchronized) {
                result.push(idx);
            }
        });
        return result;
    }

    getConfig(): Record<string, any> {
        return {
            parity: this.parity
        };
    }

    applyConfig(config: any): void {
        if (config && config.parity !== undefined) {
            this.parity = config.parity;
        }
    }
}
