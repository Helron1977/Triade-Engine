import { HypercubeCpuGrid } from './HypercubeCpuGrid';
import { ParameterMapper } from './ParameterMapper';
import { EngineDescriptor } from '../engines/EngineManifest';
import { Shape } from './Shapes';
import { Rasterizer } from './Rasterizer';
import { V8_PARAMS_OFFSET, V8_PARAM_STRIDE } from './UniformPresets';
import { IHypercubeEngine } from '../engines/IHypercubeEngine';

/**
 * V8 EngineProxy - The high-level handle for real-time interaction.
 * Orchestrates parameters and world objects without pipeline stalls.
 */
export class V8EngineProxy {
    private mapper: ParameterMapper;

    constructor(
        public grid: HypercubeCpuGrid,
        public descriptor: EngineDescriptor,
        public engine: IHypercubeEngine
    ) {
        this.mapper = new ParameterMapper(descriptor);
        this.initializeUniforms();
    }

    /**
     * @description Initialise les paramètres par défaut dans la grille.
     */
    private initializeUniforms() {
        const defaults = this.mapper.getDefaults();
        // V8 Rule: Centralized Offsets (Flowless Framework)
        if ((this.grid as any).uniforms) {
            for (let i = 0; i < defaults.length; i++) {
                (this.grid as any).uniforms[V8_PARAMS_OFFSET + i * V8_PARAM_STRIDE] = defaults[i];
            }
        }
    }

    /**
     * @description Mise à jour sémantique d'un paramètre.
     * Zero-Refactor : Le dev utilise le nom défini dans son manifeste.
     */
    setParam(name: string, value: number) {
        const offset = this.mapper.getOffset(name);
        if ((this.grid as any).uniforms) {
            // V8 Rule: Aligned Offset from Source of Truth
            (this.grid as any).uniforms[V8_PARAMS_OFFSET + offset * V8_PARAM_STRIDE] = value;
        }
    }

    /**
     * @description Injection d'un objet géométrique en cours de simulation.
     */
    addShape(shape: Shape) {
        Rasterizer.paint(this.grid, shape);
        // Note: La sync des bords est gérée par le Rasterizer
    }

    /**
     * @description Taille globale du domaine.
     */
    get nx() { return this.grid.nx; }
    get ny() { return this.grid.ny; }
    get nz() { return this.grid.nz; }

    /**
     * @description Exécute un pas de temps (calcul + sync).
     * V8 Rule: Synchronise automatiquement les faces marquées 'isSynchronized' dans le manifeste.
     */
    async compute() {
        const syncIndices = this.descriptor.faces
            .filter(f => f.isSynchronized)
            .map((_, i) => i);

        await this.grid.compute(syncIndices);
    }

    /**
     * @description Retourne l'indice de la face contenant le résultat du dernier calcul.
     * En mode GPU, c'est toujours 0 (car les phys buffers sont swappés).
     * En mode CPU, c'est 0 ou 1 selon la parité du ping-pong.
     */
    get activeFaceIndex(): number {
        if (this.grid.mode === 'gpu') {
            return 0;
        }
        // En mode CPU, si parity est impaire (ex: 1), c'est qu'on vient d'écrire dans Face 1.
        const p = (this.engine as any).parity ?? 0;
        return (p % 2 !== 0) ? 1 : 0;
    }
}
