import { EngineDescriptor } from '../engines/EngineManifest';

/**
 * V8 ParameterMapper - Translates Semantic Names to Physical Buffer Offsets.
 * This resolution happens ONCE at startup to ensure 0-overhead during execution.
 */
export class ParameterMapper {
    private mapping: Map<string, number> = new Map();

    constructor(private descriptor: EngineDescriptor) {
        this.resolveOffsets();
    }

    /**
     * @description Calcule les offsets physiques des paramètres dans le buffer d'uniformes.
     * En V8, on suit l'ordre de déclaration du Manifeste de manière déterministe.
     */
    private resolveOffsets() {
        // En WebGPU, les uniformes sont souvent allignés sur 4 octets (float32).
        // On suppose ici un layout linéaire simple pour le draft V8.
        this.descriptor.parameters.forEach((param, index) => {
            this.mapping.set(param.name, index);
        });
    }

    /**
     * @description retourne l'index physique (slot) associé à un nom sémantique.
     */
    getOffset(name: string): number {
        const offset = this.mapping.get(name);
        if (offset === undefined) {
            throw new Error(`[ParameterMapper] Paramètre '${name}' inconnu dans le manifeste de l'engine '${this.descriptor.name}'.`);
        }
        return offset;
    }

    /**
     * @description Retourne la liste complète des valeurs par défaut pour l'initialisation du buffer.
     */
    getDefaults(): Float32Array {
        const values = new Float32Array(this.descriptor.parameters.length);
        this.descriptor.parameters.forEach((param, index) => {
            values[index] = param.defaultValue;
        });
        return values;
    }
}
