import { TriadeMasterBuffer } from './TriadeMasterBuffer';
import type { ITriadeEngine } from '../engines/ITriadeEngine';

export class TriadeCubeV2 {
    public readonly mapSize: number;
    public readonly faces: Float32Array[] = [];
    public readonly offset: number; // Added this property
    public engine: ITriadeEngine | null = null;

    /**
     * @param mapSize La résolution (N x N)
     * @param masterBuffer L'allocateur de RAM central de l'Orchestrateur
     */
    constructor(mapSize: number, masterBuffer: TriadeMasterBuffer, numFaces: number = 6) {
        this.mapSize = mapSize;

        // 1. Demander la réservation de l'espace sur la bande mémoire (O(1) Memory Layout)
        this.offset = masterBuffer.allocateCube(mapSize, numFaces);

        // 2. Création des Vues 
        const floatCount = mapSize * mapSize;
        const bytesPerFace = floatCount * Float32Array.BYTES_PER_ELEMENT;

        // Génération des N "Views" mathématiques
        for (let i = 0; i < numFaces; i++) {
            this.faces.push(
                new Float32Array(
                    masterBuffer.buffer,
                    this.offset + (i * bytesPerFace),
                    floatCount
                )
            );
        }
    }

    /**
     * Injecte le cerveau mathématique dans ce paquet mémoire.
     */
    setEngine(engine: ITriadeEngine) {
        this.engine = engine;
    }

    /**
     * Exécute le calcul de la Frame (O1, Wavefront, Automates)
     */
    compute() {
        if (!this.engine) {
            console.warn("[TriadeCubeV2] Aucun Engine (Cerveau) n'a été assigné à ce cube.");
            return;
        }
        this.engine.compute(this.faces, this.mapSize);
    }

    /** Helper pour vider une face spécifique */
    clearFace(faceIndex: number) {
        this.faces[faceIndex].fill(0);
    }
}
