export class HypercubeMasterBuffer {
    public readonly buffer: ArrayBuffer | SharedArrayBuffer;
    private offset: number = 0;

    /**
     * Alloue un unique bloc de mémoire vive pour l'ensemble du système.
     * Utilise un SharedArrayBuffer si disponible pour le support CPU multithread (Zero-Copy).
     * @param totalBytes Taille totale de la RAM allouée (par défaut 100 MB).
     */
    constructor(totalBytes: number = 100 * 1024 * 1024) {
        if (typeof SharedArrayBuffer !== 'undefined') {
            this.buffer = new SharedArrayBuffer(totalBytes);
        } else {
            console.warn("[HypercubeMasterBuffer] SharedArrayBuffer n'est pas supporté (vérifiez vos headers COOP/COEP). Fallback sur ArrayBuffer (pas de multi-threading CPU possible).");
            this.buffer = new ArrayBuffer(totalBytes);
        }
    }

    /**
     * Alloue les octets nécessaires pour un Cube de N Faces en O(1).
     * Garantit un alignement de 256 octets (WebGPU optimal stride).
     * @param mapSize Résolution (ex: 400x400)
     * @returns Un objet contenant l'offset de début et le stride réel (padding inclus)
     */
    allocateCube(mapSize: number, numFaces: number = 6): { offset: number, stride: number } {
        const ALIGNMENT = 256;

        // 1. Aligner le début du cube
        this.offset = Math.ceil(this.offset / ALIGNMENT) * ALIGNMENT;
        const startOffset = this.offset;

        // 2. Calculer le stride aligné pour chaque face
        const bytesPerFaceRaw = mapSize * mapSize * 4; // 4 octets par float32
        const stride = Math.ceil(bytesPerFaceRaw / ALIGNMENT) * ALIGNMENT;

        const totalCubeBytes = stride * numFaces;

        if (this.offset + totalCubeBytes > this.buffer.byteLength) {
            throw new Error(`[HypercubeMasterBuffer] Out Of Memory. Impossible d'allouer ${totalCubeBytes} bytes supplémentaires.`);
        }

        this.offset += totalCubeBytes;

        return { offset: startOffset, stride };
    }

    /** Retourne la quantité de RAM consommée */
    getUsedMemoryInMB(): string {
        return (this.offset / (1024 * 1024)).toFixed(2) + ' MB';
    }
}




































