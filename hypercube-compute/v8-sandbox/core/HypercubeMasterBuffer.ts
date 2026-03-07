export class HypercubeMasterBuffer {
    public readonly buffer: ArrayBuffer | SharedArrayBuffer;
    private offset: number = 0;

    public static readonly MAGIC_NUMBER = 0x48595045; // "HYPE"
    public static readonly VERSION = 4;
    public static readonly HEADER_SIZE = 256;

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

        // Header Initialization
        const header = new Uint32Array(this.buffer, 0, 2);
        header[0] = HypercubeMasterBuffer.MAGIC_NUMBER;
        header[1] = HypercubeMasterBuffer.VERSION;

        this.offset = HypercubeMasterBuffer.HEADER_SIZE;
    }

    /**
     * Alloue les octets nécessaires pour un Cube 3D de N Faces en O(1).
     * Garantit un alignement de 256 octets (WebGPU optimal stride).
     * @param nx Largeur
     * @param ny Hauteur
     * @param nz Profondeur (par défaut 1 pour compatibilité 2D)
     * @param numFaces Nombre de faces
     * @returns Un objet contenant l'offset de début et le stride réel (padding inclus)
     */
    allocateCube(nx: number, ny: number, nz: number = 1, numFaces: number = 6): { offset: number, stride: number } {
        const ALIGNMENT = 256;

        // 1. Aligner le début du cube
        this.offset = Math.ceil(this.offset / ALIGNMENT) * ALIGNMENT;
        const startOffset = this.offset;

        // 2. Calculer le stride aligné pour chaque face
        const cellsPerFace = nx * ny * nz;
        const bytesPerFaceRaw = cellsPerFace * 4; // 4 octets par float32
        const stride = Math.ceil(bytesPerFaceRaw / ALIGNMENT) * ALIGNMENT;

        const totalCubeBytes = stride * numFaces;

        if (this.offset + totalCubeBytes > this.buffer.byteLength) {
            throw new Error(`[HypercubeMasterBuffer] Out Of Memory. Impossible d'allouer ${totalCubeBytes} bytes supplémentaires (3D: ${nx}x${ny}x${nz}).`);
        }

        this.offset += totalCubeBytes;

        return { offset: startOffset, stride };
    }

    /** Retourne la quantité de RAM consommée */
    getUsedMemoryInMB(): string {
        return (this.offset / (1024 * 1024)).toFixed(2) + ' MB';
    }
}




































