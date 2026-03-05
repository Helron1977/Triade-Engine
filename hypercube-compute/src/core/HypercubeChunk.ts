import { HypercubeMasterBuffer } from './HypercubeMasterBuffer';
import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

export class HypercubeChunk {
    public readonly nx: number;
    public readonly ny: number;
    public readonly nz: number;
    public readonly faces: Float32Array[] = [];
    public gpuBuffer: GPUBuffer | null = null; // Un seul buffer contigu pour le GPU (V3 GodMode)
    public readonly offset: number;
    public readonly stride: number; // Exposé pour la WorkerPool
    public engine: IHypercubeEngine | null = null;
    public readonly x: number;
    public readonly y: number;
    public readonly z: number;
    private masterBuffer: HypercubeMasterBuffer;

    // Async Readback / Double Buffering
    private stagingBuffers: Map<string, GPUBuffer> = new Map();
    private pendingSyncPromises: Promise<void>[] = [];

    constructor(x: number, y: number, nx: number, ny: number, nz: number = 1, masterBuffer: HypercubeMasterBuffer, numFaces: number = 6, z: number = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.masterBuffer = masterBuffer;
        this.nx = nx;
        this.ny = ny;
        this.nz = nz;

        const allocation = masterBuffer.allocateCube(nx, ny, nz, numFaces);
        this.offset = allocation.offset;
        this.stride = allocation.stride;

        const floatCount = nx * ny * nz;

        for (let i = 0; i < numFaces; i++) {
            this.faces.push(
                new Float32Array(
                    masterBuffer.buffer,
                    this.offset + (i * this.stride),
                    floatCount
                )
            );
        }
    }

    /**
     * Retourne l'index linéaire pour une position (x, y, z) locale au chunk.
     */
    public getIndex(lx: number, ly: number, lz: number = 0): number {
        return (lz * this.ny * this.nx) + (ly * this.nx) + lx;
    }

    /**
     * Extrait une tranche 2D (Slice Z) d'une face spécifique.
     * @returns Un Float32Array (copie) représentant la couche demandée.
     */
    public getSlice(faceIndex: number, lz: number): Float32Array {
        const sliceSize = this.nx * this.ny;
        const offset = lz * sliceSize;
        return this.faces[faceIndex].slice(offset, offset + sliceSize);
    }

    setEngine(engine: IHypercubeEngine) {
        this.engine = engine;
    }

    /**
     * Initialise la contrepartie GPU (VRAM) de ce Cube Logiciel.
     * Appelé par le HypercubeGrid lors de sa création en mode 'webgpu'.
     */
    initGPU() {
        if (!this.engine) return;

        const totalSize = this.faces.length * this.stride;

        this.gpuBuffer = HypercubeGPUContext.device.createBuffer({
            size: totalSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        // Upload initial des données (Faces + Padding)
        (HypercubeGPUContext.device.queue as any).writeBuffer(
            this.gpuBuffer,
            0,
            this.masterBuffer.buffer,
            this.offset,
            totalSize
        );

        // Informer le moteur mathématique
        if (this.engine.initGPU) {
            this.engine.initGPU(HypercubeGPUContext.device, this.gpuBuffer, this.stride, this.nx, this.ny, this.nz);
        }
    }

    async compute() {
        if (!this.engine) return;
        await (this.engine.compute as any)(this.faces, this.nx, this.ny, this.nz, this.x, this.y, this.z);
    }

    /** Helper pour vider une face spécifique */
    clearFace(faceIndex: number) {
        this.faces[faceIndex].fill(0);
    }

    /**
     * Rapatrie les données de la VRAM vers la RAM (Zero-Copy Host Buffer).
     * @param faceIndices Liste optionnelle des faces à synchroniser.
     * @param block Si vrai, attend la fin du transfert (comportement par défaut).
     */
    async syncToHost(faceIndices?: number[], block: boolean = true) {
        if (!this.gpuBuffer) return;

        const device = HypercubeGPUContext.device;

        // Si on a des promesses en cours d'une frame précédente, on les attend d'abord
        if (this.pendingSyncPromises.length > 0) {
            await Promise.all(this.pendingSyncPromises);
            this.pendingSyncPromises = [];
        }

        const commandEncoder = device.createCommandEncoder();
        const activeStaging: { buffer: GPUBuffer, faceIdx: number, size: number }[] = [];

        const facesToSync = faceIndices && faceIndices.length > 0 ? faceIndices : Array.from({ length: this.faces.length }, (_, i) => i);

        for (const faceIdx of facesToSync) {
            if (faceIdx < 0 || faceIdx >= this.faces.length) continue;

            const faceSize = this.stride;
            const key = `face_${faceIdx}`;
            let stagingBuffer = this.stagingBuffers.get(key);

            if (!stagingBuffer) {
                stagingBuffer = device.createBuffer({
                    size: faceSize,
                    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
                });
                this.stagingBuffers.set(key, stagingBuffer);
            }

            commandEncoder.copyBufferToBuffer(
                this.gpuBuffer, faceIdx * this.stride,
                stagingBuffer, 0,
                faceSize
            );
            activeStaging.push({ buffer: stagingBuffer, faceIdx, size: faceSize });
        }

        device.queue.submit([commandEncoder.finish()]);

        const syncProcess = async () => {
            await Promise.all(activeStaging.map(as => as.buffer.mapAsync(GPUMapMode.READ)));

            for (const as of activeStaging) {
                const arrayBuffer = as.buffer.getMappedRange();
                const dstOffset = this.offset + (as.faceIdx * this.stride);
                new Uint8Array(this.masterBuffer.buffer, dstOffset, as.size).set(new Uint8Array(arrayBuffer));
                as.buffer.unmap();
            }
        };

        if (block) {
            await syncProcess();
        } else {
            this.pendingSyncPromises.push(syncProcess());
        }
    }

    /**
     * Envoie les données de la RAM (MasterBuffer) vers la VRAM (GPUBuffer).
     * Indispensable pour l'interactivité ou l'initialisation complexe.
     * @param faceIndices Liste optionnelle des faces à synchroniser (ex: [0, 1]). Si vide, synchronise tout.
     */
    syncFromHost(faceIndices?: number[]) {
        if (!this.gpuBuffer) return;

        const device = HypercubeGPUContext.device;

        if (!faceIndices || faceIndices.length === 0) {
            const totalSize = this.faces.length * this.stride;
            device.queue.writeBuffer(
                this.gpuBuffer,
                0,
                this.masterBuffer.buffer,
                this.offset,
                totalSize
            );
        } else {
            for (const faceIdx of faceIndices) {
                if (faceIdx < 0 || faceIdx >= this.faces.length) continue;

                device.queue.writeBuffer(
                    this.gpuBuffer,
                    faceIdx * this.stride,
                    this.masterBuffer.buffer,
                    this.offset + (faceIdx * this.stride),
                    this.stride
                );
            }
        }
    }
}




































