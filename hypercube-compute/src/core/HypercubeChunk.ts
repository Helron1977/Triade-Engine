import { HypercubeMasterBuffer } from './HypercubeMasterBuffer';
import type { IHypercubeEngine } from '../engines/IHypercubeEngine';
import { HypercubeGPUContext } from './gpu/HypercubeGPUContext';

export class HypercubeChunk {
    public readonly mapSize: number;
    public readonly faces: Float32Array[] = [];
    public gpuBuffer: GPUBuffer | null = null; // Un seul buffer contigu pour le GPU (V3 GodMode)
    public readonly offset: number;
    public readonly stride: number; // Exposé pour la WorkerPool
    public engine: IHypercubeEngine | null = null;
    public readonly x: number;
    public readonly y: number;
    private masterBuffer: HypercubeMasterBuffer;

    constructor(x: number, y: number, mapSize: number, masterBuffer: HypercubeMasterBuffer, numFaces: number = 6) {
        this.x = x;
        this.y = y;
        this.masterBuffer = masterBuffer;
        this.mapSize = mapSize;
        const allocation = masterBuffer.allocateCube(mapSize, numFaces);
        this.offset = allocation.offset;
        this.stride = allocation.stride;

        const floatCount = mapSize * mapSize;

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
            this.engine.initGPU(HypercubeGPUContext.device, this.gpuBuffer, this.stride, this.mapSize);
        }
    }

    async compute() {
        if (!this.engine) return;
        await this.engine.compute(this.faces, this.mapSize, this.x, this.y);
    }

    /** Helper pour vider une face spécifique */
    clearFace(faceIndex: number) {
        this.faces[faceIndex].fill(0);
    }

    /**
     * Rapatrie les données de la VRAM vers la RAM (Zero-Copy Host Buffer).
     * Nécessaire pour la visualisation ou la validation CPU des résultats GPU.
     */
    async syncToHost() {
        if (!this.gpuBuffer) return;

        const totalSize = this.faces.length * this.stride;
        const device = HypercubeGPUContext.device;

        // 1. Créer un buffer de lecture (Staging)
        const stagingBuffer = device.createBuffer({
            size: totalSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // 2. Copier de la VRAM vers le Staging
        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.gpuBuffer, 0, stagingBuffer, 0, totalSize);
        device.queue.submit([commandEncoder.finish()]);

        // 3. Mapper et copier vers le MasterBuffer (RAM)
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = stagingBuffer.getMappedRange();

        // Copie directe dans l'ArrayBuffer partagé
        new Uint8Array(this.masterBuffer.buffer, this.offset, totalSize).set(new Uint8Array(arrayBuffer));

        stagingBuffer.unmap();
        stagingBuffer.destroy();
    }

    /**
     * Envoie les données de la RAM (MasterBuffer) vers la VRAM (GPUBuffer).
     * Indispensable pour l'interactivité ou l'initialisation complexe.
     */
    syncFromHost() {
        if (!this.gpuBuffer) return;

        const totalSize = this.faces.length * this.stride;

        HypercubeGPUContext.device.queue.writeBuffer(
            this.gpuBuffer,
            0,
            this.masterBuffer.buffer,
            this.offset,
            totalSize
        );
    }
}




































