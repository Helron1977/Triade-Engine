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
    private masterBuffer: HypercubeMasterBuffer;

    constructor(mapSize: number, masterBuffer: HypercubeMasterBuffer, numFaces: number = 6) {
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
        await this.engine.compute(this.faces, this.mapSize);
    }

    /** Helper pour vider une face spécifique */
    clearFace(faceIndex: number) {
        this.faces[faceIndex].fill(0);
    }
}




































