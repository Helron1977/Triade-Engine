import { DataContract } from './DataContract';

/**
 * Manages the global simulation parity (Ping-Pong buffer indices).
 * Ensures that engines always know which buffer to read from and which to write to.
 */
export class ParityManager {
    private faceCache: Map<string, { baseIdx: number; isPingPong: boolean }> = new Map();
    private tick: number = 0;

    constructor(private dataContract: DataContract) {
        this.cacheMappings();
    }

    private cacheMappings(): void {
        const mappings = this.dataContract.getFaceMappings();
        let currentBaseIdx = 0;
        for (const mapping of mappings) {
            this.faceCache.set(mapping.name, {
                baseIdx: currentBaseIdx,
                isPingPong: mapping.isPingPong
            });
            currentBaseIdx += mapping.isPingPong ? 2 : 1;
        }
    }

    /**
     * Increments the simulation tick and swaps parity.
     */
    public nextTick(): void {
        this.tick++;
    }

    /**
     * Gets the current buffer indices for a specific face.
     * @returns { read: number, write: number } Indices into the chunk's physical faces array.
     */
    public getFaceIndices(faceName: string): { read: number; write: number } {
        const cached = this.faceCache.get(faceName);

        if (!cached) {
            throw new Error(`ParityManager: Face "${faceName}" not found in contract.`);
        }

        if (!cached.isPingPong) {
            return { read: cached.baseIdx, write: cached.baseIdx };
        }

        // Swapping logic: 
        // Tick 0: Read A (0), Write B (1)
        // Tick 1: Read B (1), Write A (0)
        const isOdd = this.tick % 2 === 1;
        return {
            read: cached.baseIdx + (isOdd ? 1 : 0),
            write: cached.baseIdx + (isOdd ? 0 : 1)
        };
    }

    public get currentTick(): number {
        return this.tick;
    }
}
