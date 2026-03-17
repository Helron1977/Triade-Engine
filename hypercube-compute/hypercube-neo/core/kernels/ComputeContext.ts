import { NumericalScheme } from '../types';
import { VirtualChunk } from '../topology/GridAbstractions';

/**
 * ComputeContext
 * Encapsulates the runtime environment for a kernel execution.
 * Decouples physics logic from low-level memory stride and padding calculations.
 * @pattern Context Object
 */
export interface ComputeContext {
    /** Local grid dimensions of the current chunk (without padding) */
    readonly nx: number;
    readonly ny: number;
    
    /** Memory stride (physical width including padding) */
    readonly pNx: number;

    /** Physical height including padding */
    readonly pNy: number;
    
    /** Physical padding/ghost cells count */
    readonly padding: number;
    
    /** Current numerical scheme/rule being executed */
    readonly scheme: NumericalScheme;
    
    /** Resolved face indices for the current parity */
    readonly indices: Record<string, { read: number; write: number }>;
    
    /** Global simulation variables */
    readonly params: {
        time: number;
        tick: number;
        [key: string]: any;
    };
    
    /** Topology information for the current chunk */
    readonly chunk: VirtualChunk;
    
    /** Global grid configuration */
    readonly gridConfig: any;
}
