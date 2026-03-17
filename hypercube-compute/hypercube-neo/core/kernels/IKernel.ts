import { NumericalScheme } from '../types';
import { VirtualChunk } from '../topology/GridAbstractions';
import { ComputeContext } from './ComputeContext';

/**
 * Metadata defining the data roles required by a kernel.
 */
export interface KernelMetadata {
    roles: {
        source: string;
        destination: string;
        obstacles?: string;
        auxiliary?: string[];
    };
}

/**
 * Interface for all Hypercube Neo physics kernels.
 */
export interface IKernel {
    readonly metadata: KernelMetadata;

    /**
     * Executes the kernel logic for a specific chunk and scheme.
     * @param views Raw face views from MasterBuffer.
     * @param context Immutable context containing all runtime parameters.
     */
    execute(
        views: Float32Array[],
        context: ComputeContext
    ): void;
}
