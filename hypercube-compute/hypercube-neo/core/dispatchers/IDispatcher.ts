import { VirtualChunk } from '../topology/GridAbstractions';
import { NumericalScheme } from '../types';

/**
 * Interface for simulation dispatchers (Mono or Parallel).
 */
export interface IDispatcher {
    /**
     * Executes the numerical rules for all chunks at time 't'.
     */
    dispatch(t: number): Promise<void> | void;
}
