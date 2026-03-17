import { NumericalScheme } from '../types';
import { IKernel } from './IKernel';

/**
 * Resolved view for a kernel execution, providing access to both read and write buffers.
 */
export interface ViewPair {
    read: Float32Array;
    write: Float32Array;
}

/**
 * Resolved views for a kernel execution.
 */
export interface BoundViews {
    source: ViewPair;
    destination: ViewPair;
    obstacles: Float32Array | null;
    auxiliary: (ViewPair | null)[];
}

/**
 * The KernelBinder is responsible for mapping logical roles 
 * defined in a kernel's metadata to physical memory views.
 */
export class KernelBinder {
    /**
     * Resolves all required views for a kernel based on the scheme and current parity indices.
     */
    static bind(
        kernel: IKernel,
        scheme: NumericalScheme,
        views: Float32Array[],
        indices: Record<string, { read: number; write: number }>
    ): BoundViews {
        const roles = kernel.metadata.roles;
        
        // 1. Resolve source (Primary read face)
        const sourceName = scheme.source || roles.source;
        const sourceIndices = (sourceName && indices[sourceName]) ? indices[sourceName] : null;
        const sourcePair = {
            read: sourceIndices ? views[sourceIndices.read] : views[0],
            write: sourceIndices ? views[sourceIndices.write] : views[0]
        };

        // 2. Resolve destination (Primary write face)
        let destName = scheme.destination;
        if (!destName && sourceName && indices[sourceName]) destName = sourceName;
        if (!destName) destName = roles.destination;

        const destIndices = (destName && indices[destName]) ? indices[destName] : null;
        const destPair = {
            read: destIndices ? views[destIndices.read] : sourcePair.read,
            write: destIndices ? views[destIndices.write] : sourcePair.write
        };

        // 3. Resolve obstacles (Optional mask)
        const obsName = (scheme.params?.obstacles_face as string) || roles.obstacles || 'obstacles';
        const obsData = (obsName && indices[obsName]) ? views[indices[obsName].read] : null;

        // 4. Resolve auxiliary faces
        const auxViews: (ViewPair | null)[] = [];
        if (roles.auxiliary) {
            for (const auxRole of roles.auxiliary) {
                const paramKey = `${auxRole}_face`;
                const faceName = (scheme.params?.[paramKey] as string) || auxRole;
                
                if (indices[faceName]) {
                    auxViews.push({
                        read: views[indices[faceName].read],
                        write: views[indices[faceName].write]
                    });
                } else {
                    auxViews.push(null);
                }
            }
        }

        return {
            source: sourcePair,
            destination: destPair,
            obstacles: obsData,
            auxiliary: auxViews
        };
    }
}
