import { IKernel } from './IKernel';
import { ComputeContext } from './ComputeContext';

/**
 * Kernel for adding Forces (Gravitational, Buoyancy, etc.).
 * Refactored for Phase 3: Uses ComputeContext for agnostic memory access.
 */
export class ForceKernel implements IKernel {
    public readonly metadata = {
        roles: {
            source: 'any',
            destination: 'any'
        }
    };

    public execute(
        views: Float32Array[],
        context: ComputeContext
    ): void {
        const { nx, ny, padding, pNx, scheme, indices } = context;

        const sourceFace = scheme.source;
        const destFace = scheme.destination || sourceFace;

        const srcIdx = indices[sourceFace].read;
        const dstIdx = indices[destFace].write;

        const src = views[srcIdx];
        const dst = views[dstIdx];

        const multiplier = (scheme.params?.multiplier as number) || 1.0;
        const dt = (scheme.params?.dt as number) || 0.1;

        for (let py = padding; py < ny + padding; py++) {
            for (let px = padding; px < nx + padding; px++) {
                const idx = py * pNx + px;

                let s = src[idx];
                let d = dst[idx];

                // Safety: Protect against Non-Finite values
                if (!Number.isFinite(s)) s = 0;
                if (!Number.isFinite(d)) d = 0;

                const delta = s * multiplier * dt;

                // Accumulate safely
                if (Number.isFinite(delta)) {
                    dst[idx] = d + delta;
                }
            }
        }
    }
}
