import { IKernel } from './IKernel';
import { ComputeContext } from './ComputeContext';

/**
 * Kernel for Semi-Lagrangian Advection.
 * Enhanced with Finite-Value safeguards and high-velocity damping.
 * Refactored for Phase 3: Uses ComputeContext for agnostic memory access.
 */
export class AdvectionKernel implements IKernel {
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
        const { nx, ny, padding, pNx, pNy, scheme, indices } = context;

        const sourceFace = scheme.source;
        const destFace = scheme.destination || sourceFace;
        const velocityField = scheme.field;

        if (!velocityField) return;

        const readIdx = indices[sourceFace].read;
        const writeIdx = indices[destFace].write;

        const velXIdx = indices[`${velocityField}X`]?.read ?? indices['vx']?.read;
        const velYIdx = indices[`${velocityField}Y`]?.read ?? indices['vy']?.read;

        if (velXIdx === undefined || velYIdx === undefined) return;

        const src = views[readIdx];
        const dst = views[writeIdx];
        const vx = views[velXIdx];
        const vy = views[velYIdx];

        const dt = (scheme.params?.dt as number) || 0.1;
        const dissipation = (scheme.params?.dissipation as number) || 1.0;

        // CFL Safety: Sample from neighboring ghost cells (width = padding)
        const maxVel = padding / (dt + 1e-6);

        for (let py = padding; py < ny + padding; py++) {
            for (let px = padding; px < nx + padding; px++) {
                const idx = py * pNx + px;

                const worldX = px - padding;
                const worldY = py - padding;

                let vValX = vx[idx];
                let vValY = vy[idx];

                // Stability: Protect against Non-Finite and Overflow
                if (!Number.isFinite(vValX)) vValX = 0;
                if (!Number.isFinite(vValY)) vValY = 0;

                // CFL Clamp
                const clampedVX = Math.max(-maxVel, Math.min(maxVel, vValX));
                const clampedVY = Math.max(-maxVel, Math.min(maxVel, vValY));

                const srcX = worldX - clampedVX * dt;
                const srcY = worldY - clampedVY * dt;

                const val = this.bilerp(srcX, srcY, src, nx, ny, padding, pNx, pNy) * dissipation;

                // Final Value Safety
                dst[idx] = Number.isFinite(val) ? val : 0;
            }
        }
    }

    private bilerp(x: number, y: number, buffer: Float32Array, nx: number, ny: number, padding: number, pNx: number, pNy: number): number {
        const minVal = -padding;
        const maxValX = nx + padding - 1;
        const maxValY = ny + padding - 1;

        let sx = Math.max(minVal, Math.min(x, maxValX - 0.001));
        let sy = Math.max(minVal, Math.min(y, maxValY - 0.001));

        if (!Number.isFinite(sx)) sx = 0;
        if (!Number.isFinite(sy)) sy = 0;

        const x0 = Math.floor(sx);
        const y0 = Math.floor(sy);

        const tx = sx - x0;
        const ty = sy - y0;

        const px0 = x0 + padding;
        const py0 = y0 + padding;
        const px1 = px0 + 1;
        const py1 = py0 + 1;

        const v00 = buffer[py0 * pNx + px0];
        const v10 = buffer[py0 * pNx + px1];
        const v01 = buffer[py1 * pNx + px0];
        const v11 = buffer[py1 * pNx + px1];

        const res = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
        return Number.isFinite(res) ? res : 0;
    }
}
