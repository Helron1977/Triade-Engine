import { IKernel } from './IKernel';
import { NumericalScheme } from '../types';
import { VirtualChunk } from '../topology/GridAbstractions';
import { TopologyResolver, BoundaryRoleID } from '../topology/TopologyResolver';
import { KernelBinder } from './KernelBinder';
import { ComputeContext } from './ComputeContext';

/**
 * NeoAeroKernel
 * ACHIEVED 1:1 Parity with legacy AerodynamicsEngine.ts.
 * Structure: Optimized for Stability and Zero-Copy safety via 'Self-Copy' world boundaries.
 */
export class NeoAeroKernel implements IKernel {
    public static readonly DEFAULT_OMEGA = 1.75;
    public static readonly DEFAULT_INFLOW_UX = 0.12;

    private topoResolver = new TopologyResolver();

    public readonly metadata = {
        roles: {
            source: 'f0',
            destination: 'f0',
            obstacles: 'obstacles',
            auxiliary: [
                'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8',
                'vx', 'vy', 'vorticity', 'smoke'
            ]
        }
    };

    public execute(
        views: Float32Array[],
        context: ComputeContext
    ): void {
        const { nx, ny, padding, pNx, pNy, scheme, indices, chunk, gridConfig } = context;

        const omega = (scheme.params?.omega as number) || NeoAeroKernel.DEFAULT_OMEGA;
        const om_1 = 1.0 - omega;

        // Resolve views via Binder (Nuclear Refactor Priority 2)
        const bound = KernelBinder.bind(this, scheme, views, indices);

        // ... populations ... (keep existing bindings)
        const f0_in = bound.source.read;
        const f0_out = bound.destination.write;
        const obstacles = bound.obstacles!; // Aero requires obstacles

        const f1_in = bound.auxiliary[0]!.read, f1_out = bound.auxiliary[0]!.write;
        const f2_in = bound.auxiliary[1]!.read, f2_out = bound.auxiliary[1]!.write;
        const f3_in = bound.auxiliary[2]!.read, f3_out = bound.auxiliary[2]!.write;
        const f4_in = bound.auxiliary[3]!.read, f4_out = bound.auxiliary[3]!.write;
        const f5_in = bound.auxiliary[4]!.read, f5_out = bound.auxiliary[4]!.write;
        const f6_in = bound.auxiliary[5]!.read, f6_out = bound.auxiliary[5]!.write;
        const f7_in = bound.auxiliary[6]!.read, f7_out = bound.auxiliary[6]!.write;
        const f8_in = bound.auxiliary[7]!.read, f8_out = bound.auxiliary[7]!.write;

        const ux_in = bound.auxiliary[8]!.read, ux_out = bound.auxiliary[8]!.write;
        const uy_in = bound.auxiliary[9]!.read, uy_out = bound.auxiliary[9]!.write;
        const curl_out = bound.auxiliary[10]!.write;
        const smoke_in = bound.auxiliary[11]!.read, smoke_out = bound.auxiliary[11]!.write;

        // Resolve topology roles with caching
        const vChunkAny = chunk as any;
        if (!vChunkAny._topoCache) {
            vChunkAny._topoCache = this.topoResolver.resolve(chunk, gridConfig.chunks, gridConfig.boundaries);
            
            // Also pre-calculate worldY0 while we're at it
            let y0 = 0;
            const cList = gridConfig.chunksList || gridConfig.vGrid?.chunks;
            if (cList) {
                for (const c of cList) {
                    if (c.x === chunk.x && c.z === chunk.z && c.y < chunk.y) y0 += c.localDimensions.ny;
                }
            }
            vChunkAny._worldY0 = y0;
        }

        const topo = vChunkAny._topoCache;
        const isLeft = topo.leftRole !== BoundaryRoleID.CONTINUITY;
        const isRight = topo.rightRole !== BoundaryRoleID.CONTINUITY;
        const isTop = topo.topRole !== BoundaryRoleID.CONTINUITY;
        const isBottom = topo.bottomRole !== BoundaryRoleID.CONTINUITY;

        const inflowUx = (scheme.params?.inflowUx as number) || NeoAeroKernel.DEFAULT_INFLOW_UX;

        // --- STEP 1: LBM ---
        for (let py = padding; py < ny + padding; py++) {
            const isWorldTop = isTop && py === padding;
            const isWorldBottom = isBottom && (py === ny + padding - 1);

            for (let px = padding; px < nx + padding; px++) {
                const i = py * pNx + px;

                // 1. OBSTACLES (Static Priority)
                const obs = obstacles[i];
                if (obs > 0.99) {
                    ux_out[i] = 0; uy_out[i] = 0; 
                    // Smoke is preserved to allow obstacles to act as persistent sources
                    smoke_out[i] = smoke_in[i]; 
                    f0_out[i] = 4.0 / 9.0;
                    f1_out[i] = 1.0 / 9.0; f2_out[i] = 1.0 / 9.0; f3_out[i] = 1.0 / 9.0; f4_out[i] = 1.0 / 9.0;
                    f5_out[i] = 1.0 / 36.0; f6_out[i] = 1.0 / 36.0; f7_out[i] = 1.0 / 36.0; f8_out[i] = 1.0 / 36.0;
                    continue;
                }

                // 2. WORLD BCs & ZERO-COPY STABILIZATION
                const isWorldLeft = isLeft && px === padding;
                const isWorldRight = isRight && px === nx + padding - 1;

                if (isWorldLeft || isWorldRight || isWorldTop || isWorldBottom) {
                    // INFLOW (Exclusive to Left Boundary if set)
                    if (isWorldLeft && topo.leftRole === BoundaryRoleID.INFLOW) {
                        let scale = 1.0;
                        const ly = py - padding;
                        if (isTop && topo.topRole === BoundaryRoleID.WALL && ly < 16) scale = ly / 16.0;
                        if (isBottom && topo.bottomRole === BoundaryRoleID.WALL && ly > ny - 17) scale = (ny - 1 - ly) / 16.0;
                        const inUx = inflowUx * scale; const inRho = 1.0;
                        ux_out[i] = inUx; uy_out[i] = 0;
                        const u_sq_15 = 1.5 * (inUx * inUx);
                        f0_out[i] = (4.0 / 9.0) * inRho * (1.0 - u_sq_15);
                        const cu1 = inUx; f1_out[i] = (1.0 / 9.0) * inRho * (1.0 + 3.0 * cu1 + 4.5 * cu1 * cu1 - u_sq_15);
                        const cu2 = 0; f2_out[i] = (1.0 / 9.0) * inRho * (1.0 + 3.0 * cu2 + 4.5 * cu2 * cu2 - u_sq_15);
                        const cu3 = -inUx; f3_out[i] = (1.0 / 9.0) * inRho * (1.0 + 3.0 * cu3 + 4.5 * cu3 * cu3 - u_sq_15);
                        const cu4 = 0; f4_out[i] = (1.0 / 9.0) * inRho * (1.0 + 3.0 * cu4 + 4.5 * cu4 * cu4 - u_sq_15);
                        const cu5 = inUx; f5_out[i] = (1.0 / 36.0) * inRho * (1.0 + 3.0 * cu5 + 4.5 * cu5 * cu5 - u_sq_15);
                        const cu6 = -inUx; f6_out[i] = (1.0 / 36.0) * inRho * (1.0 + 3.0 * cu6 + 4.5 * cu6 * cu6 - u_sq_15);
                        const cu7 = -inUx; f7_out[i] = (1.0 / 36.0) * inRho * (1.0 + 3.0 * cu7 + 4.5 * cu7 * cu7 - u_sq_15);
                        const cu8 = inUx; f8_out[i] = (1.0 / 36.0) * inRho * (1.0 + 3.0 * cu8 + 4.5 * cu8 * cu8 - u_sq_15);
                    } else {
                        // SELF-COPY: Replicate legacy 'skip' behavior for walls/unknowns.
                        ux_out[i] = ux_in[i]; uy_out[i] = uy_in[i]; smoke_out[i] = smoke_in[i];
                        f0_out[i] = f0_in[i]; f1_out[i] = f1_in[i]; f2_out[i] = f2_in[i]; f3_out[i] = f3_in[i]; f4_out[i] = f4_in[i];
                        f5_out[i] = f5_in[i]; f6_out[i] = f6_in[i]; f7_out[i] = f7_in[i]; f8_out[i] = f8_in[i];
                    }
                    continue;
                }

                // OUTFLOW (at Global Col NX-2 if Right is Outflow)
                if (isRight && px === nx + padding - 2 && topo.rightRole === BoundaryRoleID.OUTFLOW) {
                    const prev = i - 1;
                    const uH = ux_out[prev]; const vH = uy_out[prev];
                    const u2 = 1.5 * (uH * uH + vH * vH);
                    const cu1 = uH; f1_out[i] = (1.0 / 9.0) * (1.0 + 3.0 * cu1 + 4.5 * cu1 * cu1 - u2);
                    const cu2 = vH; f2_out[i] = (1.0 / 9.0) * (1.0 + 3.0 * cu2 + 4.5 * cu2 * cu2 - u2);
                    const cu3 = -uH; f3_out[i] = (1.0 / 9.0) * (1.0 + 3.0 * cu3 + 4.5 * cu3 * cu3 - u2);
                    const cu4 = -vH; f4_out[i] = (1.0 / 9.0) * (1.0 + 3.0 * cu4 + 4.5 * cu4 * cu4 - u2);
                    const cu5 = uH + vH; f5_out[i] = (1.0 / 36.0) * (1.0 + 3.0 * cu5 + 4.5 * cu5 * cu5 - u2);
                    const cu6 = -uH + vH; f6_out[i] = (1.0 / 36.0) * (1.0 + 3.0 * cu6 + 4.5 * cu6 * cu6 - u2);
                    const cu7 = -uH - vH; f7_out[i] = (1.0 / 36.0) * (1.0 + 3.0 * cu7 + 4.5 * cu7 * cu7 - u2);
                    const cu8 = uH - vH; f8_out[i] = (1.0 / 36.0) * (1.0 + 3.0 * cu8 + 4.5 * cu8 * cu8 - u2);
                    f0_out[i] = (4.0 / 9.0) * (1.0 - u2); ux_out[i] = uH; uy_out[i] = vH;
                    continue;
                }

                // FLUID INTERIOR (Canonical Form for 1:1 Parity)
                const pf0 = f0_in[i];
                const n1 = i - 1, n2 = i - pNx, n3 = i + 1, n4 = i + pNx;
                const n5 = i - pNx - 1, n6 = i - pNx + 1, n7 = i + pNx + 1, n8 = i + pNx - 1;
                const o1 = obstacles[n1], o2 = obstacles[n2], o3 = obstacles[n3], o4 = obstacles[n4];
                const o5 = obstacles[n5], o6 = obstacles[n6], o7 = obstacles[n7], o8 = obstacles[n8];

                const pf1 = f1_in[n1] * (1.0 - o1) + f3_in[i] * o1;
                const pf2 = f2_in[n2] * (1.0 - o2) + f4_in[i] * o2;
                const pf3 = f3_in[n3] * (1.0 - o3) + f1_in[i] * o3;
                const pf4 = f4_in[n4] * (1.0 - o4) + f2_in[i] * o4;
                const pf5 = f5_in[n5] * (1.0 - o5) + f7_in[i] * o5;
                const pf6 = f6_in[n6] * (1.0 - o6) + f8_in[i] * o6;
                const pf7 = f7_in[n7] * (1.0 - o7) + f5_in[i] * o7;
                const pf8 = f8_in[n8] * (1.0 - o8) + f6_in[i] * o8;

                const rho = pf0 + pf1 + pf2 + pf3 + pf4 + pf5 + pf6 + pf7 + pf8;
                const invRho = 1.0 / rho;
                const ux = ((pf1 + pf5 + pf8) - (pf3 + pf6 + pf7)) * invRho;
                const uy = ((pf2 + pf5 + pf6) - (pf4 + pf7 + pf8)) * invRho;
                ux_out[i] = ux; uy_out[i] = uy;

                const ux3 = 3.0 * ux; const uy3 = 3.0 * uy;
                const ux2 = 4.5 * ux * ux; const uy2 = 4.5 * uy * uy;
                const u_sq_15 = 1.5 * (ux * ux + uy * uy);
                const rOmega = rho * omega;

                f0_out[i] = pf0 * om_1 + ((4.0 / 9.0) * rho * (1.0 - u_sq_15)) * omega;
                f1_out[i] = pf1 * om_1 + ((1.0 / 9.0) * rOmega * (1.0 + ux3 + ux2 - u_sq_15));
                f2_out[i] = pf2 * om_1 + ((1.0 / 9.0) * rOmega * (1.0 + uy3 + uy2 - u_sq_15));
                f3_out[i] = pf3 * om_1 + ((1.0 / 9.0) * rOmega * (1.0 - ux3 + ux2 - u_sq_15));
                f4_out[i] = pf4 * om_1 + ((1.0 / 9.0) * rOmega * (1.0 - uy3 + uy2 - u_sq_15));

                const uxy3 = ux3 + uy3;
                const uxy2 = 4.5 * (ux + uy) * (ux + uy);
                f5_out[i] = pf5 * om_1 + ((1.0 / 36.0) * rOmega * (1.0 + uxy3 + uxy2 - u_sq_15));

                const unxy3 = -ux3 + uy3;
                const unxy2 = 4.5 * (-ux + uy) * (-ux + uy);
                f6_out[i] = pf6 * om_1 + ((1.0 / 36.0) * rOmega * (1.0 + unxy3 + unxy2 - u_sq_15));

                const unnxy3 = -ux3 - uy3;
                const unnxy2 = 4.5 * (-ux - uy) * (-ux - uy);
                f7_out[i] = pf7 * om_1 + ((1.0 / 36.0) * rOmega * (1.0 + unnxy3 + unnxy2 - u_sq_15));

                const uany3 = ux3 - uy3;
                const uany2 = 4.5 * (ux - uy) * (ux - uy);
                f8_out[i] = pf8 * om_1 + ((1.0 / 36.0) * rOmega * (1.0 + uany3 + uany2 - u_sq_15));
            }
        }

        // --- STEP 2: VORTICITY ---
        if (true) { // Always update for 1:1 visual match with legacy
            const vPyStart = isTop ? padding + 1 : padding;
            const vPyEnd = ny + padding - (isBottom ? 1 : 0);
            const vPxStart = isLeft ? padding + 1 : padding;
            const vPxEnd = nx + padding - (isRight ? 1 : 0);

            for (let py = vPyStart; py < vPyEnd; py++) {
                const row = py * pNx;
                const rowPrev = row - pNx;
                const rowNext = row + pNx;
                for (let px = vPxStart; px < vPxEnd; px++) {
                    const i = row + px;
                    const dUy_dx = (uy_out[i + 1] - uy_out[i - 1]) * 0.5;
                    const dUx_dy = (ux_out[rowNext + px] - ux_out[rowPrev + px]) * 0.5;
                    curl_out[i] = dUy_dx - dUx_dy;
                }
            }
        }

        // --- STEP 3: SMOKE ---
        for (let py = padding; py < ny + padding; py++) {
            for (let px = padding; px < nx + padding; px++) {
                const i = py * pNx + px;
                const obs = obstacles[i];
                if (obs > 0.99) { 
                    smoke_out[i] = smoke_in[i]; 
                    continue; 
                }

                const vx = ux_out[i]; const vy = uy_out[i];
                const sx = px - vx; const sy = py - vy;
                let x0 = Math.floor(sx); let y0 = Math.floor(sy);

                if (x0 < 0) x0 = 0; if (x0 > pNx - 2) x0 = pNx - 2;
                if (y0 < 0) y0 = 0; if (y0 > pNy - 2) y0 = pNy - 2;

                const x1 = x0 + 1; const y1 = y0 + 1;
                const fx = sx - x0; const fy = sy - y0;

                const row0 = y0 * pNx; const row1 = y1 * pNx;
                const val00 = smoke_in[row0 + x0]; const val10 = smoke_in[row0 + x1];
                const val01 = smoke_in[row1 + x0]; const val11 = smoke_in[row1 + x1];
                const raw = (val00 * (1 - fx) + val10 * fx) * (1 - fy) + (val01 * (1 - fx) + val11 * fx) * fy;
                const neighborAvg = (smoke_in[i - 1] + smoke_in[i + 1] + smoke_in[i - pNx] + smoke_in[i + pNx]) * 0.25;
                smoke_out[i] = (raw * 0.995 + neighborAvg * 0.005) * 0.9999;

                if (isLeft && topo.leftRole === BoundaryRoleID.INFLOW && px === padding) {
                    const worldY = (vChunkAny._worldY0 || 0) + (py - padding);
                    const pitch = 25; // Matching legacy visual
                    if ((worldY + 2) % pitch <= 2) smoke_out[i] = 1.0;
                }
            }
        }
    }
}
