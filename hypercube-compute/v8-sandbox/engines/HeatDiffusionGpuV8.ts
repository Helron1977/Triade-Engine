import { AbstractGpuEngine } from './AbstractGpuEngine';
import { HeatDiffusionV8 } from './HeatDiffusionV8';
import { V8_UNIFORM_WGSL } from '../core/UniformPresets';

export class HeatDiffusionGpuV8 extends AbstractGpuEngine {
    constructor() {
        super(HeatDiffusionV8);
    }

    protected getShaderSource(): string {
        const T_IDX = this.getFaceIndex('Temperature');
        const OBST_IDX = this.getFaceIndex('Obstacles');

        return `
            ${V8_UNIFORM_WGSL}

            @group(0) @binding(0) var<storage, read> cube_in: array<f32>;
            @group(0) @binding(1) var<storage, read_write> cube_out: array<f32>;

            @compute @workgroup_size(16, 16, 1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                if (id.x >= u32(u.nx) || id.y >= u32(u.ny)) { return; }

                let nx = u32(u.nx);
                let ny = u32(u.ny);
                let idx = id.z * nx * ny + id.y * nx + id.x;
                let stride = u32(u.stride);
                
                // V8 GPU Rule: Standard Ping-Pong uses a SINGLE face index (0)
                // because AbstractGpuEngine already swaps the physical buffers.
                let face_in = ${T_IDX}u * stride;
                let face_out = ${T_IDX}u * stride;
                let face_obst = ${OBST_IDX}u * stride;

                // 1. MIRRORING (Crucial for V8 Ping-Pong Persistence)
                cube_out[face_obst + idx] = cube_in[face_obst + idx];

                // 2. OBSTACLE CHECK
                if (cube_in[face_obst + idx] > 0.5) {
                    cube_out[face_out + idx] = 0.0;
                    return;
                }

                // 3. GLOBAL BOUNDARY LOGIC (Inlets / Walls)
                // Left
                if (u.offX == 0.0 && id.x == 0u) {
                    if (u.lRole == 1.0) { cube_out[face_out + idx] = u.lVal; return; }
                    if (u.lRole == 3.0) { cube_out[face_out + idx] = 0.0; return; }
                }
                // Right
                if (id.x == nx - 1u) {
                    if (u.rRole == 1.0) { cube_out[face_out + idx] = u.rVal; return; }
                    if (u.rRole == 3.0) { cube_out[face_out + idx] = 0.0; return; }
                }
                // Top
                if (u.offY == 0.0 && id.y == 0u) {
                    if (u.tRole == 1.0) { cube_out[face_out + idx] = u.tVal; return; }
                    if (u.tRole == 3.0) { cube_out[face_out + idx] = 0.0; return; }
                }
                // Bottom
                if (id.y == ny - 1u) {
                    if (u.bRole == 1.0) { cube_out[face_out + idx] = u.bVal; return; }
                    if (u.bRole == 3.0) { cube_out[face_out + idx] = 0.0; return; }
                }

                // 4. INTER-CHUNK GHOST CELLS (Default: Preserve synced value)
                if (id.x == 0u || id.x == nx - 1u || id.y == 0u || id.y == ny - 1u) {
                    cube_out[face_out + idx] = cube_in[face_in + idx];
                    return;
                }

                // 5. PHYSICAL LOGIC (9-Point Laplacian for WOW Smoothness)
                let val = cube_in[face_in + idx];
                let diffusionRate = u.params[0].x; 

                // Weights for isotropy: Orthogonal (0.5), Diagonal (0.25)
                let laplacian = (
                    (cube_in[face_in + idx - 1u] + cube_in[face_in + idx + 1u] +
                     cube_in[face_in + idx - nx] + cube_in[face_in + idx + nx]) * 0.5 +
                    (cube_in[face_in + idx - nx - 1u] + cube_in[face_in + idx - nx + 1u] +
                     cube_in[face_in + idx + nx - 1u] + cube_in[face_in + idx + nx + 1u]) * 0.25
                ) - 3.0 * val; 

                cube_out[face_out + idx] = val + diffusionRate * laplacian;
            }
        `;
    }
}
