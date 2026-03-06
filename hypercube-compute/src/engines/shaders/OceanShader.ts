export const OCEAN_SHADER_WGSL = `
struct Uniforms {
    nx: u32, ny: u32, nz: u32, strideFace: u32,
    tau_0: f32, smagorinsky: f32, cflLimit: f32, isClosed: f32,
    bioDiffusion: f32, bioGrowth: f32, parity: u32
};

@group(0) @binding(0) var<storage, read_write> cube_in: array<f32>;
@group(0) @binding(1) var<storage, read_write > cube_out: array<f32>;
@group(0) @binding(2) var<uniform>u: Uniforms;

const cx = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const cy = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const w = array<f32, 9>(0.444444, 0.111111, 0.111111, 0.111111, 0.111111, 0.027778, 0.027778, 0.027778, 0.027778);
const opp = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);

@compute @workgroup_size(16, 16, 1)
fn compute_lbm(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x); let y = i32(id.y); let z = i32(id.z);
    if (x < 1 || y < 1 || x >= i32(u.nx) - 1 || y >= i32(u.ny) - 1) { return; }

    let idx = u32(z) * u.nx * u.ny + u32(y) * u.nx + u32(x);
    let stride = u.strideFace;
    let f_in = u.parity * 9u * stride;
    let f_out = (1u - u.parity) * 9u * stride;

    let obst = cube_in[18u * stride + idx];
    cube_out[18u * stride + idx] = obst; 

    if (obst > 0.5) {
        for (var k = 0u; k < 9u; k = k + 1u) {
            cube_out[f_out + k * stride + idx] = w[k];
        }
        return;
    }

    var pf: array<f32, 9>;
    pf[0] = cube_in[f_in + 0u * stride + idx];

    for (var k = 1u; k < 9u; k = k + 1u) {
        var sx = x - cx[k];
        var sy = y - cy[k];
        if (sx < 0 || sx >= i32(u.nx) || sy < 0 || sy >= i32(u.ny)) {
            pf[k] = cube_in[f_in + opp[k] * stride + idx];
        } else {
            let n_idx = u32(z) * u.nx * u.ny + u32(sy) * u.nx + u32(sx);
            if (cube_in[18u * stride + n_idx] > 0.5) {
                pf[k] = cube_in[f_in + opp[k] * stride + idx];
            } else {
                pf[k] = cube_in[f_in + k * stride + n_idx];
            }
        }
    }

    var r = 0.0;
    for (var k = 0u; k < 9u; k = k + 1u) { r += pf[k]; }
    
    var pf_temp = pf; // Utiliser une copie pour la remise à zéro si nécessaire
    if (r < 0.1 || r > 10.0) { 
        r = 1.0; 
        for (var k = 0u; k < 9u; k = k + 1u) { pf_temp[k] = w[k]; }
    }
    pf = pf_temp;

    var vx = (pf[1] + pf[5] + pf[8] - (pf[3] + pf[6] + pf[7])) / (r + 1e-6);
    var vy = (pf[2] + pf[5] + pf[6] - (pf[4] + pf[7] + pf[8])) / (r + 1e-6);

    let v_mag = sqrt(vx * vx + vy * vy);
    if (v_mag > u.cflLimit) {
        vx *= u.cflLimit / v_mag;
        vy *= u.cflLimit / v_mag;
    }

    cube_out[22u * stride + idx] = r;
    cube_out[19u * stride + idx] = vx;
    cube_out[20u * stride + idx] = vy;

    let u2_15 = 1.5 * (vx * vx + vy * vy);
    let inv_tau = 1.0 / (u.tau_0 + 1e-5);

    for (var k = 0u; k < 9u; k = k + 1u) {
        let cu = 3.0 * (f32(cx[k]) * vx + f32(cy[k]) * vy);
        let feq = w[k] * r * (1.0 + cu + 0.5 * cu * cu - u2_15);
        cube_out[f_out + k * stride + idx] = pf[k] - (pf[k] - feq) * inv_tau;
    }

    let xP = u32(x + 1); let xM = u32(x - 1);
    let yP = u32(y + 1); let yM = u32(y - 1);
    let dUy_dx = (cube_in[20u * stride + u32(z) * u.nx * u.ny + u32(y) * u.nx + xP]- cube_in[20u * stride + u32(z) * u.nx * u.ny + u32(y) * u.nx + xM]) * 0.5;
    let dUx_dy = (cube_in[19u * stride + u32(z) * u.nx * u.ny + yP * u.nx + u32(x)]- cube_in[19u * stride + u32(z) * u.nx * u.ny + yM * u.nx + u32(x)]) * 0.5;
    cube_out[21u * stride + idx] = dUy_dx - dUx_dy;
}

@compute @workgroup_size(16, 16, 1)
fn compute_bio(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x); let y = i32(id.y); let z = i32(id.z);
    if (x < 1 || y < 1 || x >= i32(u.nx) - 1 || y >= i32(u.ny) - 1) { return; }

    let idx = u32(z) * u.nx * u.ny + u32(y) * u.nx + u32(x);
    let stride = u.strideFace;
    let b_in = (23u + u.parity) * stride;
    let b_out = (23u + (1u - u.parity)) * stride;

    let bo = cube_in[b_in + idx];
    let lap = cube_in[b_in + idx - 1u] + cube_in[b_in + idx + 1u]+ cube_in[b_in + idx - u.nx] + cube_in[b_in + idx + u.nx] - 4.0 * bo;

    let ux = cube_in[19u* stride + idx];
    let uy = cube_in[20u* stride + idx];

    let ax = clamp(f32(x) - ux * 0.8, 1.0, f32(u.nx) - 2.0);
    let ay = clamp(f32(y) - uy * 0.8, 1.0, f32(u.ny) - 2.0);
    let ix = u32(ax); let iy = u32(ay);
    let fx = ax - f32(ix); let fy = ay - f32(iy);

    let base = u32(z) * u.nx * u.ny;
    let v00 = cube_in[b_in + base + iy * u.nx + ix];
    let v10 = cube_in[b_in + base + iy * u.nx + ix + 1u];
    let v01 = cube_in[b_in + base + (iy + 1u)* u.nx + ix];
    let v11 = cube_in[b_in + base + (iy + 1u)* u.nx + ix + 1u];

    let adv = (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * ((1.0 - fx) * v01 + fx * v11);
    cube_out[b_out + idx] = clamp(adv + u.bioDiffusion * lap + u.bioGrowth * bo * (1.0 - bo), 0.0, 1.0);
}
`;
