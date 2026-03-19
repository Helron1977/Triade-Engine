struct Params {
    nx: u32,
    ny: u32,
    chunksX: u32,
    chunksY: u32,
    omega: f32, // tau_0 for ocean
    inflowUx: f32, // cflLimit for ocean
    time: f32,
    currentTick: u32,
    chunkX: u32,
    chunkY: u32,
    strideFace: u32,
    numObjects: u32,
    nz: u32,
    // [13-21] Unified Semantic Indices
    obsIdx: u32,
    vxReadIdx: u32,
    vyReadIdx: u32,
    rhoReadIdx: u32,
    bioReadIdx: u32,
    vxWriteIdx: u32,
    vyWriteIdx: u32,
    rhoWriteIdx: u32,
    bioWriteIdx: u32,
    fBase: u32,
    // [23+] Extensions
    bioDiffusion: f32,
    bioGrowth: f32,
    _pad25: u32,
    _pad26: u32,
    _pad27: u32,
    _pad28: u32,
    _pad29: u32,
    _pad30: u32,
    _pad31: u32,
    objects: array<GpuObject, 8> 
};

struct GpuObject {
    pos: vec2<f32>,
    dim: vec2<f32>,
    isObstacle: f32,
    biology: f32,
    objType: u32,
    rho: f32
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

fn get_f_idx(d: u32, parity: u32, strideFace: u32, fBase: u32, i: u32) -> u32 {
    return (fBase + d * 2u + parity) * strideFace + i;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = params.nx;
    let ny = params.ny;
    if (id.x >= nx || id.y >= ny) { return; }

    let px = id.x + 1u;
    let py = id.y + 1u;
    let pNx = nx + 2u;
    let pNy = ny + 2u;
    let nz = params.nz;
    
    // Top-Layer Indexing: (z * NY + (NY-1)) * NX + x
    let gy = select(0u, params.ny - 1u, params.ny > 1u);
    let i = (id.y * params.ny + gy) * nx + id.x;
    
    // Padded indices for neighborhood (2D slice logic)
    let i_padded = py * pNx + px;

    let strideFace = params.strideFace;
    let readParity = params.currentTick % 2u;
    let writeParity = (params.currentTick + 1u) % 2u;

    let dx_lbm = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
    let dy_lbm = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
    let opp = array<u32, 9>(0, 3, 4, 1, 2, 7, 8, 5, 6);
    let w_lbm = array<f32, 9>(4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);

    let obsIdx = params.obsIdx * strideFace + i;
    let vxIdx = params.vxReadIdx * strideFace + i;
    let vyIdx = params.vyReadIdx * strideFace + i;
    let rhoIdx = params.rhoReadIdx * strideFace + i;
    let bioReadIdx = params.bioReadIdx * strideFace + i;
    let bioWriteIdx = params.bioWriteIdx * strideFace + i;

    var bioInjection = 0.0;
    var rhoInjection = 0.0;
    var obs = data[obsIdx];
    
    for (var j = 0u; j < params.numObjects; j = j + 1u) {
        let obj = params.objects[j];
        var inObj = false;
        if (obj.objType == 1u) { // Circle
            let r = obj.dim.x * 0.5;
            let ddx = f32(id.x) - obj.pos.x;
            let ddy = f32(id.y) - obj.pos.y;
            if (ddx*ddx + ddy*ddy <= r*r) { inObj = true; }
        } else if (obj.objType == 2u) { // Rect
            if (f32(id.x) >= obj.pos.x && f32(id.x) <= obj.pos.x + obj.dim.x &&
                f32(id.y) >= obj.pos.y && f32(id.y) <= obj.pos.y + obj.dim.y) { inObj = true; }
        }
        if (inObj) { 
            bioInjection = max(bioInjection, obj.biology);
            rhoInjection = max(rhoInjection, obj.rho);
            if (obj.isObstacle > 0.5) { obs = 1.0; }
        }
    }

    var pf = array<f32, 9>();
    let isLeft = (id.x == 0u); let isRight = (id.x == nx - 1u); 
    let isTop = (id.y == 0u); let isBottom = (id.y == ny - 1u); // id.y is z-axis here

    let dzStride = params.ny * params.nx;

    pf[0] = data[get_f_idx(0u, readParity, strideFace, params.fBase, i)];
    pf[1] = data[get_f_idx(1u, readParity, strideFace, params.fBase, i - 1u)];
    pf[2] = data[get_f_idx(2u, readParity, strideFace, params.fBase, i - dzStride)];
    pf[3] = data[get_f_idx(3u, readParity, strideFace, params.fBase, i + 1u)];
    pf[4] = data[get_f_idx(4u, readParity, strideFace, params.fBase, i + dzStride)];
    pf[5] = data[get_f_idx(5u, readParity, strideFace, params.fBase, i - dzStride - 1u)];
    pf[6] = data[get_f_idx(6u, readParity, strideFace, params.fBase, i - dzStride + 1u)];
    pf[7] = data[get_f_idx(7u, readParity, strideFace, params.fBase, i + dzStride + 1u)];
    pf[8] = data[get_f_idx(8u, readParity, strideFace, params.fBase, i + dzStride - 1u)];

    if (isLeft) { pf[1] = data[get_f_idx(3u, readParity, strideFace, params.fBase, i)]; pf[5] = data[get_f_idx(7u, readParity, strideFace, params.fBase, i)]; pf[8] = data[get_f_idx(6u, readParity, strideFace, params.fBase, i)]; }
    if (isRight) { pf[3] = data[get_f_idx(1u, readParity, strideFace, params.fBase, i)]; pf[6] = data[get_f_idx(8u, readParity, strideFace, params.fBase, i)]; pf[7] = data[get_f_idx(5u, readParity, strideFace, params.fBase, i)]; }
    if (isTop) { pf[2] = data[get_f_idx(4u, readParity, strideFace, params.fBase, i)]; pf[5] = data[get_f_idx(7u, readParity, strideFace, params.fBase, i)]; pf[6] = data[get_f_idx(8u, readParity, strideFace, params.fBase, i)]; }
    if (isBottom) { pf[4] = data[get_f_idx(2u, readParity, strideFace, params.fBase, i)]; pf[7] = data[get_f_idx(5u, readParity, strideFace, params.fBase, i)]; pf[8] = data[get_f_idx(6u, readParity, strideFace, params.fBase, i)]; }

    if (obs > 0.5) {
        var localRho = 0.0;
        for (var d = 0u; d < 9u; d = d + 1u) { data[get_f_idx(d, writeParity, strideFace, params.fBase, i)] = pf[opp[d]]; localRho += pf[opp[d]]; }
        data[rhoIdx] = localRho; data[vxIdx] = 0.0; data[vyIdx] = 0.0; data[bioWriteIdx] = 0.0;
        return;
    }

    var rho = 0.0; for (var d = 0u; d < 9u; d = d + 1u) { rho += pf[d]; }
    if (rhoInjection > rho) { rho = rhoInjection; for (var d = 0u; d < 9u; d = d + 1u) { pf[d] = w_lbm[d] * rho; } }
    
    var invRho = 1.0 / max(rho, 0.001);
    var vx = (pf[1] + pf[5] + pf[8] - (pf[3] + pf[6] + pf[7])) * invRho;
    var vy = (pf[2] + pf[5] + pf[6] - (pf[4] + pf[7] + pf[8])) * invRho;

    data[rhoIdx] = rho; data[vxIdx] = vx; data[vyIdx] = vy;
    let u2 = 1.5 * (vx * vx + vy * vy); let invTau = 1.0 / params.omega;
    
    for (var d = 0u; d < 9u; d = d + 1u) {
        let cu = 3.0 * (f32(dx_lbm[d]) * vx + f32(dy_lbm[d]) * vy);
        let feq = w_lbm[d] * rho * (1.0 + cu + 0.5 * cu * cu - u2);
        data[get_f_idx(d, writeParity, strideFace, params.fBase, i)] = pf[d] - invTau * (pf[d] - feq);
    }

    let b = data[bioReadIdx];
    let lap = data[bioReadIdx - 1u] + data[bioReadIdx + 1u] + data[bioReadIdx - pNx] + data[bioReadIdx + pNx] - 4.0 * b;
    let rawBio = b + params.bioDiffusion * lap + params.bioGrowth * b * (1.0 - b);
    data[bioWriteIdx] = clamp(max(rawBio, bioInjection), 0.0, 1.0);
}
