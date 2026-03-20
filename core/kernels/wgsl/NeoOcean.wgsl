struct Params {
    nx: u32,
    ny: u32,
    chunksX: u32,
    chunksY: u32,
    tau_0: f32,
    cflLimit: f32,
    time: f32,
    currentTick: u32,
    chunkX: u32,
    chunkY: u32,
    strideFace: u32,
    numObjects: u32,
    // Slots 12-21: Unified Semantic Indices
    obsIdx: u32,
    vxReadIdx: u32,
    vyReadIdx: u32,
    rhoIdx: u32,
    bioReadIdx: u32,
    vxWriteIdx: u32,
    vyWriteIdx: u32,
    rhoWriteIdx: u32, // rho is not ping-ponged normally but kept for layout
    bioWriteIdx: u32,
    fBase: u32,
    // Slots 22+: Extensions (Total 10 slots before objects at 32)
    bioDiffusion: f32,  // Slot 22
    bioGrowth: f32,      // Slot 23
    leftRole: u32,       // Slot 24 (Protected Topology Block)
    rightRole: u32,      // 25
    topRole: u32,        // 26
    bottomRole: u32,     // 27
    frontRole: u32,      // 28
    backRole: u32,       // 29
    _pad30: u32,         // 30
    _pad31: u32,         // 31
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

// Agnostic Face Accessor for LBM Populations (Always Ping-Ponged)
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
    let i = py * pNx + px;

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
    let rhoIdx = params.rhoIdx * strideFace + i;
    let bioReadIdx = params.bioReadIdx * strideFace + i;
    let bioWriteIdx = params.bioWriteIdx * strideFace + i;

    // --- Dynamic Inputs (Splash Injections) ---
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

    // --- LBM Physics (Bounce-Back Pull-Scheme) ---
    var pf = array<f32, 9>();
    
    let isLeft = (px == 1u);
    let isRight = (px == nx);
    let isTop = (py == 1u);
    let isBottom = (py == ny);

    pf[0] = data[get_f_idx(0u, readParity, strideFace, params.fBase, i)];
    pf[1] = data[get_f_idx(1u, readParity, strideFace, params.fBase, i - 1u)];
    pf[2] = data[get_f_idx(2u, readParity, strideFace, params.fBase, i - pNx)];
    pf[3] = data[get_f_idx(3u, readParity, strideFace, params.fBase, i + 1u)];
    pf[4] = data[get_f_idx(4u, readParity, strideFace, params.fBase, i + pNx)];
    pf[5] = data[get_f_idx(5u, readParity, strideFace, params.fBase, i - pNx - 1u)];
    pf[6] = data[get_f_idx(6u, readParity, strideFace, params.fBase, i - pNx + 1u)];
    pf[7] = data[get_f_idx(7u, readParity, strideFace, params.fBase, i + pNx + 1u)];
    pf[8] = data[get_f_idx(8u, readParity, strideFace, params.fBase, i + pNx - 1u)];

    if (isLeft) {
        pf[1] = data[get_f_idx(3u, readParity, strideFace, params.fBase, i)];
        pf[5] = data[get_f_idx(7u, readParity, strideFace, params.fBase, i)];
        pf[8] = data[get_f_idx(6u, readParity, strideFace, params.fBase, i)];
    }
    if (isRight) {
        pf[3] = data[get_f_idx(1u, readParity, strideFace, params.fBase, i)];
        pf[6] = data[get_f_idx(8u, readParity, strideFace, params.fBase, i)];
        pf[7] = data[get_f_idx(5u, readParity, strideFace, params.fBase, i)];
    }
    if (isTop) {
        pf[2] = data[get_f_idx(4u, readParity, strideFace, params.fBase, i)];
        pf[5] = data[get_f_idx(7u, readParity, strideFace, params.fBase, i)];
        pf[6] = data[get_f_idx(8u, readParity, strideFace, params.fBase, i)];
    }
    if (isBottom) {
        pf[4] = data[get_f_idx(2u, readParity, strideFace, params.fBase, i)];
        pf[7] = data[get_f_idx(5u, readParity, strideFace, params.fBase, i)];
        pf[8] = data[get_f_idx(6u, readParity, strideFace, params.fBase, i)];
    }

    if (obs > 0.5) {
        var localRho = 0.0;
        data[get_f_idx(0u, writeParity, strideFace, params.fBase, i)] = pf[0]; localRho += pf[0];
        data[get_f_idx(1u, writeParity, strideFace, params.fBase, i)] = pf[3]; localRho += pf[3];
        data[get_f_idx(2u, writeParity, strideFace, params.fBase, i)] = pf[4]; localRho += pf[4];
        data[get_f_idx(3u, writeParity, strideFace, params.fBase, i)] = pf[1]; localRho += pf[1];
        data[get_f_idx(4u, writeParity, strideFace, params.fBase, i)] = pf[2]; localRho += pf[2];
        data[get_f_idx(5u, writeParity, strideFace, params.fBase, i)] = pf[7]; localRho += pf[7];
        data[get_f_idx(6u, writeParity, strideFace, params.fBase, i)] = pf[8]; localRho += pf[8];
        data[get_f_idx(7u, writeParity, strideFace, params.fBase, i)] = pf[5]; localRho += pf[5];
        data[get_f_idx(8u, writeParity, strideFace, params.fBase, i)] = pf[6]; localRho += pf[6];
        data[rhoIdx] = localRho;
        data[vxIdx] = 0.0; data[vyIdx] = 0.0;
        data[bioWriteIdx] = 0.0;
        return;
    }

    var rho = 0.0;
    for (var d = 0u; d < 9u; d = d + 1u) { rho = rho + pf[d]; }
    
    if (rhoInjection > rho) {
        rho = rhoInjection;
        for (var d = 0u; d < 9u; d = d + 1u) { pf[d] = w_lbm[d] * rho; }
    }
    if (rho < 0.1 || rho > 10.0) { rho = 1.0; for (var d = 0u; d < 9u; d = d + 1u) { pf[d] = w_lbm[d]; } }

    var invRho = 1.0 / rho;
    var vx = (pf[1] + pf[5] + pf[8] - (pf[3] + pf[6] + pf[7])) * invRho;
    var vy = (pf[2] + pf[5] + pf[6] - (pf[4] + pf[7] + pf[8])) * invRho;

    let vMagSq = vx * vx + vy * vy;
    let cflSq = params.cflLimit * params.cflLimit;
    if (vMagSq > cflSq) {
        let scale = params.cflLimit / sqrt(vMagSq);
        vx = vx * scale;
        vy = vy * scale;
    }

    data[rhoIdx] = rho; data[vxIdx] = vx; data[vyIdx] = vy;

    let u2 = 1.5 * (vx * vx + vy * vy);
    let invTau = 1.0 / params.tau_0;
    
    let cu1 = 3.0 * vx; let cu2 = 3.0 * vy; let cu3 = 3.0 * -vx; let cu4 = 3.0 * -vy;
    let cu5 = 3.0 * (vx + vy); let cu6 = 3.0 * (-vx + vy); let cu7 = 3.0 * (-vx - vy); let cu8 = 3.0 * (vx - vy);

    data[get_f_idx(0u, writeParity, strideFace, params.fBase, i)] = pf[0] - invTau * (pf[0] - w_lbm[0] * rho * (1.0 - u2));
    data[get_f_idx(1u, writeParity, strideFace, params.fBase, i)] = pf[1] - invTau * (pf[1] - w_lbm[1] * rho * (1.0 + cu1 + 0.5 * cu1 * cu1 - u2));
    data[get_f_idx(2u, writeParity, strideFace, params.fBase, i)] = pf[2] - invTau * (pf[2] - w_lbm[2] * rho * (1.0 + cu2 + 0.5 * cu2 * cu2 - u2));
    data[get_f_idx(3u, writeParity, strideFace, params.fBase, i)] = pf[3] - invTau * (pf[3] - w_lbm[3] * rho * (1.0 + cu3 + 0.5 * cu3 * cu3 - u2));
    data[get_f_idx(4u, writeParity, strideFace, params.fBase, i)] = pf[4] - invTau * (pf[4] - w_lbm[4] * rho * (1.0 + cu4 + 0.5 * cu4 * cu4 - u2));
    data[get_f_idx(5u, writeParity, strideFace, params.fBase, i)] = pf[5] - invTau * (pf[5] - w_lbm[5] * rho * (1.0 + cu5 + 0.5 * cu5 * cu5 - u2));
    data[get_f_idx(6u, writeParity, strideFace, params.fBase, i)] = pf[6] - invTau * (pf[6] - w_lbm[6] * rho * (1.0 + cu6 + 0.5 * cu6 * cu6 - u2));
    data[get_f_idx(7u, writeParity, strideFace, params.fBase, i)] = pf[7] - invTau * (pf[7] - w_lbm[7] * rho * (1.0 + cu7 + 0.5 * cu7 * cu7 - u2));
    data[get_f_idx(8u, writeParity, strideFace, params.fBase, i)] = pf[8] - invTau * (pf[8] - w_lbm[8] * rho * (1.0 + cu8 + 0.5 * cu8 * cu8 - u2));

    let b = data[bioReadIdx];
    let lap = data[bioReadIdx - 1u] + data[bioReadIdx + 1u] + data[bioReadIdx - pNx] + data[bioReadIdx + pNx] - 4.0 * b;

    let ax = clamp(f32(px) - vx * 0.8, 0.0, f32(pNx - 2u));
    let ay = clamp(f32(py) - vy * 0.8, 0.0, f32(pNy - 2u));
    let ix = u32(floor(ax)); let iy = u32(floor(ay));
    let fx_b = ax - f32(ix); let fy_b = ay - f32(iy);

    let br = params.bioReadIdx;
    let v00 = data[br * strideFace + (iy * pNx + ix)]; 
    let v10 = data[br * strideFace + (iy * pNx + ix + 1u)];
    let v01 = data[br * strideFace + ((iy + 1u) * pNx + ix)]; 
    let v11 = data[br * strideFace + ((iy + 1u) * pNx + ix + 1u)];

    let adv = (1.0 - fy_b) * ((1.0 - fx_b) * v00 + fx_b * v10) + fy_b * ((1.0 - fx_b) * v01 + fx_b * v11);
    let rawBio = adv + params.bioDiffusion * lap + params.bioGrowth * b * (1.0 - b);
    data[bioWriteIdx] = clamp(max(rawBio, bioInjection), 0.0, 1.0);
}
