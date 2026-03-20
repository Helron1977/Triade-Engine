struct Params {
    nx: u32,
    ny: u32,
    chunksX: u32,
    chunksY: u32,
    omega: f32,
    inflowUx: f32,
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
    vortReadIdx: u32,
    smokeReadIdx: u32,
    vxWriteIdx: u32,
    vyWriteIdx: u32,
    vortWriteIdx: u32,
    smokeWriteIdx: u32,
    fBase: u32,
    // Slots 22+: Extensions (Total 10 slots before objects at 32)
    jfaStep: u32,       // Slot 22
    baseX: u32,         // Slot 23
    leftRole: u32,      // Slot 24
    rightRole: u32,     // 25
    topRole: u32,       // 26
    bottomRole: u32,    // 27
    frontRole: u32,     // 28
    backRole: u32,      // 29
    baseY: u32,         // Slot 30
    _pad31: u32,        // 31
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

fn get_idx(id: vec2<u32>) -> u32 {
    return id.y * params.nx + id.x;
}

fn get_phys_idx(base_idx: u32, parity: u32, i: u32) -> u32 {
    let strideFace = params.strideFace;
    return (base_idx + parity) * strideFace + i;
}

fn distSq(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    if (x2 < -9000.0 || y2 < -9000.0) { return 1.0e10; }
    let dx = x1 - x2;
    let dy = y1 - y2;
    return dx * dx + dy * dy;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = params.nx;
    let ny = params.ny;
    if (id.x >= nx || id.y >= ny) { return; }

    let readParity = params.currentTick % 2u;
    let writeParity = (params.currentTick + 1u) % 2u;
    
    let i = get_idx(id.xy);
    let gX = f32(params.chunkX * nx + id.x);
    let gY = f32(params.chunkY * ny + id.y);

    let baseX = params.baseX;
    let baseY = params.baseY;

    var bestX = data[get_phys_idx(baseX, readParity, i)];
    var bestY = data[get_phys_idx(baseY, readParity, i)];
    var bestDist = distSq(gX, gY, bestX, bestY);

    let step = i32(params.jfaStep);
    if (step > 0) {
        for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
            for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
                if (dx == 0i && dy == 0i) { continue; }
                let nix = i32(id.x) + dx * step;
                let niy = i32(id.y) + dy * step;
                if (nix < 0 || nix >= i32(nx) || niy < 0 || niy >= i32(ny)) { continue; }
                let ni = u32(niy * i32(params.nx) + nix);
                let seedX = data[get_phys_idx(baseX, readParity, ni)];
                let seedY = data[get_phys_idx(baseY, readParity, ni)];
                let d = distSq(gX, gY, seedX, seedY);
                if (d < bestDist) {
                    bestDist = d; bestX = seedX; bestY = seedY;
                }
            }
        }
    } else {
            for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
            for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
                if (dx == 0i && dy == 0i) { continue; }
                let nix = i32(id.x) + dx;
                let niy = i32(id.y) + dy;
                if (nix < 0 || nix >= i32(nx) || niy < 0 || niy >= i32(ny)) { continue; }
                let ni = u32(niy * i32(params.nx) + nix);
                let seedX = data[get_phys_idx(baseX, readParity, ni)];
                let seedY = data[get_phys_idx(baseY, readParity, ni)];
                let d = distSq(gX, gY, seedX, seedY);
                if (d < bestDist) {
                    bestDist = d; bestX = seedX; bestY = seedY;
                }
            }
        }
    }
    data[get_phys_idx(baseX, writeParity, i)] = bestX;
    data[get_phys_idx(baseY, writeParity, i)] = bestY;
}
