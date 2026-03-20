@group(0) @binding(0) var<storage, read_write> uCells: array<f32>;

struct Params {
    nx: u32,
    ny: u32,
    chunksX: u32,
    chunksY: u32,
    _pad4: f32,
    _pad5: f32,
    _pad6: f32,
    currentTick: u32,
    chunkX: u32,
    chunkY: u32,
    strideFace: u32,
    numObjects: u32,
    _pad12: u32, _pad13: u32, _pad14: u32, _pad15: u32, _pad16: u32,
    _pad17: u32, _pad18: u32, _pad19: u32, _pad20: u32, _pad21: u32,
    _pad22: u32, _pad23: u32,
    leftRole: u32,
    rightRole: u32,
    topRole: u32,
    bottomRole: u32,
    frontRole: u32,
    backRole: u32,
    _pad30: u32, _pad31: u32,
    objects: array<GpuObject, 8> 
};

struct GpuObject {
    pos: vec2<f32>,
    dim: vec2<f32>,
    isObstacle: f32,
    _pad: f32,
    objType: u32,
    _pad2: f32
};

@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = params.nx;
    let ny = params.ny;
    let pNx = nx + 2u;
    
    let px = id.x + 1u;
    let py = id.y + 1u;
    
    if (id.x >= nx || id.y >= ny) { return; }

    let readOffset = params.currentTick % 2u;
    let writeOffset = (params.currentTick + 1u) % 2u;
    let stride = params.strideFace;
    
    var neighbors = 0u;
    for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
        for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
            if (dx == 0i && dy == 0i) { continue; }
            let ni = u32(i32(py) + dy) * pNx + u32(i32(px) + dx);
            if (uCells[readOffset * stride + ni] > 0.5) {
                neighbors = neighbors + 1u;
            }
        }
    }
    
    let i = py * pNx + px;
    let alive = uCells[readOffset * stride + i] > 0.5;
    
    var nextState = 0.0;
    if (alive) {
        if (neighbors == 2u || neighbors == 3u) {
            nextState = 1.0;
        }
    } else {
        if (neighbors == 3u) {
            nextState = 1.0;
        }
    }
    
    uCells[writeOffset * stride + i] = nextState;
}
