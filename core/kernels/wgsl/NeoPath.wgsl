@group(0) @binding(0) var<storage, read_write> uData: array<f32>;

struct GpuObject {
    pos: vec2<f32>,
    dim: vec2<f32>,
    isObstacle: f32,
    _pad: f32,
    objType: u32,
    _pad2: f32
};

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
    obsIdx: u32,       // Slot 12
    vxReadIdx: u32,    // 13 (Using VX Read for Distance Read)
    _pad14: u32,
    _pad15: u32,
    _pad16: u32,
    vxWriteIdx: u32,   // 17 (Using VX Write for Distance Write)
    _pad18: u32,
    _pad19: u32,
    _pad20: u32,
    _pad21: u32,
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

@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = params.nx;
    let ny = params.ny;
    let pNx = nx + 2u;
    
    let px = id.x + 1u;
    let py = id.y + 1u;
    
    if (id.x >= nx || id.y >= ny) { return; }
    
    let stride = params.strideFace;
    let distRead = params.vxReadIdx * stride;
    let distWrite = params.vxWriteIdx * stride;
    let obsOffset = params.obsIdx * stride;
    
    let i = py * pNx + px;
    
    if (uData[obsOffset + i] > 0.5) {
        uData[distWrite + i] = 1000000.0;
        return;
    }
    
    var minDist = uData[distRead + i];
    
    // Check neighbors
    let n1 = (py - 1u) * pNx + px;
    let n2 = (py + 1u) * pNx + px;
    let n3 = py * pNx + (px - 1u);
    let n4 = py * pNx + (px + 1u);
    
    minDist = min(minDist, uData[distRead + n1] + 1.0);
    minDist = min(minDist, uData[distRead + n2] + 1.0);
    minDist = min(minDist, uData[distRead + n3] + 1.0);
    minDist = min(minDist, uData[distRead + n4] + 1.0);
    
    uData[distWrite + i] = minDist;
}
