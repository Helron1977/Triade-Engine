@group(0) @binding(0) var<storage, read_write> uData : array<f32>;

struct Uniforms {
    nx : u32,
    ny : u32,
    nz : u32,
    rank : f32,
    reg : f32,
    idxModeA : u32,
    idxModeB : u32,
    idxModeC : u32,
    idxTarget : u32,
    idxRecon : u32,
    strideFace : u32,
    tick : u32,
};
@group(0) @binding(1) var<uniform> u : Uniforms;

@compute @workgroup_size(16, 1)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let nx = u.nx;
    let ny = u.ny;
    let nz = u.nz;
    let rank = u32(u.rank);
    let stride = u.strideFace;
    let lr = 0.01;
    let reg = u.reg;

    // 1. RECONSTRUCTION PASS (Always happens for visualization)
    // We reuse (id.x, id.y) threads to fill the recon buffer
    // Since we changed workgroup size to (16, 1), we need to adapt
    let i_recon = id.x;
    if (i_recon < nx) {
        for (var j_recon: u32 = 0u; j_recon < ny; j_recon = j_recon + 1u) {
            for (var k_recon: u32 = 0u; k_recon < nz; k_recon = k_recon + 1u) {
                let idx = i_recon + j_recon * nx + k_recon * nx * ny;
                var pred: f32 = 0.0;
                for (var r: u32 = 0u; r < rank; r = r + 1u) {
                    pred += uData[u.idxModeA * stride + i_recon * rank + r] * 
                            uData[u.idxModeB * stride + j_recon * rank + r] * 
                            uData[u.idxModeC * stride + k_recon * rank + r];
                }
                uData[u.idxRecon * stride + idx] = pred;
            }
        }
    }

    // 2. BATCH FACTOR UPDATE (Rotating per tick to avoid race conditions)
    // tick % 3: 0 -> Update A, 1 -> Update B, 2 -> Update C
    let mode = u.tick % 3u;

    if (mode == 0u) {
        // Update Mode A: parallel over i
        let i = id.x;
        if (i < nx) {
            for (var r: u32 = 0u; r < rank; r = r + 1u) {
                var grad: f32 = 0.0;
                var count: f32 = 0.0;
                for (var j: u32 = 0u; j < ny; j = j + 1u) {
                    for (var k: u32 = 0u; k < nz; k = k + 1u) {
                        let idx = i + j * nx + k * nx * ny;
                        let val = uData[u.idxTarget * stride + idx];
                        if (val > 0.0) {
                            var pred_local: f32 = 0.0;
                            for (var rr: u32 = 0u; rr < rank; rr = rr + 1u) {
                                pred_local += uData[u.idxModeA * stride + i * rank + rr] * 
                                              uData[u.idxModeB * stride + j * rank + rr] * 
                                              uData[u.idxModeC * stride + k * rank + rr];
                            }
                            grad += (val - pred_local) * uData[u.idxModeB * stride + j * rank + r] * uData[u.idxModeC * stride + k * rank + r];
                            count += 1.0;
                        }
                    }
                }
                if (count > 0.0) {
                    let idxA = u.idxModeA * stride + i * rank + r;
                    uData[idxA] += lr * (grad - reg * uData[idxA]);
                }
            }
        }
    } else if (mode == 1u) {
        // Update Mode B: parallel over j
        let j = id.x; // Use id.x as our parallel dimension
        if (j < ny) {
            for (var r: u32 = 0u; r < rank; r = r + 1u) {
                var grad: f32 = 0.0;
                var count: f32 = 0.0;
                for (var i: u32 = 0u; i < nx; i = i + 1u) {
                    for (var k: u32 = 0u; k < nz; k = k + 1u) {
                        let idx = i + j * nx + k * nx * ny;
                        let val = uData[u.idxTarget * stride + idx];
                        if (val > 0.0) {
                            var pred_local: f32 = 0.0;
                            for (var rr: u32 = 0u; rr < rank; rr = rr + 1u) {
                                pred_local += uData[u.idxModeA * stride + i * rank + rr] * 
                                              uData[u.idxModeB * stride + j * rank + rr] * 
                                              uData[u.idxModeC * stride + k * rank + rr];
                            }
                            grad += (val - pred_local) * uData[u.idxModeA * stride + i * rank + r] * uData[u.idxModeC * stride + k * rank + r];
                            count += 1.0;
                        }
                    }
                }
                if (count > 0.0) {
                    let idxB = u.idxModeB * stride + j * rank + r;
                    uData[idxB] += lr * (grad - reg * uData[idxB]);
                }
            }
        }
    } else {
        // Update Mode C: parallel over k
        let k = id.x;
        if (k < nz) {
            for (var r: u32 = 0u; r < rank; r = r + 1u) {
                var grad: f32 = 0.0;
                var count: f32 = 0.0;
                for (var i: u32 = 0u; i < nx; i = i + 1u) {
                    for (var j: u32 = 0u; j < ny; j = j + 1u) {
                        let idx = i + j * nx + k * nx * ny;
                        let val = uData[u.idxTarget * stride + idx];
                        if (val > 0.0) {
                            var pred_local: f32 = 0.0;
                            for (var rr: u32 = 0u; rr < rank; rr = rr + 1u) {
                                pred_local += uData[u.idxModeA * stride + i * rank + rr] * 
                                              uData[u.idxModeB * stride + j * rank + rr] * 
                                              uData[u.idxModeC * stride + k * rank + rr];
                            }
                            grad += (val - pred_local) * uData[u.idxModeA * stride + i * rank + r] * uData[u.idxModeB * stride + j * rank + r];
                            count += 1.0;
                        }
                    }
                }
                if (count > 0.0) {
                    let idxC = u.idxModeC * stride + k * rank + r;
                    uData[idxC] += lr * (grad - reg * uData[idxC]);
                }
            }
        }
    }
}
