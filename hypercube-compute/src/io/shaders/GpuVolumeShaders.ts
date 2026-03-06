export const GPU_VOLUME_SHADERS = `
struct Uniforms {
    nx: u32, ny: u32, nz: u32, strideFace: u32,
    faceIdx: u32, obsIdx: u32, minVal: f32, maxVal: f32,
    scale: f32, opacity: f32, time: f32, colormapIdx: u32,
    mode: u32, canvasWidth: f32, canvasHeight: f32, cols: u32,
    rows: u32, chunkX: u32, chunkY: u32, vorticityIdx: u32
};

@group(0) @binding(0) var<storage, read> cube: array<f32>;
@group(0) @binding(1) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>
};

@vertex
fn vs_main(
    @builtin(vertex_index) vIdx: u32,
    @builtin(instance_index) iIdx: u32
) -> VertexOutput {
    let vNX = u.nx - 2u;
    let vNY = u.ny - 2u;
    
    // Instance ID -> local X, Y (skip ghost cells)
    let lx = (iIdx % vNX) + 1u;
    let ly = (iIdx / vNX) + 1u;
    let idx = ly * u.nx + lx;

    let val = cube[u.faceIdx * u.strideFace + idx];
    var isObs = false;
    if (u.obsIdx < 100u) {
        isObs = cube[u.obsIdx * u.strideFace + idx] > 0.5;
    }

    // --- COLORMAP LOGIC ---
    var color = vec4<f32>(0.2, 0.4, 0.8, 1.0); // Default Blue
    
    if (u.colormapIdx == 3u) { // ARCTIC (Case 01)
        // Background: Light Blue
        color = vec4<f32>(0.68, 0.78, 0.94, 1.0); 
        
        // Smoke (Navy)
        let s = clamp((val - u.minVal) / (u.maxVal - u.minVal + 0.001), 0.0, 1.0);
        let tS = pow(s, 0.6);
        let navy = vec4<f32>(0.04, 0.10, 0.35, 1.0);
        color = mix(color, navy, tS);
        
        // Vorticity (Red Highlights)
        if (u.vorticityIdx < 100u) {
            let vData = cube[u.vorticityIdx * u.strideFace + idx];
            let vRes = min(1.0, abs(vData) * 60.0); // Boosted from 30.0 to 60.0
            if (vRes > 0.1) {
                let tR = clamp((vRes - 0.1) * 2.0, 0.0, 1.0);
                color = mix(color, vec4<f32>(1.0, 0.0, 0.0, 1.0), tR);
            }
        }
    } else if (u.colormapIdx == 2u) { // HEATMAP
        let v = clamp((val - u.minVal) / (u.maxVal - u.minVal + 0.001), 0.0, 1.0);
        color = vec4<f32>(v, v * 0.5, 0.1, 1.0);
    } else if (u.colormapIdx == 1u) { // OCEAN (Case 02)
        // Ocean surface contrast (focus on variation around 1.0)
        let intensity = (val - 1.0) * 8.0; 
        let r = clamp(0.1 + intensity, 0.0, 1.0);
        let g = clamp(0.4 + intensity, 0.0, 1.0);
        let b = clamp(0.8 + intensity, 0.0, 1.0);
        color = vec4<f32>(r, g, b, 1.0);
    }

    if (isObs) {
        color = vec4<f32>(0.2, 0.2, 0.2, 1.0);
    }

    // Skip transparent quads (Only if colormap doesn't have a solid background)
    // Arctic (3u) has a solid light-blue background, so we don't skip it.
    if (val < 0.001 && !isObs && u.colormapIdx != 3u) {
        return VertexOutput(vec4<f32>(0.0, 0.0, 2.0, 1.0), vec4<f32>(0.0));
    }

    // --- PROJECTION LOGIC ---
    let totalVX = f32(vNX * u.cols);
    let totalVY = f32(vNY * u.rows);
    let worldX = f32(u.chunkX * vNX + (lx - 1u));
    let worldY = f32(u.chunkY * vNY + (ly - 1u));

    var out: VertexOutput;
    
    if (u.mode == 0u) { // TOPDOWN 2D (Case 01)
        let quadX = f32(vIdx % 2u);
        let quadY = f32(vIdx / 2u);
        let screenX = ((worldX + quadX) / totalVX) * 2.0 - 1.0;
        let screenY = ((worldY + quadY) / totalVY) * 2.0 - 1.0;
        out.position = vec4<f32>(screenX, -screenY, 0.1, 1.0);
    } else { // ISOMETRIC 3D (Case 02)
        let ISO_X = 0.866;
        let ISO_Y = 0.5;
        let scale = u.scale;
        
        let midW = totalVX * 0.5;
        let midH = totalVY * 0.5;
        let relX = worldX - midW;
        let relY = worldY - midH;
        
        let baseX = (relX - relY) * ISO_X * scale;
        let baseY = (relX + relY) * ISO_Y * scale;

        let h = select(val * scale * 25.0, scale * 10.0, isObs);
        let offX = f32(vIdx % 2u) * scale * 2.0;
        let offY = f32(vIdx / 2u) * h;
        
        // WebGPU Y-axis points DOWN, so we negate the final Y screen coordinate
        let outX = (baseX + offX) / (u.canvasWidth * 0.5);
        let outY = (baseY - offY) / (u.canvasHeight * 0.5);
        out.position = vec4<f32>(outX, -outY, 0.1, 1.0);
    }

    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.color.a < 0.01) { discard; }
    return in.color;
}
`;
