/**
 * V8 Standard Uniform Layout
 * This structure is mirrored exactly in WGSL.
 */

export interface V8Uniforms {
    nx: number;
    ny: number;
    nz: number;
    stride: number;
    parity: number;
    time: number;
    offX: number;
    offY: number;

    // Boundary Roles: 0=None, 1=Inlet, 2=Outlet, 3=Wall
    lRole: number; rRole: number; tRole: number; bRole: number;
    lVal: number; rVal: number; tVal: number; bVal: number;

    // Reserved for engine specific params
    params: Float32Array; // Usually 16 floats
}

/**
 * V8 Core Offsets (Source of Truth)
 */
export const V8_METADATA_OFFSET = 0; // nx...bVal
export const V8_PARAMS_OFFSET = 16;  // User parameters start here
export const V8_PARAM_STRIDE = 4;    // 16-byte alignment (vec4)

export const V8_UNIFORM_WGSL = `
struct V8Uniforms {
    nx: f32,
    ny: f32,
    nz: f32,
    stride: f32,
    parity: f32,
    time: f32,
    offX: f32,
    offY: f32,
    
    // Boundary Info
    lRole: f32, rRole: f32, tRole: f32, bRole: f32,
    lVal: f32,  rVal: f32,  tVal: f32,  bVal: f32,

    params: array<vec4<f32>, 16>,
};

@group(0) @binding(2) var<uniform> u: V8Uniforms;
`;
