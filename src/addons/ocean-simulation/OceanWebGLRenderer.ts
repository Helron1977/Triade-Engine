import { TriadeGrid } from '../../core/TriadeGrid';

export class OceanWebGLRenderer {
    private gl: WebGL2RenderingContext;
    private program: WebGLProgram;
    private indexCount: number = 0;

    private texRho: WebGLTexture;
    private texUx: WebGLTexture;
    private texUy: WebGLTexture;
    private texObst: WebGLTexture;

    private globalW: number;
    private globalH: number;
    private chunkSize: number;

    constructor(canvas: HTMLCanvasElement, cols: number, rows: number, chunkSize: number) {
        this.chunkSize = chunkSize;
        this.globalW = cols * chunkSize;
        this.globalH = rows * chunkSize;

        const gl = canvas.getContext('webgl2');
        if (!gl) throw new Error("WebGL2 non supporté.");
        this.gl = gl;

        // Extensions pour Float32 Textures
        this.gl.getExtension("EXT_color_buffer_float");
        this.gl.getExtension("OES_texture_float_linear");

        const vsSource = `#version 300 es
        in vec2 a_position;
        out vec2 v_uv;
        out vec3 v_worldPos;
        uniform sampler2D u_rho; // We sample density in Vertex Shader!
        uniform sampler2D u_obst; // Also sample obstacles in Vertex Shader!
        
        void main() {
            v_uv = a_position * 0.5 + 0.5;
            v_uv.y = 1.0 - v_uv.y;
            
            // VTF: Vertex Texture Fetch (Real 3D heights from LBM Density)
            float rho = texture(u_rho, v_uv).r;
            float obst = texture(u_obst, v_uv).r;
            float h = 0.0;
            if (obst > 0.5) {
                h = 5.0; // Island elevated plateau
            } else {
                h = (rho - 1.0) * 8.0; // Wave extrusion amplitude (softened)
            }
            
            vec3 p = vec3(a_position.x, a_position.y, h);
            v_worldPos = p;

            // 3D Camera / Perspective Math
            float angleX = -1.1; // Look down tilt
            float cosX = cos(angleX);
            float sinX = sin(angleX);
            
            // Apply Pitch rotation
            float yRot = p.y * cosX - p.z * sinX;
            float zRot = p.y * sinX + p.z * cosX;
            
            // Move camera back
            zRot -= 2.2; 
            
            // Perspective projection
            float fov = 1.7; 
            gl_Position = vec4(p.x * fov, (yRot - 0.2) * fov, zRot, -zRot);
        }`;

        const fsSource = `#version 300 es
        precision highp float;
        precision highp sampler2D;
        
        in vec2 v_uv;
        in vec3 v_worldPos;
        uniform sampler2D u_rho; // LBM Density -> Heights
        uniform sampler2D u_ux;
        uniform sampler2D u_uy;
        uniform sampler2D u_obst;

        out vec4 outColor;
        
        void main() {
            float obst = texture(u_obst, v_uv).r;
            if (obst > 0.5) {
                // Island material
                vec3 islandTop = vec3(0.5, 0.6, 0.3); // Grass / Forest
                vec3 islandSide = vec3(0.7, 0.6, 0.4); // Sand / Rock
                
                // Read local normals to detect cliffs
                float oL = textureOffset(u_obst, v_uv, ivec2(-1, 0)).r;
                float oR = textureOffset(u_obst, v_uv, ivec2(1, 0)).r;
                float oB = textureOffset(u_obst, v_uv, ivec2(0, -1)).r;
                float oT = textureOffset(u_obst, v_uv, ivec2(0, 1)).r;
                float sumO = oL + oR + oB + oT;
                
                if (sumO < 3.5) {
                    outColor = vec4(islandSide, 1.0); // Cliff / Beach
                } else {
                    outColor = vec4(islandTop, 1.0); // Plateau core
                }
                return;
            }

            // Sample density around the pixel to compute normal vector
            float rL = textureOffset(u_rho, v_uv, ivec2(-1, 0)).r;
            float rR = textureOffset(u_rho, v_uv, ivec2(1, 0)).r;
            float rB = textureOffset(u_rho, v_uv, ivec2(0, -1)).r; // Inverted Y due to texture flip
            float rT = textureOffset(u_rho, v_uv, ivec2(0, 1)).r;

            // Generate Normal from heightmap (Density is height proxy)
            float strength = 25.0; // Softened normals
            vec3 normal = normalize(vec3((rL - rR) * strength, (rB - rT) * strength, 1.0));

            // Lighting setup
            vec3 lightDir = normalize(vec3(0.5, 0.8, 0.5)); // Sun angle
            vec3 viewDir  = normalize(vec3(0.0, -0.6, 1.0)); // Looking at the water
            
            float ux = texture(u_ux, v_uv).r;
            float uy = texture(u_uy, v_uv).r;
            float speed = sqrt(ux*ux + uy*uy);

            // Water Base Color based on speed (currents)
            vec3 waterDeep = vec3(0.02, 0.2, 0.5);
            vec3 waterShallow = vec3(0.0, 0.6, 0.9);
            vec3 waterColor = mix(waterDeep, waterShallow, clamp(speed * 80.0, 0.0, 1.0));

            // Diffuse Lighting (Lambert)
            float diff = max(dot(normal, lightDir), 0.0);
            vec3 diffuse = waterColor * (diff * 0.7 + 0.3); // + ambient

            // Specular Lighting (Phong)
            vec3 reflectDir = reflect(-lightDir, normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 48.0);
            vec3 specular = vec3(1.0, 1.0, 1.0) * spec * 0.8;

            // Foam based on density peaks (breaking waves)
            float rho_center = texture(u_rho, v_uv).r;
            float foam = (rho_center > 1.015) ? clamp((rho_center - 1.015) * 15.0, 0.0, 1.0) : 0.0;
            
            outColor = vec4(diffuse + specular + foam, 1.0);
        }`;

        this.program = this.createProgram(vsSource, fsSource);

        // Generate 256x256 High-Res Grid Mesh for 3D displacement
        const segments = 256;
        const vertices = new Float32Array((segments + 1) * (segments + 1) * 2);
        let ptr = 0;
        for (let y = 0; y <= segments; y++) {
            for (let x = 0; x <= segments; x++) {
                vertices[ptr++] = (x / segments) * 2.0 - 1.0;
                vertices[ptr++] = 1.0 - (y / segments) * 2.0;
            }
        }

        const indices = new Uint32Array(segments * segments * 6);
        let iPtr = 0;
        for (let y = 0; y < segments; y++) {
            for (let x = 0; x < segments; x++) {
                const i = x + y * (segments + 1);
                indices[iPtr++] = i;
                indices[iPtr++] = i + 1;
                indices[iPtr++] = i + segments + 1;
                indices[iPtr++] = i + segments + 1;
                indices[iPtr++] = i + 1;
                indices[iPtr++] = i + segments + 2;
            }
        }
        this.indexCount = indices.length;

        const posBuf = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, posBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);

        const indexBuf = this.gl.createBuffer()!;
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, indexBuf);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, indices, this.gl.STATIC_DRAW);

        this.texRho = this.createTexture();
        this.texUx = this.createTexture();
        this.texUy = this.createTexture();
        this.texObst = this.createTexture();

        this.gl.useProgram(this.program);
        this.bindTextureUniform("u_rho", 0);
        this.bindTextureUniform("u_ux", 1);
        this.bindTextureUniform("u_uy", 2);
        this.bindTextureUniform("u_obst", 3);

        const posLoc = this.gl.getAttribLocation(this.program, "a_position");
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);
    }

    private createTexture() {
        const tex = this.gl.createTexture()!;
        this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR); // Linear for smoother visual
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);

        // Allocate full global Toric map texture
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.R32F, this.globalW, this.globalH, 0, this.gl.RED, this.gl.FLOAT, null);
        return tex;
    }

    private bindTextureUniform(name: string, unit: number) {
        const loc = this.gl.getUniformLocation(this.program, name);
        this.gl.uniform1i(loc, unit);
    }

    private updateSubTexture(tex: WebGLTexture, unit: number, xOff: number, yOff: number, data: Float32Array) {
        this.gl.activeTexture(this.gl.TEXTURE0 + unit);
        this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
        this.gl.texSubImage2D(this.gl.TEXTURE_2D, 0, xOff, yOff, this.chunkSize, this.chunkSize, this.gl.RED, this.gl.FLOAT, data);
    }

    public render(grid: TriadeGrid) {
        this.gl.useProgram(this.program);

        // Upload each chunk into the master textures using O(1) subset uploads
        for (let y = 0; y < grid.rows; y++) {
            for (let x = 0; x < grid.cols; x++) {
                const cube = grid.cubes[y][x];
                if (!cube) continue;

                const xOff = x * this.chunkSize;
                const yOff = y * this.chunkSize;

                // Face mapping for OceanEngine (Rho = Face 20)
                this.updateSubTexture(this.texRho, 0, xOff, yOff, cube.faces[20]);
                this.updateSubTexture(this.texUx, 1, xOff, yOff, cube.faces[18]);
                this.updateSubTexture(this.texUy, 2, xOff, yOff, cube.faces[19]);
                this.updateSubTexture(this.texObst, 3, xOff, yOff, cube.faces[22]);
            }
        }

        // Draw 3D Surface
        this.gl.drawElements(this.gl.TRIANGLES, this.indexCount, this.gl.UNSIGNED_INT, 0);
    }

    private createProgram(vsSource: string, fsSource: string) {
        const vs = this.gl.createShader(this.gl.VERTEX_SHADER)!;
        this.gl.shaderSource(vs, vsSource);
        this.gl.compileShader(vs);
        const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER)!;
        this.gl.shaderSource(fs, fsSource);
        this.gl.compileShader(fs);
        const prog = this.gl.createProgram()!;
        this.gl.attachShader(prog, vs);
        this.gl.attachShader(prog, fs);
        this.gl.linkProgram(prog);
        return prog;
    }
}
