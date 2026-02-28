export class WebGLAdapter {
    private gl: WebGL2RenderingContext;
    private texture: WebGLTexture;
    private program: WebGLProgram;
    public readonly mapSize: number;

    constructor(canvas: HTMLCanvasElement, mapSize: number) {
        this.mapSize = mapSize;
        const gl = canvas.getContext('webgl2');
        if (!gl) {
            throw new Error("WebGL2 n'est pas supporté par ce navigateur.");
        }
        this.gl = gl;

        // --- SHADERS ---
        // Vertex Shader : quad simple couvrant tout le canvas
        const vsSource = `#version 300 es
        in vec2 a_position;
        out vec2 v_uv;
        void main() {
            v_uv = a_position * 0.5 + 0.5; // conversion -1,1 vers 0,1
            v_uv.y = 1.0 - v_uv.y; // Flip Y
            gl_Position = vec4(a_position, 0.0, 1.0);
        }`;

        // Fragment Shader : Lit la FloatTexture et applique un gradient "Heatmap"
        const fsSource = `#version 300 es
        precision highp float;
        precision highp sampler2D;
        
        in vec2 v_uv;
        uniform sampler2D u_tensor;
        out vec4 outColor;
        
        // Palette simple: froid (bleu) -> chaud (rouge)
        vec3 heatMap(float t) {
            return clamp(vec3(1.5*t, 2.0*t - 0.5, 0.5 - t*1.5), 0.0, 1.0);
        }
        
        void main() {
            // Lecture du Float32Array exact depuis la VRAM Triade
            float val = texture(u_tensor, v_uv).r; 
            
            // Affichage avec la palette
            vec3 color = mix(vec3(0.05, 0.07, 0.1), vec3(0.0, 0.8, 1.0), clamp(val, 0.0, 1.0)); // Custom Cyan/Dark mapping
            if (val > 1.0) color = vec3(1.0, 1.0, 1.0); // Saturation
            
            outColor = vec4(color, 1.0);
        }`;

        this.program = this.createProgram(vsSource, fsSource);
        this.gl.useProgram(this.program);

        // --- QUAD GEOMETRY ---
        const positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array([
            -1, -1, 1, -1, -1, 1,
            -1, 1, 1, -1, 1, 1,
        ]);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        const posLoc = this.gl.getAttribLocation(this.program, "a_position");
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        // --- FLOAT TEXTURE O(1) BINDING ---
        // Extension obligatoire pour les textures en virgule flottante
        this.gl.getExtension("EXT_color_buffer_float");
        this.gl.getExtension("OES_texture_float_linear");

        this.texture = this.gl.createTexture()!;
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);

        // Configuration pour ne pas flouter ou filtrer (Rendu data brut)
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);

        // Alloue la texture Float32 en une seule passe sur la GPU
        this.gl.texImage2D(
            this.gl.TEXTURE_2D, 0, this.gl.R32F,
            mapSize, mapSize, 0,
            this.gl.RED, this.gl.FLOAT, null
        );
    }

    /**
     * Propulse la Face Mémoire Triade directement dans la VRAM GPU.
     * C'est l'opération la plus rapide sur navigateur (O(1) Data Upload).
     */
    public renderFaceToWebGL(faceData: Float32Array) {
        this.gl.useProgram(this.program);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.texture);

        // UPLOAD GPU DIRECT : Push du Float32Array vers le WebGL Shader
        this.gl.texSubImage2D(
            this.gl.TEXTURE_2D, 0, 0, 0,
            this.mapSize, this.mapSize,
            this.gl.RED, this.gl.FLOAT, faceData
        );

        // Dessine la Frame
        this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    }

    private createProgram(vsSource: string, fsSource: string): WebGLProgram {
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

        if (!this.gl.getProgramParameter(prog, this.gl.LINK_STATUS)) {
            console.error(this.gl.getProgramInfoLog(prog));
            throw new Error("Erreur de compilation WebGL");
        }

        return prog;
    }
}
