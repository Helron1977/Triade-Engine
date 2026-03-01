/**
 * Type alias to explicitly signal to AI that this array is a Flat Tensor (1D array simulating 2D/3D).
 * @AI_RULE DO NOT iterate using functional array methods (map, forEach, reduce). DO NOT use `push` or `splice`.
 */
export type FlatTensorView = Float32Array;

/**
 * Interface fondamentale (Le Contract) pour tout algorithme physique dans Hypercube Engine.
 * Implémentée par les moteurs LBM, GameOfLife, SAT.
 */
export interface IHypercubeEngine {
    /**
     * @description Exécute un pas de simulation mathématique à l'intérieur d'un TriadeCube.
     * @param faces Les 6 (ou N) "Views" (FlatTensorView) pointant vers la VRAM continue.
     * @param mapSize La dimension N (N x N) de ce chunk spatial.
     * 
     * @AI_RULE CRITICAL_PERFORMANCE: This method runs inside a hot loop (like 60 FPS).
     * @AI_RULE ZERO_ALLOCATION: DO NOT instantiate any object (`new Object()`, `[]`, `{}`) inside this scope to avoid Garbage Collection.
     * @AI_RULE MEMORY_ACCESS: Use 1D sequential iterations `for (let i = 0; i < length; i++)`. Read/Write values via `faces[index][i]`. Map 2D coords using `index = y * mapSize + x`.
     * @AI_RULE ERROR_HANDLING: If generating errors, be highly verbose for self-correction. Example: `throw new Error("[Hypercube] Out of bounds: index ${i} exceeds size ${length}")`
     */
    compute(faces: FlatTensorView[], mapSize: number, chunkX?: number, chunkY?: number): void | Promise<void>;

    /** 
     * Identifiant explicite du moteur algorithmique.
     * @AI_RULE This MUST match the semantic context of the Engine (e.g., 'Navier-Stokes-LBM-D2Q9', 'Game-Of-Life').
     */
    get name(): string;

    /**
     * @description Retourne le nombre de faces (buffers Float32Array) requis par ce moteur.
     * Par défaut, Hypercube utilise souvent 6 faces (Skybox style), mais le LBM en demande 22+.
     */
    getRequiredFaces(): number;

    /**
     * Code source WGSL (Compute Shader) optionnel pour l'exécution sur GPU.
     * Si fourni, HypercubeGrid l'utilisera en mode WebGPU (Accélération Matérielle).
     * @AI_RULE Le shader doit utiliser des Storage Buffers pour lire/écrire les faces.
     */
    readonly wgslSource?: string;

    /**
     * Initialisation spécifique au GPU (Optionnel).
     * Appelé par le HypercubeGrid lors du mode 'webgpu' pour que le moteur prépare ses Pipelines et BindGroups.
     * @param device Le GPUDevice actif.
     * @param cubeBuffer Le buffer unique contenant TOUTES les faces du cube (contingu).
     * @param stride Le décalage en octets (byte stride) entre le début de chaque face.
     * @param mapSize La dimension N du cube.
     */
    initGPU?(device: GPUDevice, cubeBuffer: GPUBuffer, stride: number, mapSize: number): void;

    /**
     * Dispatch GPU (Optionnel).
     * Effectue l'encodage des commandes (CommandEncoder) pour le Compute Shader.
     * @param device Le GPUDevice actif.
     * @param commandEncoder L'encodeur de commandes pour créer des passes (ComputePass).
     * @param mapSize La dimension N du cube.
     */
    computeGPU?(device: GPUDevice, commandEncoder: GPUCommandEncoder, mapSize: number): void;
}




































