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
     * @description Obligatoire : Initialise l'état physique du moteur dans la VRAM/SAB avant la boucle.
     * @param faces Les N "Views" (FlatTensorView) pointant vers la VRAM continue.
     * @param nx Largeur
     * @param ny Hauteur
     * @param nz Profondeur
     * @param isWorker Si vrai, évite de surécrire la VRAM/SAB (déjà fait par le thread principal)
     */
    init(faces: FlatTensorView[], nx: number, ny: number, nz: number, isWorker?: boolean): void;

    /**
     * @description Exécute un pas de simulation mathématique à l'intérieur d'un TriadeCube.
     * @param faces Les N "Views" (FlatTensorView) pointant vers la VRAM continue.
     * @param nx Largeur
     * @param ny Hauteur
     * @param nz Profondeur
     * @param chunkX Coordonnée spatiale X
     * @param chunkY Coordonnée spatiale Y
     * @param chunkZ Coordonnée spatiale Z
     */
    compute(faces: FlatTensorView[], nx: number, ny: number, nz: number, chunkX?: number, chunkY?: number, chunkZ?: number): void | Promise<void>;

    /** 
     * Identifiant explicite du moteur algorithmique.
     */
    get name(): string;

    /**
     * @description Retourne le nombre de faces (buffers Float32Array) requis par ce moteur.
     */
    getRequiredFaces(): number;

    /**
     * @description Retourne la liste des indices de faces (0 à N) qui doivent être 
     * synchronisées aux frontières des chunks (Boundary Exchange) après chaque compute.
     */
    getSyncFaces?(): number[];

    /**
     * @description Retourne un dictionnaire de configuration encapsulant l'état sérialisable du moteur pour les Web Workers.
     */
    getConfig?(): Record<string, any>;

    /**
     * @description Permet au contrôleur (Grid) de propager une configuration partielle ou complète au moteur.
     */
    applyConfig?(config: any): void;

    /**
     * @description Permet au contrôleur (Grid) de propager la configuration des conditions aux limites à l'Engine.
     */
    setBoundaryConfig?(config: any): void;
    /**
     * @description Provides the equilibrium distribution for a given density and velocity.
     * Essential for initiating physical disturbances (splashes) in LBM engines.
     */
    getEquilibrium?(rho: number, ux: number, uy: number): Float32Array;

    /**
     * Code source WGSL (Compute Shader) optionnel pour l'exécution sur GPU.
     */
    readonly wgslSource?: string;

    /**
     * Initialisation spécifique au GPU (Optionnel).
     * @param device Le GPUDevice actif.
     * @param readBuffer Buffer initial pour la lecture.
     * @param writeBuffer Buffer initial pour l'écriture.
     * @param stride Le décalage en octets (byte stride) entre le début de chaque face.
     * @param nx Largeur
     * @param ny Hauteur
     * @param nz Profondeur
     */
    initGPU?(device: GPUDevice, readBuffer: GPUBuffer, writeBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void;

    /**
     * Dispatch GPU (Optionnel).
     * @param device Le GPUDevice actif.
     * @param commandEncoder L'encodeur de commandes pour créer des passes (ComputePass).
     * @param nx Largeur
     * @param ny Hauteur
     * @param nz Profondeur
     * @param readBuffer Buffer source (Frame N)
     * @param writeBuffer Buffer destination (Frame N+1)
     */
    computeGPU?(device: GPUDevice, commandEncoder: GPUCommandEncoder, nx: number, ny: number, nz: number, readBuffer: GPUBuffer, writeBuffer: GPUBuffer): void;
    /**
     * @description Identifie les capacités et caractéristiques du moteur pour le rendu automatique.
     */
    getTags?(): string[];

    /**
     * @description Retourne le schéma sémantique des données contenues dans les faces.
     */
    getSchema?(): EngineSchema;

    /**
     * @description Retourne le profil visuel par défaut pour l'affichage des résultats.
     */
    getVisualProfile?(): VisualProfile;
}

/**
 * Définit le rôle sémantique de chaque face de données.
 */
export interface EngineSchema {
    faces: Array<{
        index: number;
        label: string;
        isSynchronized?: boolean;
        isReadOnly?: boolean;
    }>;
}

/**
 * Définit comment le moteur doit être rendu visuellement.
 */
export interface VisualProfile {
    /** ID d'un style prédéfini dans le VisualRegistry (facultatif) */
    styleId?: string;
    layers?: Array<{
        faceIndex?: number; // Index physique
        faceLabel?: string; // Libellé sémantique (préféré)
        role: 'primary' | 'secondary' | 'obstacle' | 'vorticity' | 'heat' | 'binary';
        colormap?: 'grayscale' | 'heatmap' | 'ocean' | 'arctic' | 'vorticity';
        range?: [number, number];
        alpha?: number;
    }>;
    defaultMode?: 'topdown' | 'isometric' | 'isovolume' | '2.5d';
}




































