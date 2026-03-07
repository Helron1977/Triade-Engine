/**
 * V8 Engine Manifest - The "Ideal World" Contract
 */

export type FaceType = 'scalar' | 'vector' | 'mask';
export type FaceDataType = 'float32' | 'uint32' | 'int32'; // Prise en charge des simulations discrètes

/**
 * Types de conditions aux limites (V8 Ideal) :
 * - joint : Connection automatique entre chunks voisins.
 * - wall : Mur physique (Bounce / No-slip).
 * - inlet : Injection (Dirichlet / Force).
 * - outlet : Sortie (Neumann / Absorb).
 * - symmetry : Miroir (Slip).
 */
export type BoundaryRole = 'joint' | 'wall' | 'inlet' | 'outlet' | 'symmetry';

export interface BoundaryProperty {
    role: BoundaryRole;
    factor?: number;     // Taux d'absorption ou de réflexion (0.0 à 1.0)
    value?: number | number[]; // Valeur fixe ou vecteur (ex: pression, vitesse)
}

export interface BoundaryManifest {
    top?: BoundaryProperty;
    bottom?: BoundaryProperty;
    left?: BoundaryProperty;
    right?: BoundaryProperty;
    all?: BoundaryProperty; // Shortcut
}

export interface FaceRequirement {
    name: string;
    type: FaceType;
    dataType?: FaceDataType; // 'float32' par défaut
    isReadOnly?: boolean;
    isSynchronized?: boolean;
    isOptional?: boolean;
    defaultValue?: number;
}

/**
 * Paramètres Sémantiques :
 * Permet d'utiliser des noms humains (ex: 'viscosity') plutôt que des constantes mathématiques (ex: 'mu').
 */
export interface ParameterRequirement {
    name: string;
    label: string;      // Nom lisible pour l'UI
    description?: string;
    defaultValue: number;
    min?: number;
    max?: number;
}
// ... rest

export interface NumericalScheme {
    type: 'advection' | 'diffusion' | 'laplacian' | 'reaction';
    method: 'Upwind' | 'Semi-Lagrangian' | 'MacCormack' | 'Explict-Euler';
    source: string; // Semantic name of the input face
    field?: string;  // Semantic name of the velocity/vector field
    params?: Record<string, number>;
}

export interface EngineDescriptor {
    name: string;
    description?: string;

    // 1. Data Contract
    faces: FaceRequirement[];

    // 2. Control Contract (Parameters)
    parameters: ParameterRequirement[];

    // 3. Compute Contract
    rules: NumericalScheme[];

    // 3. Optional Custom WGSL/Logic
    customKernels?: {
        name: string;
        source: string;
    }[];

    // 4. Output & Projection Contract (Zero-Copy extraction)
    outputs?: {
        name: string;
        type: 'fusion' | 'slice' | 'probe' | 'downsample';
        sources: string[]; // Noms des faces sources
        expression?: string; // ex: "rho + speed"
        interval?: number; // Fréquence d'extraction (ms ou frames)
    }[];

    // 5. Default Visualization Profile
    visualProfile?: {
        primary: string;
        overlay?: string;
        colormap: string;
    };
}
