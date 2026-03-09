/**
 * Core type definitions for Hypercube Neo.
 * Focused on pure declarative abstraction with sufficient detail for complex engines.
 */

export interface Dimension3D {
    nx: number; // Must be multiple of 16
    ny: number; // Must be multiple of 16
    nz: number; // 1 for 2D, multiple of 16 for 3D
}

export type BoundaryRole = 'joint' | 'wall' | 'periodic' | 'absorbing' | 'clamped' | 'dirichlet' | 'neumann' | 'symmetry';

export interface BoundarySide {
    role: BoundaryRole;
    factor?: number; // e.g., bounce factor, absorption coefficient
    value?: number | number[];  // e.g., fixed temperature, pressure (scalar or vector)
}

export interface GridBoundaries {
    left?: BoundarySide;
    right?: BoundarySide;
    top?: BoundarySide;
    bottom?: BoundarySide;
    front?: BoundarySide;
    back?: BoundarySide;
    all?: BoundarySide; // Shortcut for all sides
}

export type FaceType = 'scalar' | 'vector' | 'population' | 'macro' | 'bio' | 'field' | 'mask';

export interface EngineFace {
    name: string;
    type: FaceType;
    unit?: string;
    isSynchronized: boolean; // True if ghost cells need to be synced
    isReadOnly?: boolean;    // True for static masks/obstacles
    isPersistent?: boolean;  // True if face data must be preserved between steps. False for temporary populations.
    defaultValue?: number;
}

export interface NumericalScheme {
    type: 'lbm-d2q9' | 'lbm-macro' | 'lbm-smoke' | 'lbm-aero-fidelity-v1' | 'advection' | 'diffusion' | 'reaction' | 'stencil' | 'custom' | 'force';
    method?: 'Upwind' | 'Semi-Lagrangian' | 'Explicit-Euler' | 'Custom' | 'Force' | 'Direct';
    source: string; // Face name
    destination?: string;
    field?: string;  // For advection (velocity face name)
    params?: Record<string, number | string>;
}

export interface EngineDescriptor {
    name: string;
    version: string;
    faces: EngineFace[];
    parameters: Record<string, {
        name: string; // Human readable label
        type: 'number' | 'boolean' | 'string';
        default: any;
        min?: number;
        max?: number;
    }>;
    rules: NumericalScheme[];

    /**
     * Scaling & Physical Context.
     * Essential for real-world units (meters, seconds) and viscosity conversion.
     */
    context?: {
        unitScale: number; // e.g., 1.0 = 1 meter
        timeStep: number; // e.g., 0.001s per frame
        defaultGravity?: { x: number; y: number; z?: number };
    };

    /**
     * Semantic Projections & Data Export.
     * Maps physical faces to renderable textures or data buffers.
     */
    outputs: {
        name: string;
        sources: string[];
        expression?: string;
        unit?: string;
        export?: {
            target: 'texture' | 'buffer' | 'csv' | 'stream';
            format?: 'float32' | 'uint8' | 'rgba';
            frequency: 'per-frame' | 'final';
        };
    }[];

    /**
     * Simulation Life Cycle & Event Triggers.
     */
    lifeCycle?: {
        maxFrames?: number;
        stopConditions?: {
            expression: string; // e.g., "max(temp) > 1000"
            severity: 'stop' | 'pause' | 'warn';
        }[];
        events?: {
            frame: number;
            action: 'parameter-update' | 'object-spawn' | 'snapshot';
            payload: any;
        }[];
    };

    interactions?: {
        type: 'click' | 'drag' | 'hover';
        action: 'splash' | 'draw' | 'erase' | 'force';
        face: string; // Target face
        radius: number;
        multiplier?: number;
    }[];

    requirements: {
        ghostCells: number; // Stencil radius (0, 1, or more)
        pingPong: boolean;
    };
    visualProfile?: {
        styleId?: string;
        layers: {
            faceName: string;
            role: 'primary' | 'secondary' | 'obstacle' | 'vector';
            colormap?: string;     // Named colormap (e.g., 'magma')
            color?: string;        // Fixed hex color (e.g., '#ff0000') for obstacles or masks
            palette?: string[];    // Custom gradient palette (e.g., ['#000', '#fff'])
            range?: [number, number];
            alpha?: number;
        }[];
        defaultMode: '2d' | '2.5d' | '3d' | 'topdown';
    };

    /**
     * Performance Instrumentation & HUD.
     */
    instrumentation?: {
        enabled: boolean;
        metrics: ('fps' | 'mlups' | 'latency' | 'memory')[];
        targetPerformance?: number; // e.g. 60 for 60fps
    };
}

export interface VirtualObject {
    id: string;
    type: 'circle' | 'rect' | 'stencil' | 'ellipse' | 'polygon';
    position: { x: number; y: number; z?: number };
    dimensions: { w: number; h: number; d?: number };
    points?: { x: number; y: number }[]; // For polygon type
    properties: Record<string, number>; // e.g., { density: 1.0, isObstacle: 1 }

    /**
     * Influence Field & Rasterization logic.
     * Defines how this object affects the compute grid.
     */
    influence?: {
        falloff: 'step' | 'linear' | 'gaussian' | 'inverse-square';
        radius: number; // Effective reach of the influence
        power?: number;  // Falloff steepness
    };

    /**
     * How the properties of this object are combined into the grid.
     * 'add' is essential for heatmaps/influence accumulation (urban trees, schools).
     */
    rasterMode?: 'replace' | 'add' | 'multiply' | 'max' | 'min';

    /**
     * Animation & Path logic.
     * Allows objects to move over time (t in seconds or frames).
     */
    animation?: {
        velocity?: { x: number; y: number; z?: number };
        angularVelocity?: number;
        pathExpression?: {
            x?: string; // e.g., "128 + sin(t) * 32"
            y?: string;
            z?: string;
        };
    };

    renderOnly?: boolean; // If true, don't incorporate into compute grid
}

export interface HypercubeConfig {
    dimensions: Dimension3D;
    chunks: {
        x: number;
        y: number;
        z?: number;
    };
    boundaries: GridBoundaries;
    engine: string;
    params: Record<string, any>;
    objects?: VirtualObject[]; // Spatially distributed objects
    mode: 'cpu' | 'gpu';
    executionMode?: 'mono' | 'parallel';
    workers?: number;
}


/**
 * Defines the semantic role of each data face.
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
 * Defines how the engine should be rendered visually.
 */
export interface VisualProfile {
    /** ID of a predefined style in the VisualRegistry (optional) */
    styleId?: string;
    layers?: Array<{
        faceIndex?: number; // Physical index
        faceLabel?: string; // Semantic label (preferred)
        role: 'primary' | 'secondary' | 'obstacle' | 'vorticity' | 'heat' | 'binary';
        colormap?: 'grayscale' | 'heatmap' | 'ocean' | 'arctic' | 'vorticity';
        range?: [number, number];
        alpha?: number;
    }>;
    defaultMode?: 'topdown' | 'isometric' | 'isovolume' | '2.5d';
}

/**
 * A Self-contained Manifest (V4) defining both the engine and the configuration.
 * This is the Single Source of Truth for a simulation "Case".
 */
export interface HypercubeManifest {
    $schema?: string;
    name: string;
    version: string;
    engine: EngineDescriptor;
    config: HypercubeConfig;
    visualProfile?: VisualProfile;
}
