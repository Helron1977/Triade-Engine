/**
 * Context provided to a colormap function for advanced mapping (SDF, Criteria, etc.)
 */
export interface ColormapContext {
    minV: number;
    maxV: number;
    invRange: number;
    chunkFaces: Float32Array[];
    srcIdx: number;
    worldX: number;
    worldY: number;
    vorticityFace?: number;
    auxiliaryFaces?: (number | undefined)[];
    options: any; // RenderOptions
}

/**
 * Type for a colormap function. Returns a 32-bit ABGR color.
 */
export type ColormapFn = (val: number, ctx: ColormapContext) => number;

/**
 * Registry for colormap algorithms.
 * @pattern Registry
 */
export class ColormapRegistry {
    private static registries: Map<string, ColormapFn> = new Map();

    static register(name: string, fn: ColormapFn): void {
        this.registries.set(name, fn);
    }

    static get(name: string): ColormapFn {
        return this.registries.get(name) || this.defaultGrayscale;
    }

    private static defaultGrayscale: ColormapFn = (val, ctx) => {
        const gray = Math.floor(Math.max(0, Math.min(1.0, (val - ctx.minV) * ctx.invRange)) * 255);
        return (255 << 24) | (gray << 16) | (gray << 8) | gray;
    };
}

// --- Initial Registration of built-in colormaps ---

ColormapRegistry.register('grayscale', ColormapRegistry.get('default'));

ColormapRegistry.register('arctic', (val, ctx) => {
    const s = Math.max(0, Math.min(1.0, (val - ctx.minV) * ctx.invRange));
    let r = 180, g = 220, b = 255, a = 255;
    const tS = s * (2.0 - s);
    r = r * (1 - tS) + 15 * tS;
    g = g * (1 - tS) + 30 * tS;
    b = b * (1 - tS) + 80 * tS;

    // Optional Vorticity check (uses resolved indices from context)
    let vFace = ctx.vorticityFace;
    if (vFace === undefined && ctx.auxiliaryFaces && ctx.auxiliaryFaces.length > 0) {
        vFace = ctx.auxiliaryFaces[0];
    }
    
    if (vFace !== undefined) {
        const vortData = ctx.chunkFaces[vFace];
        if (vortData) {
            const vMag = Math.min(1.0, Math.abs(vortData[ctx.srcIdx]) * 120.0);
            if (vMag > 0.05) {
                const tC = Math.min(1.0, (vMag - 0.05) * 1.5);
                r = r * (1 - tC) + 255 * tC;
                g = g * (1 - tC);
                b = b * (1 - tC);
            }
        }
    }
    return (a << 24) | (Math.floor(b) << 16) | (Math.floor(g) << 8) | Math.floor(r);
});

ColormapRegistry.register('heatmap', (val, ctx) => {
    let r = 0, g = 0, b = 0, a = 255;
    const s = Math.max(0, Math.min(1.0, (val - ctx.minV) * ctx.invRange));
    if (s < 0.5) {
        r = Math.floor(s * 2.0 * 255);
        g = 0;
        b = Math.floor(s * 0.2 * 255);
    } else {
        r = 255;
        g = Math.floor((s - 0.5) * 2.0 * 255);
        b = Math.floor(s * 0.2 * 255);
    }
    return (a << 24) | (b << 16) | (g << 8) | r;
});

ColormapRegistry.register('magma', (val, ctx) => {
    const s = Math.max(0, Math.min(1.0, (val - ctx.minV) * ctx.invRange));
    // Simplified Magma-like gradient (Dark Purple -> Pink -> Orange -> Yellow)
    let r, g, b;
    if (s < 0.25) {
        const t = s / 0.25;
        r = 5 + 40 * t; g = 5 + 10 * t; b = 25 + 60 * t;
    } else if (s < 0.5) {
        const t = (s - 0.25) / 0.25;
        r = 45 + 140 * t; g = 15 + 30 * t; b = 85 + 40 * t;
    } else if (s < 0.75) {
        const t = (s - 0.5) / 0.25;
        r = 185 + 70 * t; g = 45 + 100 * t; b = 125 - 60 * t;
    } else {
        const t = (s - 0.75) / 0.25;
        r = 255; g = 145 + 110 * t; b = 65 + 120 * t;
    }
    return (255 << 24) | (Math.floor(b) << 16) | (Math.floor(g) << 8) | Math.floor(r);
});

ColormapRegistry.register('inferno', (val, ctx) => {
    const s = Math.max(0, Math.min(1.0, (val - ctx.minV) * ctx.invRange));
    // Simplified Inferno (Black -> Red -> Orange -> Yellow)
    let r, g, b;
    if (s < 0.33) {
        const t = s / 0.33;
        r = 255 * t; g = 0; b = 0;
    } else if (s < 0.66) {
        const t = (s - 0.33) / 0.33;
        r = 255; g = 255 * t * 0.6; b = 0;
    } else {
        const t = (s - 0.66) / 0.34;
        r = 255; g = 153 + 102 * t; b = 255 * t;
    }
    return (255 << 24) | (Math.floor(b) << 16) | (Math.floor(g) << 8) | Math.floor(r);
});

ColormapRegistry.register('viridis', (val, ctx) => {
    const s = Math.max(0, Math.min(1.0, (val - ctx.minV) * ctx.invRange));
    // Simplified Viridis (Purple -> Blue -> Green -> Yellow)
    let r, g, b;
    if (s < 0.33) {
        const t = s / 0.33;
        r = 68 - 30 * t; g = 1 - 1 * t; b = 84 + 100 * t;
    } else if (s < 0.66) {
        const t = (s - 0.33) / 0.33;
        r = 38 + 50 * t; g = 0 + 150 * t; b = 184 - 50 * t;
    } else {
        const t = (s - 0.66) / 0.34;
        r = 88 + 167 * t; g = 150 + 105 * t; b = 134 - 134 * t;
    }
    return (255 << 24) | (Math.floor(b) << 16) | (Math.floor(g) << 8) | Math.floor(r);
});

ColormapRegistry.register('heatmap-criteria', (val, ctx) => {
    let r = 180, g = 220, b = 255, a = 255;
    const criteria = ctx.options.criteria || [];
    let sumW = 0;
    for (let i = 0; i < criteria.length; i++) sumW += criteria[i].weight;

    if (sumW > 0) {
        let score = 0;
        const faces = ctx.chunkFaces;
        // Resolve physical slots for criteria faces (this is tricky, might need pre-resolved faces in context)
        // For now, let's assume ctx.options.criteriaFaces is provided by the adapter
        const criteriaFaces = (ctx.options as any).resolvedCriteriaFaces || [];

        for (let i = 0; i < criteria.length; i++) {
            const weight = criteria[i].weight;
            if (weight === 0) continue;
            const hThresh = criteria[i].distanceThreshold || 0.05;
            const hRaw = Math.max(0, Math.min(1.0, (criteriaFaces[i][ctx.srcIdx] - ctx.minV) * ctx.invRange));
            const sLoc = (hRaw >= hThresh) ? 1.0 : (hRaw / hThresh);
            score += (weight / sumW) * sLoc;
        }
        
        const steps = 6;
        const quantizedS = Math.floor(score * steps) / steps;
        if (quantizedS < 0.1) { r = 15; g = 23; b = 42; }
        else if (quantizedS < 0.3) { r = 14; g = 110; b = 180; }
        else if (quantizedS < 0.5) { r = 6; g = 182; b = 212; }
        else if (quantizedS < 0.7) { r = 234; g = 179; b = 8; }
        else if (quantizedS < 0.9) { r = 132; g = 204; b = 22; }
        else { r = 34; g = 197; b = 94; }
    }
    return (a << 24) | (b << 16) | (g << 8) | r;
});

ColormapRegistry.register('spatial-decision', (val, ctx) => {
    let r = 0, g = 0, b = 0, a = 0;
    const criteriaSDF = ctx.options.criteriaSDF || [];
    const criteriaSDFFaces = (ctx.options as any).resolvedCriteriaSDFFaces || [];
    let sumW = 0;
    for (let i = 0; i < criteriaSDF.length; i++) sumW += criteriaSDF[i].weight;

    if (sumW > 0) {
        let score = 0;
        for (let i = 0; i < criteriaSDF.length; i++) {
            const weight = criteriaSDF[i].weight;
            if (weight === 0) continue;
            const seedX = criteriaSDFFaces[i].x[ctx.srcIdx];
            const seedY = criteriaSDFFaces[i].y[ctx.srcIdx];
            if (seedX < -9000 || seedY < -9000) continue;
            const dx = ctx.worldX - seedX;
            const dy = ctx.worldY - seedY;
            const distMeters = Math.sqrt(dx * dx + dy * dy) * 2.0;
            const distThresh = criteriaSDF[i].distanceThreshold;
            let sLoc = 0;
            if (distMeters <= distThresh) sLoc = Math.pow(1.0 - (distMeters / distThresh), 0.5);
            score += (weight / sumW) * sLoc;
        }
        
        const steps = 6;
        const qS = Math.floor(score * steps) / steps;
        if (qS <= 0.05) { r = 0; g = 0; b = 0; a = 0; }
        else if (qS < 0.2) { r = 14; g = 110; b = 180; a = 150; }
        else if (qS < 0.4) { r = 6; g = 182; b = 212; a = 180; }
        else if (qS < 0.6) { r = 234; g = 179; b = 8; a = 200; }
        else if (qS < 0.8) { r = 132; g = 204; b = 22; a = 220; }
        else { r = 34; g = 197; b = 94; a = 255; }
    }
    return (a << 24) | (b << 16) | (g << 8) | r;
});
