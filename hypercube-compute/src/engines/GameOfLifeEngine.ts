import type { IHypercubeEngine } from "./IHypercubeEngine";

export interface EcosystemConfig {
    deathProb?: number;
    growthProb?: number;
    eatThresholdBase?: number;     // e.g. 3.5
    plantEatThreshold?: number;    // e.g. 2.8
    herbiEatThreshold?: number;    // e.g. 3.8
    carniEatThreshold?: number;    // e.g. 3.2
    carniStarveThreshold?: number; // e.g. 3.5
}

export class GameOfLifeEngine implements IHypercubeEngine {
    private config: Required<EcosystemConfig>;

    constructor(config: EcosystemConfig = {}) {
        this.config = {
            deathProb: config.deathProb ?? 0.015,
            growthProb: config.growthProb ?? 0.03,
            eatThresholdBase: config.eatThresholdBase ?? 3.5,
            plantEatThreshold: config.plantEatThreshold ?? 2.8,
            herbiEatThreshold: config.herbiEatThreshold ?? 3.2,
            carniEatThreshold: config.carniEatThreshold ?? 3.2,
            carniStarveThreshold: config.carniStarveThreshold ?? 3.5
        };
    }

    public get name(): string {
        return "GameOfLifeEngine";
    }

    public getRequiredFaces(): number {
        return 6;
    }

    public getSyncFaces(): number[] {
        return [1, 3];
    }

    // Seuil et probas pour équilibrer
    private readonly survivalMin = 2; // Min voisins même état pour survivre
    private readonly survivalMax = 3; // Max pour éviter surpop
    private readonly birthThreshold = 3; // Prédateurs pour naissance

    public init(faces: Float32Array[], nx: number, ny: number, nz: number, isWorker: boolean = false): void {
        // L'écosystème est typiquement initialisé par l'extérieur via des graines
    }

    private pipelineGoL: GPUComputePipeline | null = null;
    private pipelineCopy: GPUComputePipeline | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private cubeBuffer: GPUBuffer | null = null;
    public gpuEnabled: boolean = false;
    private frameCounter: number = 0;

    public initGPU(device: GPUDevice, cubeBuffer: GPUBuffer, stride: number, nx: number, ny: number, nz: number): void {
        this.cubeBuffer = cubeBuffer;

        const wgSizeX = 16;
        const wgSizeY = 16;

        const shaderModule = device.createShaderModule({ code: this.getWgslSource() });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        this.pipelineGoL = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_gol' }
        });

        this.pipelineCopy = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'copy_faces' }
        });

        const strideFace = nx * ny * nz;
        const uniformData = new ArrayBuffer(64);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);

        u32[0] = nx;
        u32[1] = ny;
        u32[2] = nz;
        u32[3] = strideFace;
        f32[4] = this.config.eatThresholdBase;
        f32[5] = this.config.plantEatThreshold;
        f32[6] = this.config.herbiEatThreshold;
        f32[7] = this.config.carniEatThreshold;
        f32[8] = this.config.carniStarveThreshold;
        u32[9] = 0; // frameCounter

        // Utilisation via le GPUContext importer dynamiquement n'est pas possible directement ici si on l'a pas en dépendance,
        // mais le framework injecte le device directement. On va créer le buffer nativement:
        this.uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: cubeBuffer } },
                { binding: 1, resource: { buffer: this.uniformBuffer } }
            ]
        });

        this.gpuEnabled = true;
    }

    public computeGPU(device: GPUDevice, commandEncoder: GPUCommandEncoder, nx: number, ny: number, nz: number): void {
        if (!this.bindGroup || !this.pipelineGoL || !this.pipelineCopy || !this.uniformBuffer) return;

        this.frameCounter++;

        const uniformData = new ArrayBuffer(64);
        const u32 = new Uint32Array(uniformData);
        const f32 = new Float32Array(uniformData);
        const strideFace = nx * ny * nz;

        u32[0] = nx; u32[1] = ny; u32[2] = nz; u32[3] = strideFace;
        f32[4] = this.config.eatThresholdBase; f32[5] = this.config.plantEatThreshold;
        f32[6] = this.config.herbiEatThreshold; f32[7] = this.config.carniEatThreshold;
        f32[8] = this.config.carniStarveThreshold; u32[9] = this.frameCounter;

        device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setBindGroup(0, this.bindGroup);

        passEncoder.setPipeline(this.pipelineGoL);
        passEncoder.dispatchWorkgroups(Math.ceil(nx / 16), Math.ceil(ny / 16), nz || 1);

        passEncoder.setPipeline(this.pipelineCopy);
        passEncoder.dispatchWorkgroups(Math.ceil((nx * ny * nz) / 256));

        passEncoder.end();
    }

    public compute(faces: Float32Array[], nx: number, ny: number, nz: number): void {
        const current = faces[1]; // État actuel t (0-3)
        const next = faces[2];    // État futur t+1 (0-3)
        const density = faces[3]; // Densité/âge pour visuel soft (0.0-1.0)

        // Clear next
        next.fill(0);

        for (let lz = 0; lz < nz; lz++) {
            const zOff = lz * ny * nx;

            // Double boucle optimisée pour accès mémoires continus
            for (let y = 0; y < ny; y++) {

                const top = (y === 0) ? ny - 1 : y - 1;
                const bottom = (y === ny - 1) ? 0 : y + 1;

                const topRow = zOff + top * nx;
                const midRow = zOff + y * nx;
                const botRow = zOff + bottom * nx;

                for (let x = 0; x < nx; x++) {
                    const left = (x === 0) ? nx - 1 : x - 1;
                    const right = (x === nx - 1) ? 0 : x + 1;

                    const idx = midRow + x;
                    const state = Math.floor(current[idx]); // 0: Vide, 1: Plante, 2: Herbi, 3: Carni

                    // Le prédateur / successeur de l'état actuel
                    const targetState = (state + 1) % 4;

                    let sameState = 0;
                    let predators = 0;

                    // Von Neumann Neighborhood (Cardinaux, poids 1.5)
                    sameState += (current[topRow + x] === state ? 1.5 : 0) + (current[botRow + x] === state ? 1.5 : 0) +
                        (current[midRow + left] === state ? 1.5 : 0) + (current[midRow + right] === state ? 1.5 : 0);
                    predators += (current[topRow + x] === targetState ? 1.5 : 0) + (current[botRow + x] === targetState ? 1.5 : 0) +
                        (current[midRow + left] === targetState ? 1.5 : 0) + (current[midRow + right] === targetState ? 1.5 : 0);

                    // Moore Neighborhood (Diagonales, poids 1)
                    sameState += (current[topRow + left] === state ? 1 : 0) + (current[topRow + right] === state ? 1 : 0) +
                        (current[botRow + left] === state ? 1 : 0) + (current[botRow + right] === state ? 1 : 0);
                    predators += (current[topRow + left] === targetState ? 1 : 0) + (current[topRow + right] === targetState ? 1 : 0) +
                        (current[botRow + left] === targetState ? 1 : 0) + (current[botRow + right] === targetState ? 1 : 0);

                    // Règles organiques d'écosystème avec Densité Active
                    const densityFactor = density[idx]; // 0..1

                    let newState = state;
                    let newDensity = density[idx];

                    // 1. Rééquilibrage des seuils (Plus symétriques)
                    let eatThreshold = this.config.eatThresholdBase;
                    if (state === 0) eatThreshold = this.config.plantEatThreshold;       // (Vide -> Plante)
                    else if (state === 1) eatThreshold = this.config.herbiEatThreshold;  // (Plante -> Herbi)
                    else if (state === 2) eatThreshold = this.config.carniEatThreshold;  // (Herbi -> Carni)
                    else if (state === 3) eatThreshold = this.config.carniStarveThreshold;  // (Carni -> Vide)

                    // 2. Bonus de Survie Asymétrique via Densité
                    if (state === 1) eatThreshold += densityFactor * 1.5; // Plantes denses (forêts) très dures à manger
                    else if (state === 2) eatThreshold += densityFactor * 0.8; // Herbi moyens
                    else if (state === 3) eatThreshold += densityFactor * 0.4; // Carni fragiles même groupés

                    if (predators >= eatThreshold) {
                        newState = targetState; // On se fait dévorer / remplacer
                        newDensity = 0.2 + Math.random() * 0.2; // La nouvelle espèce démarre émergente
                    } else {
                        // Survie et évolution lente
                        if (state === 0) {
                            // Éclosion miraculeuse (très très rare)
                            if (Math.random() < 0.0005) {
                                newState = 1;
                                newDensity = 0.1;
                            } else {
                                newDensity = 0.0;
                            }
                        } else {
                            // 3. Faim exacerbée des herbivores (Mort si peu de plantes autour)
                            let plantNeighbors = 0;
                            if (current[topRow + x] === 1) plantNeighbors++;
                            if (current[midRow + left] === 1) plantNeighbors++;
                            if (current[midRow + right] === 1) plantNeighbors++;
                            if (current[botRow + x] === 1) plantNeighbors++;

                            if (state === 2 && plantNeighbors < 2) {
                                if (Math.random() < 0.1) {
                                    newState = 0; // Décès par la Faim
                                    newDensity *= 0.5;
                                }
                            }
                            // Tolérance stricte à l'isolement/surpopulation
                            else if (sameState > 8.0 || sameState < 1.0) {
                                if (Math.random() < 0.05) newState = 0; // Décès par conditions hostiles
                                newDensity *= 0.9;
                            }
                            // Mort naturelle aléatoire
                            else if (Math.random() < 0.002) { // 0.2% de mort naturelle
                                newState = 0;
                                newDensity = 0.0;
                            }
                            // Prospérité : La densité augmente doucement en vieillissant
                            else {
                                newDensity = Math.min(1.0, newDensity + 0.02);
                            }
                        }
                    }

                    next[idx] = newState;
                    density[idx] = newDensity;
                }
            } // <- End of double for loop (y & x)

            // 4. Diffusion d'état organique (Fluidité)
            // Mélange occasionnellement les bordures pour créer de belles lignes organiques courbes
            if (Math.random() < 0.2) {
                for (let y = 1; y < ny - 1; y++) {
                    for (let x = 1; x < nx - 1; x++) {
                        const idx = zOff + y * nx + x;
                        if (Math.random() < 0.01) {
                            const dx = (Math.random() < 0.5 ? -1 : 1);
                            const dy = (Math.random() < 0.5 ? -1 : 1);
                            const nidx = zOff + (y + dy) * nx + (x + dx);
                            if (current[nidx] !== current[idx] && Math.random() < 0.3) {
                                next[nidx] = current[idx];
                            }
                        }
                    }
                }
            }
        } // <- End of nz loop

        // Swap / Recopie mémoire ultra-rapide de l'état (t+1) vers (t)
        current.set(next);
    }

    private getWgslSource(): string {
        return `
            struct Uniforms {
                nx: u32, ny: u32, nz: u32, strideFace: u32,
                eatThresholdBase: f32, plantEatThreshold: f32, herbiEatThreshold: f32, 
                carniEatThreshold: f32, carniStarveThreshold: f32, frameCounter: u32,
                pad1: u32, pad2: u32, pad3: u32, pad4: u32, pad5: u32, pad6: u32
            };

            @group(0) @binding(0) var<storage, read_write> cube: array<f32>;
            @group(0) @binding(1) var<uniform> config: Uniforms;

            // Simple hash function for PRNG
            fn rand(seed: u32) -> f32 {
                var h = seed ^ config.frameCounter;
                h = (h ^ 61u) ^ (h >> 16u);
                h = h + (h << 3u);
                h = h ^ (h >> 4u);
                h = h * 0x27d4eb2du;
                h = h ^ (h >> 15u);
                return f32(h) * (1.0 / 4294967296.0);
            }

            @compute @workgroup_size(16, 16, 1)
            fn compute_gol(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let x = global_id.x;
                let y = global_id.y;
                let z = global_id.z;

                if (x >= config.nx || y >= config.ny || z >= config.nz) { return; }
                let idx = z * config.nx * config.ny + y * config.nx + x;

                let stride = config.strideFace;
                let stateFace = stride * 1u;
                let nextFace = stride * 2u;
                let densityFace = stride * 3u;

                let state = u32(cube[stateFace + idx]);
                let targetState = (state + 1u) % 4u;
                var densityFactor = cube[densityFace + idx];

                var sameState = 0.0;
                var predators = 0.0;
                var plantNeighbors = 0;

                // Periodic boundary helpers
                let topY = select(y - 1u, config.ny - 1u, y == 0u);
                let botY = select(y + 1u, 0u, y == config.ny - 1u);
                let leftX = select(x - 1u, config.nx - 1u, x == 0u);
                let rightX = select(x + 1u, 0u, x == config.nx - 1u);

                let zOff = z * config.nx * config.ny;

                // Neighbors coordinates
                let coords = array<vec2<u32>, 8>(
                    vec2<u32>(x, topY), vec2<u32>(x, botY), vec2<u32>(leftX, y), vec2<u32>(rightX, y),
                    vec2<u32>(leftX, topY), vec2<u32>(rightX, topY), vec2<u32>(leftX, botY), vec2<u32>(rightX, botY)
                );
                
                let weights = array<f32, 8>(1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0);

                for (var i = 0u; i < 8u; i = i + 1u) {
                    let nx = coords[i].x;
                    let ny = coords[i].y;
                    let nIdx = zOff + ny * config.nx + nx;
                    let nState = u32(cube[stateFace + nIdx]);

                    if (nState == state) { sameState += weights[i]; }
                    if (nState == targetState) { predators += weights[i]; }
                    
                    if (i < 4u && nState == 1u) { plantNeighbors++; }
                }

                var newState = state;
                var newDensity = densityFactor;

                var eatThreshold = config.eatThresholdBase;
                if (state == 0u) { eatThreshold = config.plantEatThreshold; }
                else if (state == 1u) { eatThreshold = config.herbiEatThreshold; eatThreshold += densityFactor * 1.5; }
                else if (state == 2u) { eatThreshold = config.carniEatThreshold; eatThreshold += densityFactor * 0.8; }
                else if (state == 3u) { eatThreshold = config.carniStarveThreshold; eatThreshold += densityFactor * 0.4; }

                let rnd = rand(idx);

                if (predators >= eatThreshold) {
                    newState = targetState;
                    newDensity = 0.2 + rnd * 0.2;
                } else {
                    if (state == 0u) {
                        if (rnd < 0.0005) {
                            newState = 1u;
                            newDensity = 0.1;
                        } else {
                            newDensity = 0.0;
                        }
                    } else {
                        if (state == 2u && plantNeighbors < 2) {
                            if (rnd < 0.1) {
                                newState = 0u;
                                newDensity *= 0.5;
                            }
                        } else if (sameState > 8.0 || sameState < 1.0) {
                            if (rnd < 0.05) { newState = 0u; }
                            newDensity *= 0.9;
                        } else if (rnd < 0.002) {
                            newState = 0u;
                            newDensity = 0.0;
                        } else {
                            newDensity = min(1.0, newDensity + 0.02);
                        }
                    }
                }

                // Diffusion organique occassionnelle
                if (rnd < 0.002 && x > 0u && x < config.nx - 1u && y > 0u && y < config.ny - 1u) {
                    let randDir = u32(rnd * 10000.0) % 4u;
                    var dx = 0u; var dy = 0u;
                    if (randDir == 0u) { dx = 1u; } else if (randDir == 1u) { dx = 4294967295u; }
                    if (randDir == 2u) { dy = 1u; } else if (randDir == 3u) { dy = 4294967295u; }
                    let nIdx = zOff + (y + dy) * config.nx + (x + dx);
                    if (u32(cube[stateFace + nIdx]) != state && rand(nIdx) < 0.3) {
                        newState = state;
                        // Ne remplace pas le dest, mais s'étend
                    }
                }

                cube[nextFace + idx] = f32(newState);
                cube[densityFace + idx] = newDensity;
            }

            @compute @workgroup_size(256)
            fn copy_faces(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                let total_size = config.nx * config.ny * config.nz;
                if (idx >= total_size) { return; }
                
                let stride = config.strideFace;
                cube[stride * 1u + idx] = cube[stride * 2u + idx];
            }
        `;
    }
}




































