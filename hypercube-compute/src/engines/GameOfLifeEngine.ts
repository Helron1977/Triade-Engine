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
        return "Game of Life Ecosystem (O1 Tile)";
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
}




































