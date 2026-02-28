import type { ITriadeEngine } from "./ITriadeEngine";

export class EcosystemEngineO1 implements ITriadeEngine {
    public get name(): string {
        return "Guerre des Triades (Rouge vs Bleu)";
    }

    public compute(faces: Float32Array[], mapSize: number): void {
        const current = faces[1];
        const next = faces[2];

        // Automate Cellulaire Combat (Jeu de la vie + Bataille de Factions)
        // STRICT O(1) : Sans mémoire asymétrique ni allocation.
        for (let y = 0; y < mapSize; y++) {
            for (let x = 0; x < mapSize; x++) {
                const idx = y * mapSize + x;
                const state = current[idx];

                let blues = 0;
                let reds = 0;

                const yM = y > 0 ? y - 1 : mapSize - 1;
                const yP = y < mapSize - 1 ? y + 1 : 0;
                const xM = x > 0 ? x - 1 : mapSize - 1;
                const xP = x < mapSize - 1 ? x + 1 : 0;

                // Comptage de voisinage (Moore)
                const n1 = current[yM * mapSize + xM]; if (n1 === 2) blues++; else if (n1 === 3) reds++;
                const n2 = current[yM * mapSize + x]; if (n2 === 2) blues++; else if (n2 === 3) reds++;
                const n3 = current[yM * mapSize + xP]; if (n3 === 2) blues++; else if (n3 === 3) reds++;
                const n4 = current[y * mapSize + xM]; if (n4 === 2) blues++; else if (n4 === 3) reds++;
                const n5 = current[y * mapSize + xP]; if (n5 === 2) blues++; else if (n5 === 3) reds++;
                const n6 = current[yP * mapSize + xM]; if (n6 === 2) blues++; else if (n6 === 3) reds++;
                const n7 = current[yP * mapSize + x]; if (n7 === 2) blues++; else if (n7 === 3) reds++;
                const n8 = current[yP * mapSize + xP]; if (n8 === 2) blues++; else if (n8 === 3) reds++;

                const total = blues + reds;

                if (state !== 2 && state !== 3) { // Vide (ou anciens restes de map végétale)
                    // Règle de Naissance GOL (Exactement 3 parents vivants)
                    if (total === 3) {
                        next[idx] = (blues > reds) ? 2 : 3; // L'Allégeance de l'enfant va au camp majoritaire
                    }
                    // Renforts aéroportés aléatoires pour garantir l'agitation infinie (Noise factor)
                    else if (Math.random() < 0.0005) {
                        next[idx] = Math.random() < 0.5 ? 2 : 3;
                    }
                    else {
                        next[idx] = 0; // Reste un champ de bataille vide
                    }
                }
                else if (state === 2) { // Troupe BLEUE
                    // Combat: Frappe fatale ! Si 2 Rouges sont au contact, le Bleu se fait annihiler
                    if (reds >= 2) next[idx] = 0;
                    // Règle de Survie GOL classique (2 ou 3 camarades de vie)
                    else if (total === 2 || total === 3) next[idx] = 2;
                    // Isolement, ou Étouffement par surpopulation GOL
                    else next[idx] = 0;
                }
                else if (state === 3) { // Troupe ROUGE
                    // Combat: Frappe fatale ! Si 2 Bleus sont au contact, le Rouge se fait annihilé
                    if (blues >= 2) next[idx] = 0;
                    // Règle de Survie GOL classique
                    else if (total === 2 || total === 3) next[idx] = 3;
                    // Isolement ou Surpopulation
                    else next[idx] = 0;
                }
            }
        }

        current.set(next); // Synchronisation du Front O(1)
    }
}
