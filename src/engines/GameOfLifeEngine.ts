import type { ITriadeEngine } from "./ITriadeEngine";

export class GameOfLifeEngine implements ITriadeEngine {
    public get name(): string {
        return "Ecosystème Tensoriel (Plantes, Herbivores, Carnivores)";
    }

    public compute(faces: Float32Array[], mapSize: number): void {
        const current = faces[1]; // Face 1: État actuel (t)
        const next = faces[2];    // Face 2: État futur (t+1)

        // Double boucle optimisée pour accès mémoires continus
        for (let y = 0; y < mapSize; y++) {

            const top = (y === 0) ? mapSize - 1 : y - 1;
            const bottom = (y === mapSize - 1) ? 0 : y + 1;

            const topRow = top * mapSize;
            const midRow = y * mapSize;
            const botRow = bottom * mapSize;

            for (let x = 0; x < mapSize; x++) {
                const left = (x === 0) ? mapSize - 1 : x - 1;
                const right = (x === mapSize - 1) ? 0 : x + 1;

                const idx = midRow + x;
                const state = current[idx]; // 0: Vide, 1: Plante, 2: Herbi, 3: Carni

                // Le prédateur / successeur de l'état actuel
                let targetState = state + 1;
                if (targetState > 3) targetState = 0;

                let predators = 0;

                // Von Neumann Neighborhood (Plus organique pour la croissance)
                if (current[topRow + x] === targetState) predators++;
                if (current[midRow + left] === targetState) predators++;
                if (current[midRow + right] === targetState) predators++;
                if (current[botRow + x] === targetState) predators++;

                // Moore Neighborhood (Diagonales) avec moins de poids
                if (current[topRow + left] === targetState) predators++;
                if (current[topRow + right] === targetState) predators++;
                if (current[botRow + left] === targetState) predators++;
                if (current[botRow + right] === targetState) predators++;

                // Seuil à 1 ou 2 selon l'état offre une dynamique d'essaim très organique
                const threshold = (state === 0) ? 1 : 2; // Le vide est colonisé vite, les autres survivent plus

                if (predators >= threshold) {
                    next[idx] = targetState;
                } else {
                    next[idx] = state;
                }
            }
        }

        // Swap / Recopie mémoire ultra-rapide de l'état (t+1) vers (t)
        current.set(next);
    }
}
