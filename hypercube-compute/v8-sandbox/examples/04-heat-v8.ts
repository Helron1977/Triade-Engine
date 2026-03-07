import { HypercubeFactory } from '../core/HypercubeFactory';
import { HeatDiffusionV8 } from '../engines/HeatDiffusionV8';
import { Circle, Box } from '../core/Shapes';

/**
 * V8 Ideal Hand-on Example
 */
async function main() {
    // 1. Instantiation Déclarative (Le Monde Idéal)
    const proxy = await HypercubeFactory.instantiate(HeatDiffusionV8, {
        dimensions: {
            nx: 512,
            ny: 256,
            chunks: [4, 1]
        },
        // CONDITIONS AUX LIMITES PARFAITES
        boundaries: {
            left: { role: 'inlet', value: 100 }, // Source de chaleur à 100°
            right: { role: 'outlet', factor: 0.1 }, // Sortie avec 90% d'absorption
            top: { role: 'wall' },    // Mur isolant (rebond complet)
            bottom: { role: 'joint' } // Connecté à un voisin (si existe)
        },
        // INITIALISATION DU MONDE (DÉCLARATIF - SHAPE ACTORS)
        initialState: [
            new Circle({ x: 128, y: 128, z: 0 }, 40, {
                'Obstacles': { role: 'wall', factor: 0.9 } // Un cylindre qui est un mur physique
            }),
            new Box({ x: 400, y: 128, z: 0 }, 50, 50, 1, {
                'Temperature': { role: 'inlet', value: 100 } // Un bloc qui est une source de chaleur
            })
        ],
        params: {
            diffusionRate: 0.05,
        },
        mode: 'gpu'
    });

    // 2. Interaction Sémantique (Plus d'indices !)
    console.log(`Grid V8 Ready: ${proxy.nx}x${proxy.ny}`);

    // Ex: Mise à jour d'un paramètre en cours de route
    proxy.setParam('diffusionRate', 0.04);

    // 3. Rendu Automatique par Contrat
    // Hypercube.autoRender(proxy.grid, canvas); 
}

main();
