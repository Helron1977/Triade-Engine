import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { GameOfLifeEngine } from '../src/engines/GameOfLifeEngine';
import { HeatmapEngine } from '../src/engines/HeatmapEngine';

async function runEcosystemDemo() {
    const mapSize = 128;
    // We need 6 faces for Ecosystem + 6 faces for Heatmap = 12 faces worth of 128x128 Floats
    const masterBuffer = new HypercubeMasterBuffer(mapSize * mapSize * 12 * 4);

    // 1. Initialiser le GameOfLifeEngine (Écosystème)
    const ecosystemEngine = new GameOfLifeEngine({
        deathProb: 0.015,
        growthProb: 0.03,
        eatThresholdBase: 3.5,
        plantEatThreshold: 2.8,
        herbiEatThreshold: 3.8,
        carniEatThreshold: 3.2,
        carniStarveThreshold: 3.5
    });
    const gridEco = await HypercubeGrid.create(
        1, 1, mapSize, masterBuffer,
        () => ecosystemEngine,
        6, true, 'cpu', false
    );

    // 2. Initialiser le HeatmapEngine pour "flouter" visuellement la densité (Face 3 -> Face 2)
    const heatmapEngine = new HeatmapEngine(8, 0.1);
    const gridHeat = await HypercubeGrid.create(
        1, 1, mapSize, masterBuffer,
        () => heatmapEngine,
        6, true, 'cpu', false
    );

    const faces = gridEco.cubes[0][0]?.faces!;

    // INITIALISATION ALÉATOIRE
    const totalCells = mapSize * mapSize;
    for (let i = 0; i < totalCells; i++) {
        if (Math.random() > 0.9) {
            faces[1][i] = 1; // Quelques plantes au départ
            faces[3][i] = 1.0; // Poussées à fond
        }
    }

    console.log("🌱 Démarrage de la simulation d'Écosystème Dynamique...");

    console.time("100 steps Ecosystem + Heatmap");
    // SIMULATION LOOP
    for (let i = 0; i < 100; i++) {
        // Step 1: Simuler la logique de survie (Écrit l'état sur face 2, et la densité sur face 3)
        await gridEco.compute();

        // Transfer state 2 to 1 done internally by EcosystemEngine
        // Face 3 now contains the raw blocky density maps.

        // Step 2: "Blur / Heatmap" la densité. The Heatmap engine automatically reads from face 1 and writes to face 2 of ITS OWN grid chunk.
        // So we copy the density from ecosystem face 3 to heatmap face 1.
        const heatmapFaces = gridHeat.cubes[0][0]?.faces!;
        heatmapFaces[1].set(faces[3]);

        await gridHeat.compute();

        if (i === 99) {
            let activeCells = 0;
            let totalDensity = 0;
            for (let j = 0; j < totalCells; j++) {
                if (faces[1][j] !== 0) {
                    activeCells++;
                    totalDensity += faces[3][j];
                }
            }
            console.log(`Step 100: ${activeCells} cellules vivantes. Densité Ecosystem moyenne: ${(totalDensity / activeCells).toFixed(2)}`);
            console.log(`Densité "Floutée" Heatmap (center): ${heatmapFaces[2][(mapSize / 2) * mapSize + (mapSize / 2)].toFixed(4)}`);
        }
    }
    console.timeEnd("100 steps Ecosystem + Heatmap");

    console.log("\n💡 Faces prêtes pour le Compositor :");
    console.log("   gridEco Face 1 : État Entier (1=Plante, 2=Herbi, 3=Carni)");
    console.log("   gridHeat Face 2: Densité Floutée spatialement pour un rendu organique !");
}

runEcosystemDemo();
