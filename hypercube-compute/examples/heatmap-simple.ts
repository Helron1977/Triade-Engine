import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { HeatmapEngine } from '../src/engines/HeatmapEngine';

async function runHeatmapDemo() {
    const mapSize = 256;
    const masterBuffer = new HypercubeMasterBuffer(mapSize * mapSize * 5 * 4);

    // Initialise une grille pour Compute Shader complet
    const grid = await HypercubeGrid.create(
        1, 1, mapSize, masterBuffer,
        () => new HeatmapEngine(20, 0.05), // Rayon de 20 cellules
        5, false, 'cpu', false
    );

    const faces = grid.cubes[0][0]?.faces!;

    // GENERATE RANDOM CROWD DENSITY (Face 1)
    console.log("Génération de 500 agents virtuels...");
    for (let i = 0; i < 500; i++) {
        const x = Math.floor(Math.random() * mapSize);
        const y = Math.floor(Math.random() * mapSize);
        faces[1][y * mapSize + x] += 1.0;
    }

    // RUN HEATMAP ENGINE
    console.time("Heatmap diffusion (CPU)");
    await grid.compute();
    console.timeEnd("Heatmap diffusion (CPU)");

    // ACCEDER AU RÉSULTAT O(1)
    const resultHeatmap = faces[2];

    // Simple Check
    let totalHeat = 0;
    for (let i = 0; i < resultHeatmap.length; i++) {
        totalHeat += resultHeatmap[i];
    }

    console.log(`Heatmap générée avec succès. Chaleur totale lissée : ${totalHeat.toFixed(2)}`);
    console.log("ℹ️ Vous pouvez utiliser resultHeatmap (Face 2) dans HypercubeCompositor pour afficher les zones de densité.");
}

runHeatmapDemo();
