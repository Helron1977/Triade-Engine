import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { FluidEngine } from '../src/engines/FluidEngine';

// Création d'un buffer de 1MB
const masterBuffer = new HypercubeMasterBuffer(1024 * 1024);

async function runFluidSim() {
    // Création d'une Grid 64x64, 6 faces, mode 'cpu' séquentiel
    const grid = await HypercubeGrid.create(
        1, 1, 64, masterBuffer,
        () => new FluidEngine(1.0, 0.4, 0.98),
        6, false, 'cpu', false
    );

    const engine = grid.cubes[0][0]?.engine as FluidEngine;
    const faces = grid.cubes[0][0]?.faces!;

    // Boucle de simulation de 100 steps
    for (let step = 0; step < 100; step++) {
        // Splat de densité et chaleur au bas-centre (y=60, x=32 +/- radius 10)
        engine.addSplat(faces, 64, 32, 60, 0, 0, 10, 1.0, 5.0);

        // Calcul d'une étape
        await grid.compute();
    }

    // Vérification de la somme finale
    const finalDensitySum = engine.getTotalDensity(faces);
    console.log(`Simulation de fluide terminée. Somme densité finale: ${finalDensitySum}`);
}

runFluidSim();
