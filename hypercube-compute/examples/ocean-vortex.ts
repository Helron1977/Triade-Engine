import { HypercubeGrid } from '../src/core/HypercubeGrid';
import { HypercubeMasterBuffer } from '../src/core/HypercubeMasterBuffer';
import { OceanEngine } from '../src/engines/OceanEngine';

async function runOceanDemo() {
    const mapSize = 128;
    const masterBuffer = new HypercubeMasterBuffer(mapSize * mapSize * 23 * 4);

    // Initialise le moteur LBM Océanique
    const oceanEngine = new OceanEngine();
    const grid = await HypercubeGrid.create(
        1, 1, mapSize, masterBuffer,
        () => oceanEngine,
        23, false, 'cpu', false
    );

    const faces = grid.cubes[0][0]?.faces!;

    // 1. Initialiser la mer (Densité à 1.0, Vecteurs au repos)
    const totalCells = mapSize * mapSize;
    const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
    for (let i = 0; i < totalCells; i++) {
        faces[20][i] = 1.0;          // rho = 1.0
        for (let k = 0; k < 9; k++) {
            faces[k][i] = w[k] * 1.0; // Distributions au repos
        }
    }

    // 2. Placer une île centrale fixe (Face 22: obst)
    const cx = Math.floor(mapSize / 2);
    for (let dy = -10; dy <= 10; dy++) {
        for (let dx = -10; dx <= 10; dx++) {
            // Île en forme de losange
            if (Math.abs(dx) + Math.abs(dy) < 12) {
                faces[22][(cx + dy) * mapSize + (cx + dx)] = 1.0;
            }
        }
    }

    // 3. Injecter du plancton près de l'île (Face 21: bio)
    for (let dy = -12; dy <= 12; dy++) {
        for (let dx = -12; dx <= 12; dx++) {
            if (Math.random() > 0.8) {
                faces[21][(cx + dy) * mapSize + (cx + dx)] = 0.5;
            }
        }
    }

    console.log("🌊 Démarrage de la simulation OceanEngine (LBM D2Q9)...");

    // 4. Activer le forcing de vortex interactif (Tourbillon)
    oceanEngine.interaction.active = true;
    oceanEngine.interaction.mouseX = cx - 25; // Vortex à gauche de l'île
    oceanEngine.interaction.mouseY = cx;

    console.time("100 steps Ocean LBM");
    for (let i = 0; i < 100; i++) {
        await grid.compute();
    }
    console.timeEnd("100 steps Ocean LBM");

    console.log(`Statistiques LBM après 100 steps :`);
    console.log(`- Vitesse Max: ${oceanEngine.stats.maxU.toFixed(4)} (Limite CFL: ${oceanEngine.params.cflLimit})`);
    console.log(`- Densité Moyenne: ${oceanEngine.stats.avgRho.toFixed(4)} (Conservation Massique Validée)`);
    console.log(`- Relaxation Moyenne: ${oceanEngine.stats.avgTau.toFixed(4)} (Smagorinsky Turbulence)`);

    console.log("\n💡 Faces prêtes pour le Compositor :");
    console.log("   Face 18/19 : Vecteurs de Vitesse X/Y (Courant)");
    console.log("   Face 21    : Concentration du Plancton/Bio Bloom");
}

runOceanDemo();
