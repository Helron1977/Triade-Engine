import { HypercubeGrid, HypercubeMasterBuffer, HypercubeCompositor, GameOfLifeEngine, HeatmapEngine } from 'hypercube-compute';

// Simulation Settings
const MAP_SIZE = 256;
const BLUR_RADIUS = 5;

// WGSL Shader implementing the Organic Colored Blur + Core Ecosystem Visuals
const ecosystemCompositorShader = `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
    );
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
    );
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = uv[vertexIndex];
    return output;
}

@group(0) @binding(0) var<storage, read> cube: array<f32>;
struct Config {
    mapSize: u32,
    stride: u32,
};
@group(0) @binding(1) var<uniform> config: Config;

@fragment
fn fragment_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let size = f32(config.mapSize);
    let x = u32(uv.x * size);
    let y = u32(uv.y * size);
    if (x >= config.mapSize || y >= config.mapSize) { discard; }

    let idx = y * config.mapSize + x;
    
    // Read State from Grid 1 (Ecosystem Face 1)
    let state = u32(cube[1u * config.stride + idx]); 
    
    // Read Blurred Density from Grid 2 (Heatmap Face 2 = Overall Face 8)
    let dens = cube[8u * config.stride + idx];       

    var color: vec3<f32>;

    if (state == 0u)      { color = vec3<f32>(0.05, 0.08, 0.15); }      // Vide : bleu nuit
    else if (state == 1u) { color = vec3<f32>(0.1, 0.8, 0.2) * (0.6 + dens * 0.4); } // Plante : vert lumineux
    else if (state == 2u) { color = vec3<f32>(0.9, 0.7, 0.1) * (0.5 + dens * 0.5); } // Herbi : jaune/orange
    else                  { color = vec3<f32>(0.8, 0.1, 0.3) * (0.4 + dens * 0.6); } // Carni : rouge-violet

    // Ajout d’un léger glow / halo basé sur densité lissée
    let glow = smoothstep(0.0, 1.0, dens) * 0.3;
    color += vec3<f32>(glow);

    return vec4<f32>(color, 1.0);
}
`;

async function init() {
  const canvas = document.getElementById('gpuCanvas') as HTMLCanvasElement;
  if (!canvas) throw new Error("Canvas introuvable.");

  // Ajustement de la taille HD du canvas
  canvas.width = 1024;
  canvas.height = 1024;

  console.log("🚀 Initialisation de l'Écosystème Organique V3...");

  // Allocation VRAM: 1 chunk Eco (6 faces) + 1 chunk Heatmap (6 faces) + 1 chunk GPU Render Proxy (12 faces) = 24 faces
  const masterBuffer = new HypercubeMasterBuffer(MAP_SIZE * MAP_SIZE * 24 * 4);

  // 1. Instanciation du GameOfLifeEngine (Écosystème) avec configurations customisables
  const ecosystemEngine = new GameOfLifeEngine({
    deathProb: 0.005,       // Mort naturelle très rare, c'est la prédation qui motive le système
    growthProb: 0.05,       // Les plantes colonisent un tout petit peu plus vite le vide
    eatThresholdBase: 3.0,
    plantEatThreshold: 2.0, // Herbis mangent facilement les plantes isolées 
    herbiEatThreshold: 2.5, // Carnis mangent facilement les herbis
    carniEatThreshold: 1.5, // Carnis isolés meurent de faim doucement
    carniStarveThreshold: 2.8
  });
  const gridEco = await HypercubeGrid.create(
    1, 1, MAP_SIZE, masterBuffer,
    () => ecosystemEngine,
    6, true, 'cpu', false
  );

  // 2. Instanciation du HeatmapEngine pour flouter la densité (Face 3 -> Sortie sur Face 2 Heatmap, c.a.d config.stride * 8)
  const heatmapEngine = new HeatmapEngine(BLUR_RADIUS, 0.15);
  const gridHeat = await HypercubeGrid.create(
    1, 1, MAP_SIZE, masterBuffer, // Utilise la même VRAM
    () => heatmapEngine,
    6, true, 'cpu', false
  );

  // 3. Grille de Rendu WebGPU (Proxy VRAM)
  // Partage le même MasterBuffer mais en mode 'webgpu' pour instancier les buffers matériels.
  const gridRender = await HypercubeGrid.create(
    1, 1, MAP_SIZE, masterBuffer,
    () => ecosystemEngine,
    12, true, 'webgpu', false
  );

  // Semis initial aléatoire
  const facesEco = gridEco.cubes[0][0]?.faces!;
  for (let i = 0; i < MAP_SIZE * MAP_SIZE; i++) {
    if (Math.random() > 0.95) {
      facesEco[1][i] = 1;      // Plantes
      facesEco[3][i] = 1.0;    // Densité Max
    } else if (Math.random() > 0.98) {
      facesEco[1][i] = 2;      // Quelques herbivores
      facesEco[3][i] = 1.0;
    } else if (Math.random() > 0.99) {
      facesEco[1][i] = 3;      // Très peu de carnivores
      facesEco[3][i] = 1.0;
    }
  }

  // 4. Initialisation du GPU Compositor pour le rendu
  const compositor = new HypercubeCompositor(gridRender, {
    canvas,
    wgslFragmentSource: ecosystemCompositorShader
  });

  await compositor.init();

  // Réf locales pour la boucle ultra optimisée
  // facesEco est déjà déclaré plus haut
  const facesHeat = gridHeat.cubes[0][0]?.faces!;
  const facesRender = gridRender.cubes[0][0]?.faces!;

  let isRunning = true;
  let frameCount = 0;
  let lastTime = performance.now();

  // Event Stop/Play
  canvas.addEventListener('click', () => { isRunning = !isRunning; });

  async function loop() {
    if (isRunning) {
      // STEP 1 : Logique Biologique Discrète (CPU) -> Calcule Face 1 (État) et Face 3 (Densité pure)
      await gridEco.compute();

      // STEP 2 : Transfert de la Densité pure (Face 3 Eco) vers l'Input du Heatmap (Face 1 Heat)
      // L'écosystème modifie les données, nous informons le Heatmap du champ.
      facesHeat[1].set(facesEco[3]);

      // STEP 3 : Lissage Spatial SAT O(1) (CPU) -> Produit la Density Floutée dans Face 2 Heat
      await gridHeat.compute();
    }

    // TRUC AVANCÉ : Les 6 premières faces sont L'Eco, les 6 suivantes sont L'Heatmap.
    // Transférons les maps vers la VRAM proxy qui seront copiées vers le GPU 
    // Le Shader lit: Etat -> cube[1], Blur Density -> cube[8]

    facesRender[1].set(facesEco[1]);
    facesRender[8].set(facesHeat[2]);

    // Upload RAM -> VRAM
    gridRender.cubes[0][0]?.syncFromHost();

    // Execution du WGSL
    await compositor.render();

    frameCount++;
    if (performance.now() - lastTime > 1000) {
      console.log(`FPS: ${frameCount}`);
      frameCount = 0;
      lastTime = performance.now();
    }

    requestAnimationFrame(loop);
  }

  loop();
  console.log("✅ Rendu Compositor branché.");
}

init().catch(console.error);
