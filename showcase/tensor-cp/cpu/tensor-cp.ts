import { HypercubeNeoFactory } from '../../../../core/HypercubeNeoFactory';

async function launch() {
    const factory = new HypercubeNeoFactory();
    const urlParams = new URLSearchParams(window.location.search);
    const isGPU = urlParams.get('backend') === 'gpu';
    const manifestUrl = isGPU ? './manifest-tensor-cp-gpu.json' : './manifest-tensor-cp-cpu.json';
    
    console.log(`Tensor-CP: Loading manifest from ${manifestUrl}`);
    const manifest = await factory.fromManifest(manifestUrl);
    
    // Override rank if present in URL
    const requestedRank = parseInt(urlParams.get('rank') || '0');
    if (requestedRank > 0) {
        manifest.config.params.rank = requestedRank;
        manifest.engine.rules[0].params!.rank = requestedRank;
    }

    const engine = await factory.build(manifest.config, manifest.engine);
    const bridge = (engine as any).bridge;
    const { nx, ny, nz } = manifest.config.dimensions;
    const rank = manifest.config.params.rank;

    console.log(`Grid: ${nx}x${ny}x${nz}, Rank: ${rank}, Mode: ${manifest.config.mode}`);

    // 1. Random Initialization of Factors (Faces 0, 1, 2)
    // We access the raw Float32Array views from the bridge
    const chunkId = (engine as any).vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    
    // Initialize mode_a, mode_b, mode_c with small random values
    for (let f = 0; f < 3; f++) {
        const view = views[f];
        for (let i = 0; i < view.length; i++) {
            view[i] = Math.random() * 0.1;
        }
    }

    // 2. Load CSV Data into 'target' (Face 3)
    try {
        const csvPath = "../../assets/tensor-sample.csv";
        const response = await fetch(csvPath);
        const text = await response.text();
        const lines = text.trim().split('\n');
        const target = views[3];
        target.fill(0); // Clear

        // Skip header: user_id,film_id,genre_id,value
        for (let i = 1; i < lines.length; i++) {
            const [u, f, g, val] = lines[i].split(',').map(Number);
            if (!isNaN(u) && u < nx && f < ny && g < nz) {
                const idx = u + f * nx + g * nx * ny;
                target[idx] = val;
            }
        }
        console.log(`Tensor-CP: Loaded ${lines.length - 1} entries from CSV.`);
    } catch (e) {
        console.warn("Tensor-CP: Failed to load CSV, using default zeros.", e);
    }

    let iteration = 0;
    const maxIter = manifest.config.params.maxIterations || 30;

    async function loop() {
        if (iteration >= maxIter) {
            console.log("Tensor-CP: Decomposition Complete.");
            return;
        }
        
        await engine.step(iteration);
        iteration++;
        
        // In a real implementation, we would compute the MSE here
        // For the demo, we show the iteration progress
        console.log(`Iteration ${iteration}/${maxIter}`);
        
        requestAnimationFrame(loop);
    }
    loop();
}

launch();
