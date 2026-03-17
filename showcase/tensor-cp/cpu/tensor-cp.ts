import { HypercubeNeoFactory } from '../../core/HypercubeNeoFactory';

async function launch() {
    const factory = new HypercubeNeoFactory();
    const isGPU = window.location.search.includes('backend=gpu');
    const manifestUrl = isGPU ? './manifest-tensor-cp-gpu.json' : './manifest-tensor-cp-cpu.json';
    
    console.log(`Tensor-CP: Loading manifest from ${manifestUrl}`);
    const manifest = await factory.fromManifest(manifestUrl);
    const engine = await factory.build(manifest.config, manifest.engine);

    // Initial random factors injection (simplified)
    const bridge = (engine as any).bridge;
    const { nx, ny, nz } = manifest.config.dimensions;
    const rank = manifest.engine.params.rank;

    // TODO: Implement CSV loading here and inject into 'target' face

    let iteration = 0;
    async function loop() {
        if (iteration >= manifest.engine.params.maxIterations) return;
        
        await engine.step();
        iteration++;
        
        // Compute Reconstruction Error (Dummy for now)
        const error = Math.random() * 0.1 / iteration;
        console.log(`Iteration ${iteration}: Error = ${error.toFixed(6)}`);
        
        requestAnimationFrame(loop);
    }
    loop();
}

launch();
