import { HypercubeNeoFactory } from '../../../../core/HypercubeNeoFactory';

async function launch() {
    const factory = new HypercubeNeoFactory();
    const manifestUrl = './aero-gpu.json';
    
    console.log(`Aero GPU: Loading manifest from ${manifestUrl}`);
    const manifest = await factory.fromManifest(manifestUrl);
    const engine = await factory.build(manifest.config, manifest.engine);

    await engine.init();

    let iteration = 0;
    async function loop() {
        await engine.step(iteration++);
        requestAnimationFrame(loop);
    }
    loop();
}

launch();
