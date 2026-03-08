# integration Guide: JavaScript & TypeScript

The **HypercubeNeoFactory** is the entry point for running a Neo Manifest.

## 1. Minimal Setup

```ts
import { HypercubeNeoFactory } from 'hypercube-compute';

async function start() {
    const factory = new HypercubeNeoFactory();
    
    // 1. Load the SSOT Manifest
    const manifest = await factory.fromManifest('./simulation-case-01.json');
    
    // 2. Build the engine (Parallel CPU Workers)
    const engine = await factory.build(manifest.config, manifest.engine);
    
    // 3. Step the physics
    await engine.step(1.0);
}
```

## 2. Hybrid Manifest (Static + Dynamic)

You can load a static JSON and then inject dynamic points (like a procedurally generated wing) before building.

```ts
const manifest = await factory.fromManifest('showcase.json');

// Inject points into a specific object
const wing = manifest.config.objects.find(o => o.id === 'my_wing');
wing.points = generatePoints(); 

// Build with the modified manifest
const engine = await factory.build(manifest.config, manifest.engine);
```

## 3. Rendering

Use `HypercubeNeo.autoRender` to visualize one of the faces defined in the manifest.

```ts
import { HypercubeNeo } from 'hypercube-compute';

// Inside your animation loop
HypercubeNeo.autoRender(engine, canvas, {
    faceIndex: engine.parityManager.getFaceIndices('smoke').read,
    colormap: 'arctic',
    minVal: 0.0,
    maxVal: 1.0,
    obstaclesFace: engine.parityManager.getFaceIndices('obstacles').read
});
```
