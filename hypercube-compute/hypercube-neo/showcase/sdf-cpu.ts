import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { CanvasAdapterNeo } from '../io/CanvasAdapterNeo';
import { BenchmarkHUD } from '../io/BenchmarkHUD';

// Leaflet Global
declare const L: any;

// Types for our raw GeoJSON-like data
interface POI { id: string; x: number; y: number; }
interface OsmNode { type: string; id: number; lat: number; lon: number; tags?: any; }
interface OsmWay { type: string; id: number; nodes: number[]; tags?: any; }
type OsmElement = OsmNode | OsmWay;
interface ParisData { elements: OsmElement[] }

// Helper Map
const MAPPINGS: { [key: string]: { tags: Record<string, string | string[]>, weight: string, dist: string } } = {
    metro: { tags: { 'station': 'subway' }, weight: 'slider-metro-weight', dist: 'slider-metro-dist' },
    school: { tags: { 'amenity': ['school', 'university', 'college'] }, weight: 'slider-school-weight', dist: 'slider-school-dist' },
    park: { tags: { 'leisure': ['park', 'garden', 'pitch'] }, weight: 'slider-park-weight', dist: 'slider-park-dist' },
    hospital: { tags: { 'amenity': ['hospital', 'clinic'] }, weight: 'slider-hospital-weight', dist: 'slider-hospital-dist' },
    shop: { tags: { 'shop': ['supermarket', 'convenience', 'bakery', 'mall'] }, weight: 'slider-shop-weight', dist: 'slider-shop-dist' },
    water: { tags: { 'natural': 'water', 'waterway': ['river', 'canal'] }, weight: 'slider-water-weight', dist: 'slider-water-dist' }
};

const BBOX = { minLat: 48.860, minLon: 2.360, maxLat: 48.870, maxLon: 2.375 };
const SIZE = 512;

function checkTag(elementTags: any, rules: Record<string, string | string[]>): boolean {
    if (!elementTags) return false;
    for (const [key, val] of Object.entries(rules)) {
        if (elementTags[key]) {
            if (Array.isArray(val) && val.includes(elementTags[key])) return true;
            if (val === elementTags[key]) return true;
        }
    }
    return false;
}

// Global Injector for exact Voronoi coordinates [X,Y]
function injectVoronoiSeeds(engine: any, faceBaseName: string, pois: POI[]) {
    const vGrid = engine.vGrid;
    const nChunksX = vGrid.config.chunks.x;
    const nChunksY = vGrid.config.chunks.y;
    const vnx = SIZE / nChunksX;
    const vny = SIZE / nChunksY;

    // Default initialization is heavily out of bounds (-10000)
    const initialSeedX = new Float32Array(SIZE * SIZE).fill(-10000);
    const initialSeedY = new Float32Array(SIZE * SIZE).fill(-10000);

    for (const p of pois) {
        if (p.x >= 0 && p.x < SIZE && p.y >= 0 && p.y < SIZE) {
            initialSeedX[p.y * SIZE + p.x] = p.x;
            initialSeedY[p.y * SIZE + p.x] = p.y;
        }
    }

    // Distribute X & Y to chunks
    for (const chunk of vGrid.chunks) {
        const nxPhys = vnx + 2;
        const nyPhys = vny + 2;
        const chunkDataX = new Float32Array(nxPhys * nyPhys).fill(-10000);
        const chunkDataY = new Float32Array(nxPhys * nyPhys).fill(-10000);

        const worldXOffset = chunk.x * vnx;
        const worldYOffset = chunk.y * vny;

        for (let ly = 1; ly < nyPhys - 1; ly++) {
            const worldY = worldYOffset + (ly - 1);
            const dstRowOffset = ly * nxPhys;
            const srcRowOffset = worldY * SIZE;
            for (let lx = 1; lx < nxPhys - 1; lx++) {
                const worldX = worldXOffset + (lx - 1);
                chunkDataX[dstRowOffset + lx] = initialSeedX[srcRowOffset + worldX];
                chunkDataY[dstRowOffset + lx] = initialSeedY[srcRowOffset + worldX];
            }
        }
        engine.bridge.setFaceData(chunk.id, `${faceBaseName}_x`, chunkDataX);
        engine.bridge.setFaceData(chunk.id, `${faceBaseName}_y`, chunkDataY);
    }
}

async function bootSDF() {
    const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;
    const btnRecompute = document.getElementById('btn-recompute') as HTMLButtonElement;
    const hud = new BenchmarkHUD('SDF Jump Flooding (O(1))', `${SIZE}x${SIZE}`);

    // === LEAFLET MAP INITIALIZATION ===
    const map = L.map('leaflet-map', {
        zoomControl: false,
        attributionControl: false
    }).setView([(BBOX.minLat + BBOX.maxLat) / 2, (BBOX.minLon + BBOX.maxLon) / 2], 15);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        subdomains: 'abcd',
        maxZoom: 19,
        crossOrigin: true
    }).addTo(map);

    // Sync our transparent canvas over the absolute BBOX!
    const imageBounds = [[BBOX.minLat, BBOX.minLon], [BBOX.maxLat, BBOX.maxLon]];

    // To make sure the Canvas syncs with Leaflet zoom/pan, we use L.imageOverlay with a dummy initial url,
    // but actually we position the canvas wrapper directly. Wait, the easiest way is to let Leaflet handle the overlay!
    // But since CanvasAdapterNeo requires a <canvas> reference to write into, we'll keep the DOM canvas and sync its CSS.

    function syncCanvas() {
        const topLeft = map.latLngToContainerPoint([BBOX.maxLat, BBOX.minLon]);
        const bottomRight = map.latLngToContainerPoint([BBOX.minLat, BBOX.maxLon]);
        canvas.style.position = 'absolute';
        canvas.style.left = `${topLeft.x}px`;
        canvas.style.top = `${topLeft.y}px`;
        canvas.style.width = `${bottomRight.x - topLeft.x}px`;
        canvas.style.height = `${bottomRight.y - topLeft.y}px`;
        canvas.style.objectFit = 'fill'; // Prevent letterboxing
        canvas.style.transformOrigin = 'top left';
    }

    map.on('zoom zoomend move moveend', syncCanvas);
    syncCanvas();

    const resManifest = await fetch('./showcase-sdf-cpu.json?v=' + Date.now());
    const manifest = await resManifest.json();

    console.log("Loading Urban Data...");
    const resData = await fetch('./assets/paris_data.json?v=' + Date.now());
    const geoData: ParisData = await resData.json();

    const factory = new HypercubeNeoFactory();
    const engine = await factory.build(manifest.config, manifest.engine);
    console.log("Hypercube Neo Engine initialized!");

    // Data Parsing
    const pois: Record<string, POI[]> = { metro: [], school: [], park: [], hospital: [], shop: [], water: [] };
    const nodes = new Map<number, { lon: number, lat: number }>();

    for (const el of geoData.elements) {
        if (el.type === 'node') {
            const node = el as OsmNode;
            nodes.set(node.id, { lon: node.lon, lat: node.lat });

            const px = Math.floor(((node.lon - BBOX.minLon) / (BBOX.maxLon - BBOX.minLon)) * SIZE);
            const py = Math.floor((1.0 - (node.lat - BBOX.minLat) / (BBOX.maxLat - BBOX.minLat)) * SIZE);

            for (const [key, mapping] of Object.entries(MAPPINGS)) {
                if (checkTag(node.tags, mapping.tags)) {
                    pois[key].push({ id: node.id.toString(), x: px, y: py });
                }
            }
        }
    }

    // Centroids for polygons (parks, water, hospitals)
    for (const el of geoData.elements) {
        if (el.type === 'way') {
            const way = el as OsmWay;
            for (const [key, mapping] of Object.entries(MAPPINGS)) {
                if (checkTag(way.tags, mapping.tags)) {
                    let cx = 0, cy = 0, count = 0;
                    for (const nid of way.nodes) {
                        const n = nodes.get(nid);
                        if (n) {
                            cx += ((n.lon - BBOX.minLon) / (BBOX.maxLon - BBOX.minLon)) * SIZE;
                            cy += (1.0 - (n.lat - BBOX.minLat) / (BBOX.maxLat - BBOX.minLat)) * SIZE;
                            count++;
                        }
                    }
                    if (count > 0) pois[key].push({ id: way.id.toString(), x: Math.floor(cx / count), y: Math.floor(cy / count) });
                }
            }
        }
    }

    console.log("POIs Categorized:", Object.fromEntries(Object.entries(pois).map(([k, v]) => [k, v.length])));

    // Obstacles
    const obstaclesData = new Float32Array(SIZE * SIZE);
    const hidCanvas = document.getElementById('hidden-canvas') as HTMLCanvasElement;
    const ctx = hidCanvas.getContext('2d', { willReadFrequently: true })!;
    ctx.fillStyle = "black"; ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = "white"; // White = Obstacle (blocks distance propagation)

    for (const el of geoData.elements) {
        if (el.type === 'way' && (el as OsmWay).tags?.building) {
            ctx.beginPath();
            let first = true;
            for (const nid of (el as OsmWay).nodes) {
                const n = nodes.get(nid);
                if (n) {
                    const px = ((n.lon - BBOX.minLon) / (BBOX.maxLon - BBOX.minLon)) * SIZE;
                    const py = (1.0 - (n.lat - BBOX.minLat) / (BBOX.maxLat - BBOX.minLat)) * SIZE;
                    if (first) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                    first = false;
                }
            }
            ctx.closePath();
            ctx.fill();
        }
    }

    const imgData = ctx.getImageData(0, 0, SIZE, SIZE).data;
    for (let i = 0; i < SIZE * SIZE; i++) obstaclesData[i] = imgData[i * 4] > 128 ? 1.0 : 0.0;



    // Correct Chunk Obstacle Insertion
    const nChunksX = manifest.config.chunks.x;
    for (const chunk of engine.vGrid.chunks) {
        const vnx = SIZE / nChunksX;
        const vny = SIZE / nChunksX;
        const nxPhys = vnx + 2;
        const chunkData = new Float32Array(nxPhys * nxPhys);
        const wXOffset = chunk.x * vnx;
        const wYOffset = chunk.y * vny;
        for (let ly = 1; ly < nxPhys - 1; ly++) {
            const wY = wYOffset + (ly - 1);
            for (let lx = 1; lx < nxPhys - 1; lx++) {
                const wX = wXOffset + (lx - 1);
                chunkData[ly * nxPhys + lx] = obstaclesData[wY * SIZE + wX];
            }
        }
        engine.bridge.setFaceData(chunk.id, 'obstacles', chunkData);
    }

    // Inject exact POI coordinates
    for (const key of Object.keys(pois)) {
        injectVoronoiSeeds(engine, `sdf_${key}`, pois[key]);
    }

    // Synchronize to populate ghost cells
    (engine as any).synchronizer.syncAll(engine.vGrid, engine.bridge, engine.parityManager, 'read');

    // === SDF BAKE ALGORITHM (Eikonal / Dilation) ===
    // Since we rely on 1-cell ghost overlap between chunks for O(1) multi-chunk scaling, 
    // we cannot use true Jump Flooding (which requires reading N/2 cells away).
    // Instead, we use iterative 1-pixel dilation. Since this is a one-time bake, 
    // running 512 passes is perfectly acceptable on CPU (~150ms total).

    async function bakeSDF() {
        btnRecompute.textContent = "Baking Distance Field (Max 700 passes)...";
        btnRecompute.disabled = true;

        // The maximum possible distance in a 512x512 grid is diagonal ~ 724
        const maxDiagonal = Math.ceil(Math.sqrt(SIZE * SIZE + SIZE * SIZE));
        const passes = Math.min(750, maxDiagonal);

        // We run batches of passes to avoid blocking the UI thread for too long
        const batchSize = 50;

        for (let batch = 0; batch < passes; batch += batchSize) {
            const currentBatch = Math.min(batchSize, passes - batch);

            for (let i = 0; i < currentBatch; i++) {
                // Advance physics engine by 1 tick (1-pixel dilation)
                await (engine as any).step(1);
                // Synchronize ghost cells using boundary orchestrator
                (engine as any).synchronizer.syncAll(engine.vGrid, engine.bridge, engine.parityManager, 'read');
            }

            // Yield to UI Thread
            btnRecompute.textContent = `Baking... ${Math.round((batch / passes) * 100)}%`;
            await new Promise(r => setTimeout(r, 0));
        }

        console.log("SDF Bake Complete!");
        btnRecompute.textContent = "Bake Finished (SDF Stored)";
    }

    await bakeSDF();

    // Setup Analytics Realtime Rendering Adapter O(1)
    const criteriaIndices = Object.keys(MAPPINGS).map(key => ({
        key,
        xFace: engine.getFaceLogicalIndex(`sdf_${key}_x`),
        yFace: engine.getFaceLogicalIndex(`sdf_${key}_y`),
        uiWeight: document.getElementById(MAPPINGS[key].weight) as HTMLInputElement,
        uiDist: document.getElementById(MAPPINGS[key].dist) as HTMLInputElement,
        uiValWeight: document.getElementById(MAPPINGS[key].weight.replace('slider-', 'val-')),
        uiValDist: document.getElementById(MAPPINGS[key].dist.replace('slider-', 'val-'))
    }));

    const renderOptions: any = {
        obstaclesFace: engine.getFaceLogicalIndex('obstacles'),
        colormap: 'spatial-decision',
        criteriaSDF: []
    };

    function renderLoop() {
        // Dynamically pull the DOM slider values and inject them into the O(1) mathematical overlay
        renderOptions.criteriaSDF = criteriaIndices.map(crit => ({
            xFace: crit.xFace,
            yFace: crit.yFace,
            weight: parseInt(crit.uiWeight.value),
            distanceThreshold: parseInt(crit.uiDist.value)
        }));

        // We DO NOT call engine.compute() here! The Worklets are ASLEEP!
        // We only tell the Adapter to fetch the static pre-baked distances and cross them 
        // with the UI Sliders in O(1) time complexity.
        CanvasAdapterNeo.render(engine as any, canvas, renderOptions);
        hud.tickFrame();
        requestAnimationFrame(renderLoop);
    }

    renderLoop();

    // Event Listeners for UI updates (instantly triggers O(1) re-render logic via reference pulling inside adapter)
    criteriaIndices.forEach(crit => {
        crit.uiWeight.addEventListener('input', (e) => {
            const val = parseInt((e.target as HTMLInputElement).value);
            if (crit.uiValWeight) crit.uiValWeight.textContent = `Poids: ${crit.uiWeight.value}`;
        });
        crit.uiDist.addEventListener('input', (e) => {
            if (crit.uiValDist) crit.uiValDist.textContent = `${(e.target as HTMLInputElement).value}m`;
        });
    });
}

bootSDF().catch(err => console.error("SDF Error:", err));
