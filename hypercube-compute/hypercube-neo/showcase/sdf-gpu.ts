import { HypercubeNeoFactory } from '../core/HypercubeNeoFactory';
import { CanvasAdapterNeo } from '../io/CanvasAdapterNeo';
import { BenchmarkHUD } from '../io/BenchmarkHUD';
import { WebGpuRendererNeo } from '../io/WebGpuRendererNeo';

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
        const nxPhys = vnx; // No ghosts
        const nyPhys = vny; // No ghosts
        const chunkDataX = new Float32Array(nxPhys * nyPhys).fill(-10000);
        const chunkDataY = new Float32Array(nxPhys * nyPhys).fill(-10000);

        const worldXOffset = chunk.x * vnx;
        const worldYOffset = chunk.y * vny;

        for (let ly = 0; ly < nyPhys; ly++) {
            const worldY = worldYOffset + ly;
            const dstRowOffset = ly * nxPhys;
            const srcRowOffset = worldY * SIZE;
            for (let lx = 0; lx < nxPhys; lx++) {
                const worldX = worldXOffset + lx;
                chunkDataX[dstRowOffset + lx] = initialSeedX[srcRowOffset + worldX];
                chunkDataY[dstRowOffset + lx] = initialSeedY[srcRowOffset + worldX];
            }
        }
        engine.bridge.setFaceData(chunk.id, `${faceBaseName}_x`, chunkDataX, true);
        engine.bridge.setFaceData(chunk.id, `${faceBaseName}_y`, chunkDataY, true);
    }
}

async function bootSDF() {
    const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;
    const btnRecompute = document.getElementById('btn-recompute') as HTMLButtonElement;
    const hud = new BenchmarkHUD('SDF Jump Flooding (GPU)', `${SIZE}x${SIZE}`);

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

    function syncCanvas() {
        const topLeft = map.latLngToContainerPoint([BBOX.maxLat, BBOX.minLon]);
        const bottomRight = map.latLngToContainerPoint([BBOX.minLat, BBOX.maxLon]);
        canvas.style.position = 'absolute';
        canvas.style.left = `${topLeft.x}px`;
        canvas.style.top = `${topLeft.y}px`;
        canvas.style.width = `${bottomRight.x - topLeft.x}px`;
        canvas.style.height = `${bottomRight.y - topLeft.y}px`;
        canvas.style.objectFit = 'fill'; 
        canvas.style.transformOrigin = 'top left';
    }

    map.on('zoom zoomend move moveend', syncCanvas);
    syncCanvas();

    const resManifest = await fetch('./showcase-sdf-gpu.json?v=' + Date.now());
    const manifest = await resManifest.json();

    console.log("Loading Urban Data...");
    const resData = await fetch('./assets/paris_data.json?v=' + Date.now());
    const geoData: ParisData = await resData.json();

    const factory = new HypercubeNeoFactory();
    const engine = await factory.build(manifest.config, manifest.engine);

    // IA Observability (Web MCP)
    const { DebugBridge } = await import('../helpers/DebugBridge');
    DebugBridge.setup(engine, manifest.config);

    console.log("Hypercube Neo GPU Engine initialized!");

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

    // Centroids for polygons
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

    // Obstacles
    const obstaclesData = new Float32Array(SIZE * SIZE);
    const hidCanvas = document.getElementById('hidden-canvas') as HTMLCanvasElement;
    const ctx = hidCanvas.getContext('2d', { willReadFrequently: true })!;
    ctx.fillStyle = "black"; ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = "white"; 

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

    // Correct Chunk Obstacle Insertion (No Ghosts)
    for (const chunk of engine.vGrid.chunks) {
        const vnx = SIZE / manifest.config.chunks.x;
        const vny = SIZE / manifest.config.chunks.y;
        const chunkData = new Float32Array(vnx * vny);
        const wXOffset = chunk.x * vnx;
        const wYOffset = chunk.y * vny;
        
        for (let ly = 0; ly < vny; ly++) {
            const wY = wYOffset + ly;
            for (let lx = 0; lx < vnx; lx++) {
                const wX = wXOffset + lx;
                chunkData[ly * vnx + lx] = obstaclesData[wY * SIZE + wX];
            }
        }
        engine.bridge.setFaceData(chunk.id, 'obstacles', chunkData);
    }

    // Inject exact POI coordinates
    for (const key of Object.keys(pois)) {
        injectVoronoiSeeds(engine, `sdf_${key}`, pois[key]);
    }

    // CRITICAL: Sync injected CPU data to GPU VRAM before baking!
    await engine.bridge.syncToDevice();
    console.log("SDF: Seeds synchronized to VRAM.");

    // Trigger GPU step to ensure initial state
    await engine.step(0); 

    const gpuRenderer = new WebGpuRendererNeo(canvas);

    async function bakeSDF() {
        btnRecompute.textContent = "GPU Baking Distance Field (Jump Flooding)...";
        btnRecompute.disabled = true;

        // Jump Flooding Algorithm (JFA): log2(N) passes
        const passes = Math.ceil(Math.log2(SIZE)); 
        
        for (let p = passes - 1; p >= 0; p--) {
            const step = Math.pow(2, p);
            await engine.step(step);
            
            btnRecompute.textContent = `GPU JFA Pass: ${passes - p} / ${passes}`;
            await new Promise(r => setTimeout(r, 0));
        }

        console.log("GPU SDF JFA Complete!");
        btnRecompute.textContent = "GPU Bake Finished (Direct VRAM)";
    }

    await bakeSDF();

    // Setup Analytics Realtime Rendering
    const criteriaIndices = Object.keys(MAPPINGS).map(key => ({
        key,
        xFace: engine.getFaceLogicalIndex(`sdf_${key}_x`),
        yFace: engine.getFaceLogicalIndex(`sdf_${key}_y`),
        uiWeight: document.getElementById(MAPPINGS[key].weight) as HTMLInputElement,
        uiDist: document.getElementById(MAPPINGS[key].dist) as HTMLInputElement,
        uiValWeight: document.getElementById(MAPPINGS[key].weight.replace('slider-', 'val-')),
        uiValDist: document.getElementById(MAPPINGS[key].dist.replace('slider-', 'val-'))
    }));

    function renderLoop() {
        const renderOptions: any = {
            faceIndex: 0,
            colormap: 'spatial-decision',
            obstaclesFace: engine.getFaceLogicalIndex('obstacles'),
            criteriaSDF: criteriaIndices.map(crit => ({
                xFace: crit.xFace,
                yFace: crit.yFace,
                weight: parseInt(crit.uiWeight.value),
                distanceThreshold: parseInt(crit.uiDist.value)
            }))
        };

        gpuRenderer.render(engine as any, renderOptions);
        hud.tickFrame();
        requestAnimationFrame(renderLoop);
    }

    renderLoop();

    criteriaIndices.forEach(crit => {
        crit.uiWeight.addEventListener('input', (e) => {
            if (crit.uiValWeight) crit.uiValWeight.textContent = `Poids: ${crit.uiWeight.value}`;
        });
        crit.uiDist.addEventListener('input', (e) => {
            if (crit.uiValDist) crit.uiValDist.textContent = `${(e.target as HTMLInputElement).value}m`;
        });
    });
}

bootSDF().catch(err => console.error("GPU SDF Error:", err));
