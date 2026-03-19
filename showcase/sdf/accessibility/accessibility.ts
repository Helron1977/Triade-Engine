import { HypercubeNeoFactory } from '../../../core/HypercubeNeoFactory';
import { CanvasAdapterNeo } from '../../../io/CanvasAdapterNeo';

// Leaflet Global
declare const L: any;

interface POI { id: string; x: number; y: number; }
interface OsmNode { type: string; id: number; lat: number; lon: number; tags?: any; }
interface OsmWay { type: string; id: number; nodes: number[]; tags?: any; }
interface ParisData { elements: (OsmNode | OsmWay)[] }

const MAPPINGS: Record<string, { tags: Record<string, string | string[]> }> = {
    metro: { tags: { 'station': 'subway', 'railway': 'station' } },
    school: { tags: { 'amenity': ['school', 'university', 'college'] } },
    park: { tags: { 'leisure': ['park', 'garden', 'pitch'], 'natural': 'wood' } },
    hospital: { tags: { 'amenity': ['hospital', 'clinic', 'doctors'] } },
    water: { tags: { 'natural': 'water', 'waterway': ['river', 'canal'] } }
};

let engine: any = null;
let map: any = null;
const SIZE = 512;

/**
 * Fetch POIs from Overpass API based on current map bounds
 */
async function fetchUrbanData(bounds: any) {
    const buffer = 0.02; // ~2km buffer in lat/lon for city scale
    const s = bounds.getSouth() - buffer;
    const w = bounds.getWest() - buffer;
    const n = bounds.getNorth() + buffer;
    const e = bounds.getEast() + buffer;
    const bbox = `${s},${w},${n},${e}`;

    
    // Query for Stations, Schools, Parks, Hospitals, Water, and Buildings
    const query = `
        [out:json][timeout:25];
        (
          node["railway"="station"](${bbox});
          node["amenity"~"school|university|college"](${bbox});
          node["leisure"~"park|garden"](${bbox});
          way["leisure"~"park|garden"](${bbox});
          node["amenity"~"hospital|clinic"](${bbox});
          way["amenity"~"hospital|clinic"](${bbox});
          way["natural"="water"](${bbox});
          way["waterway"~"river|canal"](${bbox});
          way["building"](${bbox});
        );
        out center body;
        >;
        out skel qt;
    `;

    const response = await fetch('https://overpass-api.de/api/interpreter', {
        method: 'POST',
        body: 'data=' + encodeURIComponent(query)
    });
    
    if (!response.ok) throw new Error("Overpass API Error");
    return await response.json();
}


async function boot() {
    const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;
    const btnBake = document.getElementById('btn-recompute') as HTMLButtonElement;
    const loader = document.getElementById('loader');

    // 1. Setup Leaflet
    map = L.map('leaflet-map', { zoomControl: false, attributionControl: false }).setView([48.8566, 2.3522], 14);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19, crossOrigin: true }).addTo(map);

    // 2. Load Engine
    const params = new URLSearchParams(window.location.search);
    const useGPU = params.get('backend') === 'gpu';
    const manifestFile = useGPU ? './manifest-gpu.json' : './manifest.json';
    
    const resManifest = await fetch(manifestFile + '?v=' + Date.now());
    const manifest = await resManifest.json();

    const factory = new HypercubeNeoFactory();
    engine = await factory.build(manifest.config, manifest.engine);

    if (loader) loader.style.opacity = '0';
    setTimeout(() => loader?.remove(), 500);


    // 3. Sync Logic
    function getMapContext() {
        const bounds = map.getBounds();
        const nw = bounds.getNorthWest();
        const se = bounds.getSouthEast();
        const zoom = map.getZoom();

        // Project to world pixels (Web Mercator)
        const nwPoint = map.project(nw, zoom);
        const sePoint = map.project(se, zoom);
        
        const widthPx = sePoint.x - nwPoint.x;
        const heightPx = sePoint.y - nwPoint.y;

        const widthMeters = nw.distanceTo(bounds.getNorthEast());
        const metersPerGridPx = widthMeters / SIZE;

        return {
            nwPoint,
            widthPx,
            heightPx,
            zoom,
            widthMeters,
            metersPerPx: metersPerGridPx
        };
    }

    function syncCanvas() {
        const bounds = map.getBounds();
        const topLeft = map.latLngToContainerPoint(bounds.getNorthWest());
        const bottomRight = map.latLngToContainerPoint(bounds.getSouthEast());
        
        canvas.style.left = `${topLeft.x}px`;
        canvas.style.top = `${topLeft.y}px`;
        canvas.style.width = `${bottomRight.x - topLeft.x}px`;
        canvas.style.height = `${bottomRight.y - topLeft.y}px`;
    }

    async function bakeSDF() {
        if (!engine) return;
        btnBake.disabled = true;
        btnBake.textContent = "Fetching Data...";

        const bounds = map.getBounds();
        const ctx = getMapContext();
        
        try {
            const data = await fetchUrbanData(bounds);
            btnBake.textContent = "Baking Distance Field...";

            const pois: Record<string, POI[]> = { metro: [], school: [], park: [], hospital: [], water: [] };
            const hurdles = new Float32Array(SIZE * SIZE); // Obstacles map

            for (const el of data.elements) {
                const lat = el.center ? el.center.lat : el.lat;
                const lon = el.center ? el.center.lon : el.lon;
                if (!lat || !lon) continue;

                const p = map.project([lat, lon], ctx.zoom);
                const px = Math.floor(((p.x - ctx.nwPoint.x) / ctx.widthPx) * SIZE);
                const py = Math.floor(((p.y - ctx.nwPoint.y) / ctx.heightPx) * SIZE);


                if (px >= 0 && px < SIZE && py >= 0 && py < SIZE) {
                    const tags = el.tags || {};
                    // Categorize
                    if (tags.railway === 'station') pois.metro.push({ id: el.id, x: px, y: py });
                    if (tags.amenity === 'school' || tags.amenity === 'university') pois.school.push({ id: el.id, x: px, y: py });
                    if (tags.leisure === 'park' || tags.leisure === 'garden') pois.park.push({ id: el.id, x: px, y: py });
                    if (tags.amenity === 'hospital' || tags.amenity === 'clinic') pois.hospital.push({ id: el.id, x: px, y: py });
                    if (tags.natural === 'water' || tags.waterway) pois.water.push({ id: el.id, x: px, y: py });
                    
                    // Obstacles (Buildings)
                    if (tags.building) {
                        hurdles[py * SIZE + px] = 1.0;
                    }
                }
            }

            // Inject Seeds & Obstacles
            const bridge = engine.bridge;
            const chunk = engine.vGrid.chunks[0];
            const nx = 512;
            const ny = 512;
            const nxPhys = nx + 2;

            for (const key of Object.keys(pois)) {
                const cDataX = new Float32Array(nxPhys * nxPhys).fill(-10000);
                const cDataY = new Float32Array(nxPhys * nxPhys).fill(-10000);
                const cDataObs = new Float32Array(nxPhys * nxPhys);
                
                for (const p of pois[key]) {
                    const localIdx = (p.y + 1) * nxPhys + (p.x + 1);
                    cDataX[localIdx] = p.x;
                    cDataY[localIdx] = p.y;
                }
                
                // Obstacles (Buildings) shared across all SDF passes
                for (let y = 0; y < ny; y++) {
                    const row = (y + 1) * nxPhys;
                    const srcRow = y * SIZE;
                    for (let x = 0; x < nx; x++) {
                        cDataObs[row + (x + 1)] = hurdles[srcRow + x];
                    }
                }

                bridge.setFaceData(chunk.id, `sdf_${key}_x`, cDataX);
                bridge.setFaceData(chunk.id, `sdf_${key}_y`, cDataY);
                bridge.setFaceData(chunk.id, 'obstacles', cDataObs);
            }


            const start = performance.now();
            if (useGPU) await bridge.syncToDevice();

            for (let i = 0; i < 512; i++) {
                await engine.step(1);
                if (!useGPU) engine.synchronizer.syncAll(engine.vGrid, engine.bridge, engine.parityManager, 'read');
            }

            // If GPU, sync results back to Host (CPU) so CanvasAdapterNeo can render them
            if (useGPU) {
                await bridge.syncToHost();
            }

            const end = performance.now();

            document.getElementById('stat-compute')!.textContent = `${useGPU ? 'GPU' : 'CPU'} Bake Time: ${(end - start).toFixed(1)}ms`;
            document.getElementById('stat-res')!.textContent = `Grid: ${SIZE}x${SIZE} | ${ctx.metersPerPx.toFixed(2)}m/px`;
            
            const areaKm = (ctx.widthMeters / 1000).toFixed(1);
            const legendNote = document.getElementById('legend-area-note');
            if (legendNote) legendNote.textContent = `Area covered: ${areaKm} x ${areaKm} km`;

            
        } catch (e) {
            console.error(e);
            btnBake.textContent = "API Error (Retrying...)";
        }
        
        btnBake.disabled = false;
        btnBake.textContent = "Recalculate Bounds";
    }


    // Initialize rendering
    const sliders = ['metro', 'school', 'park', 'hospital', 'water'].map(key => ({
        key,
        xFace: engine.getFaceLogicalIndex(`sdf_${key}_x`),
        yFace: engine.getFaceLogicalIndex(`sdf_${key}_y`),
        ui: document.getElementById(`slider-${key}-weight`) as HTMLInputElement,
        uiVal: document.getElementById(`val-${key}-weight`)
    }));

    function frame() {
        const ctx = getMapContext();
        const renderOptions: any = {
            colormap: 'accessibility-sdf',
            pixelScale: ctx.metersPerPx,
            criteriaSDF: sliders.map(s => ({
                xFace: s.xFace,
                yFace: s.yFace,
                weight: parseInt(s.ui.value),
                distanceThreshold: 1500 // Return to constant 1.5km threshold for simplicity
            }))
        };
        CanvasAdapterNeo.render(engine, canvas, renderOptions);
        requestAnimationFrame(frame);
    }



    // Handlers
    map.on('moveend zoomend', () => {
        syncCanvas();
        bakeSDF();
    });
    map.on('move zoom', syncCanvas);

    sliders.forEach(s => {
        s.ui.oninput = () => { if (s.uiVal) s.uiVal.textContent = `${s.ui.value}%`; };
    });



    btnBake.onclick = () => bakeSDF();

    syncCanvas();
    await bakeSDF();
    frame();
}

boot().catch(console.error);
