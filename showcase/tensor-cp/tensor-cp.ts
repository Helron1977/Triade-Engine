import { HypercubeNeoFactory } from '../../core/HypercubeNeoFactory';
import { CanvasAdapterNeo } from '../../io/CanvasAdapterNeo';

declare const Chart: any;

const MOVIELENS_LABELS = {
    users: Array.from({length: 100}, (_, i) => `User ${100 + i}`),
    movies: ["Inception", "Titanic", "The Matrix", "Pulp Fiction", "Avatar", "Interstellar", "Gladiator", "The Godfather", "Star Wars", "The Dark Knight"],
    genres: ["Action", "Drama", "Sci-Fi", "Comedy", "Horror", "Thriller", "Romance", "Docs"]
};

let engine: any = null;
let chart: any = null;
let isRunning = false;
let currentIteration = 0;
const errorHistory: number[] = [];

async function initUI() {
    const btnRun = document.getElementById('btn-run') as HTMLButtonElement;
    const btnStop = document.getElementById('btn-stop') as HTMLButtonElement;
    const btnExport = document.getElementById('btn-export') as HTMLButtonElement;
    const btnLoadSample = document.getElementById('btn-load-sample') as HTMLButtonElement;
    const btnLoadPower = document.getElementById('btn-load-power') as HTMLButtonElement;
    const btnLoadSynthetic = document.getElementById('btn-load-synthetic') as HTMLButtonElement;
    const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
    const csvUpload = document.getElementById('csv-upload') as HTMLInputElement;
    const sliceSlider = document.getElementById('slice-slider') as HTMLInputElement;
    const sliceLabel = document.getElementById('slice-label');
    const sliceVal = document.getElementById('slice-val');

    btnRun.onclick = () => startDecomposition();
    btnStop.onclick = () => stopDecomposition();
    btnExport.onclick = () => exportFactors();
    btnLoadSample.onclick = () => loadSample('tensor-sample.csv');
    btnLoadPower.onclick = () => loadSample('power-tensor.csv');
    btnLoadSynthetic.onclick = () => handleSynthetic();
    btnReset.onclick = () => resetFactors();
    
    csvUpload.onchange = async (e: any) => {
        const file = e.target.files[0];
        if (file) {
            const text = await file.text();
            handleCSV(text);
        }
    };

    sliceSlider.oninput = () => {
        const val = parseInt(sliceSlider.value);
        if (sliceLabel) sliceLabel.innerText = `Z-Slice: ${val}`;
        if (sliceVal) sliceVal.innerText = val.toString();
        renderFrame();
    };

    initChart();
}

function initChart() {
    const ctx = (document.getElementById('error-chart') as HTMLCanvasElement).getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Reconstruction Error (MSE)',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: false, grid: { color: '#2d2d35' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#2d2d35' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

async function loadSample(filename: string = 'tensor-sample.csv') {
    try {
        const response = await fetch(`../assets/${filename}`);
        const text = await response.text();
        handleCSV(text);
        console.log(`Loaded ${filename} successfully.`);
    } catch (e) {
        console.error(`Failed to load ${filename}`, e);
    }
}

async function handleSynthetic() {
    if (!engine) await setupEngine();
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const target = views[3];
    const { nx, ny, nz } = (engine.vGrid as any).config.dimensions;

    console.log("Generating Synthetic Rank-3 Tensor...");
    
    // 1. Create 3 small ground-truth factors for Rank 3
    const groundRank = 3;
    const trueA = new Float32Array(nx * groundRank);
    const trueB = new Float32Array(ny * groundRank);
    const trueC = new Float32Array(nz * groundRank);

    // Fill with some patterns (Blocks/Waves)
    for (let r = 0; r < groundRank; r++) {
        for (let i = 0; i < nx; i++) trueA[i * groundRank + r] = Math.sin((i + r * 5) * 0.5) * 0.5 + 0.5;
        for (let j = 0; j < ny; j++) trueB[j * groundRank + r] = Math.cos((j + r * 3) * 0.4) * 0.5 + 0.5;
        for (let k = 0; k < nz; k++) trueC[k * groundRank + r] = (k === r || k === r + 4) ? 1.0 : 0.1;
    }

    // 2. Compute the Kruskal product to create the target tensor (Dense)
    target.fill(0);
    let entries = 0;
    for (let k = 0; k < nz; k++) {
        for (let j = 0; j < ny; j++) {
            for (let i = 0; i < nx; i++) {
                const idx = i + j * nx + k * nx * ny;
                let val = 0;
                for (let r = 0; r < groundRank; r++) {
                    val += trueA[i * groundRank + r] * trueB[j * groundRank + r] * trueC[k * groundRank + r];
                }
                target[idx] = val;
                if (val > 0) entries++;
            }
        }
    }

    // 3. Reset factors to random to start clean
    for (const idx of [0, 1, 2]) {
        views[idx].forEach((v: number, i: number, arr: Float32Array) => arr[i] = Math.random() * 0.5);
    }

    if (engine.vGrid.config.mode === 'gpu') {
        await bridge.syncToDevice();
    }

    currentIteration = 0;
    lastMSE = 0;
    renderFrame();
    updateHUD(0, calculateMSE());
}

async function handleCSV(text: string) {
    if (!engine) {
        await setupEngine();
    }
    
    const lines = text.trim().split('\n');
    const bridge = (engine as any).bridge;
    const chunkId = (engine as any).vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const target = views[3];
    const { nx, ny, nz } = (engine.vGrid as any).config.dimensions;

    target.fill(0);
    let count = 0;
    for (let i = 1; i < lines.length; i++) {
        const [u, m, g, val] = lines[i].split(',').map(Number);
        if (!isNaN(u) && u < nx && m < ny && g < nz) {
            const idx = u + m * nx + g * nx * ny;
            target[idx] = val;
            count++;
        }
    }

    const sparsity = (count / (nx * ny * nz) * 100).toFixed(2);
    document.getElementById('stats-summary')!.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <div><strong>Entries:</strong> ${count}</div>
            <div><strong>Dimensions:</strong> ${nx} x ${ny} x ${nz}</div>
            <div><strong>Theoretical Max:</strong> ${nx * ny * nz}</div>
            <div><strong>Sparsity:</strong> ${100 - parseFloat(sparsity)}%</div>
            <div><strong>Density:</strong> ${sparsity}%</div>
        </div>
    `;

    // Initialize Factors
    const rank = engine.vGrid.config.params.rank || 10;
    for (let f = 0; f < 3; f++) {
        const view = views[f];
        for (let i = 0; i < view.length; i++) {
            view[i] = 0.05 + Math.random() * 0.2;
        }
    }

    if (engine.vGrid.config.mode === 'gpu') {
        await bridge.syncToDevice();
    }

    renderFrame();
    updateFactorTables();
    console.log(`Loaded ${count} entries into tensor.`);
}

async function setupEngine() {
    const factory = new HypercubeNeoFactory();
    const urlParams = new URLSearchParams(window.location.search);
    const backendSelect = document.getElementById('backend-select') as HTMLSelectElement;
    const backend = urlParams.get('backend') || backendSelect.value;
    backendSelect.value = backend; 
    const rankEl = document.getElementById('param-rank') as HTMLInputElement;
    const rank = rankEl ? parseInt(urlParams.get('rank') || rankEl.value) : 10;
    if (rankEl) rankEl.value = rank.toString();

    const regEl = document.getElementById('param-reg') as HTMLInputElement;
    const reg = regEl ? parseFloat(regEl.value) : 0.05;

    const lrEl = document.getElementById('param-lr') as HTMLInputElement;
    const lr = lrEl ? parseFloat(lrEl.value) : 0.01;

    const manifest = await factory.fromManifest('./showcase-tensor-cp.json');
    manifest.config.mode = backend as any;
    manifest.config.params.rank = rank;
    manifest.config.params.regularization = reg;

    // Critical: Inject into the rule itself so dispatchers see it
    if (manifest.engine.rules[0]) {
        manifest.engine.rules[0].params = {
            rank,
            regularization: reg,
            learningRate: lr
        };
    }
    
    engine = await factory.build(manifest.config, manifest.engine);

    // Sync Details to UI
    const memMb = engine.bridge.rawBuffer ? (engine.bridge.rawBuffer.byteLength / 1024 / 1024).toFixed(1) : "0.0";
    document.getElementById('manifest-details')!.innerHTML = `
        Rank: ${rank}<br>
        Backend: ${backend.toUpperCase()}<br>
        Memory: ~${memMb} MB
    `;
    
    const canvases = ['canvas-original', 'canvas-recon', 'canvas-error'];
    const { nx, ny } = manifest.config.dimensions;
    canvases.forEach(id => {
        const c = document.getElementById(id) as HTMLCanvasElement;
        c.width = nx;
        c.height = ny;
        c.style.width = '600px';
        c.style.height = '600px';
    });

    const sliceSlider = document.getElementById('slice-slider') as HTMLInputElement;
    sliceSlider.max = (manifest.config.dimensions.nz - 1).toString();
}

async function resetFactors() {
    console.log("Attempting reset...");
    if (!engine) {
        console.log("Engine not ready, setting up...");
        await setupEngine();
    }
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    
    for (let f = 0; f < 3; f++) {
        const view = views[f];
        for (let i = 0; i < view.length; i++) {
            view[i] = 0.05 + Math.random() * 0.2;
        }
    }
    
    // Clear reconstruction to reset visuals
    if (views[4]) views[4].fill(0);
    
    if (engine.vGrid.config.mode === 'gpu') {
        await bridge.syncToDevice();
    }
    
    // UI Reset
    currentIteration = 0;
    errorHistory.length = 0;
    if (chart) {
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.update();
    }
    document.getElementById('decomp-status')!.innerText = 'IDLE';
    document.getElementById('decomp-mse')!.innerText = 'N/A';
    
    renderFrame();
    updateFactorTables();
    console.log("Factors reset completed.");
}

async function startDecomposition() {
    console.log("Starting decomposition...");
    if (!engine) await setupEngine();
    
    // Safety check: Is there any data?
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const target = views[3];
    let hasData = false;
    for (let i = 0; i < Math.min(target.length, 1000); i++) {
        if (target[i] > 0) { hasData = true; break; }
    }
    
    if (!hasData) {
        console.warn("No data in tensor.");
        alert("No data loaded in tensor. Please upload a CSV or load the sample first.");
        return;
    }

    isRunning = true;
    currentIteration = 0;
    errorHistory.length = 0;
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();
    document.getElementById('decomp-status')!.innerText = 'RUNNING';
    document.getElementById('decomp-status')!.style.color = '#6366f1';
    console.log("Entering simulation loop...");
    lastMSE = Infinity;
    loop();
}

function stopDecomposition() {
    isRunning = false;
    document.getElementById('decomp-status')!.innerText = 'STOPPED';
    document.getElementById('decomp-status')!.style.color = '#ef4444';
}

let lastMSE = Infinity;

async function loop() {
    if (!isRunning || !engine) return;

    const maxIter = parseInt((document.getElementById('param-iter') as HTMLInputElement).value) || 500;
    const threshold = 0.0001;
    const start = performance.now();
    
    try {
        await engine.step(currentIteration);
        const ms = performance.now() - start;

        if (engine.vGrid.config.mode === 'gpu') {
            await engine.bridge.syncToHost();
        }

        const error = calculateMSE();
        
        // Early Stopping
        const delta = Math.abs(lastMSE - error);
        if (delta < threshold && currentIteration > 10) {
            console.log(`Early stopping triggered: delta=${delta.toFixed(6)} < ${threshold}`);
            isRunning = false;
            document.getElementById('decomp-status')!.innerHTML = 'Status: <span style="color: #4ade80">CONVERGED</span>';
            updateHUD(ms, error);
            updateFactorTables();
            renderFrame();
            return;
        }
        lastMSE = error;

        updateHUD(ms, error);
        updateChart(error);
        if (currentIteration % 2 === 0) updateFactorTables();
        renderFrame();

        currentIteration++;
        if (currentIteration < maxIter && isRunning) {
            requestAnimationFrame(loop);
        } else {
            console.log("Decomposition finished (max iterations).");
            isRunning = false;
            document.getElementById('decomp-status')!.innerText = 'COMPLETED';
            document.getElementById('decomp-status')!.style.color = '#22c55e';
            document.getElementById('decomp-mse')!.innerText = error.toFixed(6);
            document.getElementById('decomp-delta')!.innerText = delta.toFixed(6);
            updateHUD(ms, error);
            updateFactorTables();
            renderFrame();
        }
    } catch (e) {
        console.error("Error in decomposition loop:", e);
        isRunning = false;
    }
}

function calculateMSE(): number {
    if (!engine) return 0;
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const target = views[3];
    const recon = views[4];
    
    let sumErr = 0;
    let count = 0;
    for (let i = 0; i < target.length; i++) {
        if (target[i] > 0) {
            const diff = target[i] - recon[i];
            sumErr += diff * diff;
            count++;
        }
    }
    return count > 0 ? sumErr / count : 0;
}

function updateHUD(ms: number, error: number) {
    const maxIter = (document.getElementById('param-iter') as HTMLInputElement).value;
    const delta = Math.abs(lastMSE - error);
    document.getElementById('hud-compute')!.innerText = `${ms.toFixed(2)} ms`;
    document.getElementById('hud-iter')!.innerText = `${currentIteration}/${maxIter}`;
    document.getElementById('hud-error')!.innerText = `${error.toFixed(4)} / ${delta.toFixed(6)}`;
    document.getElementById('decomp-mse')!.innerText = error.toFixed(6);
    document.getElementById('decomp-delta')!.innerText = delta.toFixed(6);
}

function updateChart(error: number) {
    chart.data.labels.push(currentIteration);
    chart.data.datasets[0].data.push(error);
    if (chart.data.labels.length > 50) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    chart.update('none');
}

function computeDataRange(): { min: number; max: number } {
    if (!engine) return { min: 0, max: 1 };
    const views = engine.bridge.getChunkViews(engine.vGrid.chunks[0].id);
    const target = views[3];
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < target.length; i++) {
        const v = target[i];
        if (v > 0) {
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
    }
    if (!isFinite(minVal)) return { min: 0, max: 1 };
    const range = maxVal - minVal || 1;
    return { min: Math.max(0, minVal - range * 0.05), max: maxVal + range * 0.1 };
}

function computeReconRange(): { min: number; max: number } {
    if (!engine) return { min: 0, max: 1 };
    const views = engine.bridge.getChunkViews(engine.vGrid.chunks[0].id);
    const recon = views[4];
    const target = views[3];
    let minVal = Infinity, maxVal = -Infinity;
    
    // Scan only cells where recon has meaningful predictions (from observed entries)
    for (let i = 0; i < recon.length; i++) {
        const v = recon[i];
        if (!isFinite(v)) continue;
        if (target[i] > 0) {
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
    }
    
    if (!isFinite(minVal)) {
        // Fall back to full recon range if nothing observed yet or data is zero
        for (let i = 0; i < recon.length; i++) {
            const v = recon[i];
            if (isFinite(v) && Math.abs(v) > 1e-10) {
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
        }
    }
    
    if (!isFinite(minVal)) return { min: 0, max: 1 };
    const range = maxVal - minVal || 1;
    return { min: Math.max(0, minVal - range * 0.05), max: maxVal + range * 0.1 };
}

/** Inferno palette (5 stops): black → purple → red → orange → yellow/white */
const INFERNO_PALETTE: [number, number, number][] = [
    [0, 0, 4], [120, 28, 109], [238, 75, 43], [253, 173, 54], [252, 255, 164]
];

function samplePalette(t: number, palette: [number, number, number][]): [number, number, number] {
    t = Math.max(0, Math.min(1, t));
    const n = palette.length - 1;
    const idx = t * n;
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, n);
    const f = idx - lo;
    const [r1, g1, b1] = palette[lo];
    const [r2, g2, b2] = palette[hi];
    return [r1 + (r2 - r1) * f, g1 + (g2 - g1) * f, b1 + (b2 - b1) * f];
}

/**
 * Draws the error heatmap for ALL cells on the slice.
 * - Observed cells (target > 0): show |target - recon| (fit quality)
 * - Unobserved cells: show |recon| (reconstruction "hallucination")
 * Scale is auto-set to 95th-percentile of errors to avoid saturation.
 */
function renderErrorMap(canvas: HTMLCanvasElement, sliceZ: number) {
    const views = engine.bridge.getChunkViews(engine.vGrid.chunks[0].id);
    const target = views[3];
    const recon  = views[4];
    const dims = engine.vGrid.dimensions; // Authoritative dimensions
    const nx = dims.nx;
    const ny = dims.ny;

    if (canvas.width !== nx || canvas.height !== ny) {
        canvas.width  = nx;
        canvas.height = ny;
        canvas.style.width  = '600px';
        canvas.style.height = '600px';
    }

    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(nx, ny);
    // Use Uint32Array for pixel manipulation to be consistent with CanvasAdapterNeo
    const pixels32 = new Uint32Array(imageData.data.buffer);
    const sliceOffset = sliceZ * nx * ny;

    // Collect all errors for this slice to compute 95th-percentile scale
    const errors: number[] = [];
    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
            const idx = sliceOffset + j * nx + i;
            const tgt = target[idx];
            const rec = recon[idx];
            // Handle potentially non-finite values from ALS instability
            const valT = isFinite(tgt) ? tgt : 0;
            const valR = isFinite(rec) ? rec : 0;
            const err = valT > 0 ? Math.abs(valT - valR) : Math.abs(valR);
            errors.push(err);
        }
    }
    errors.sort((a, b) => a - b);
    const p95idx = Math.floor(errors.length * 0.95);
    // Use a noise floor (0.1) so we don't amplify tiny residuals into bright colors.
    // This ensures a "near-perfect" fit actually looks black/dark as expected.
    const p95Val = errors[p95idx];
    const maxErr = Math.max(0.1, p95Val > 1e-10 ? p95Val : (errors[errors.length - 1] > 1e-10 ? errors[errors.length - 1] : 1.0));

    // Draw pixels
    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
            const srcIdx  = sliceOffset + j * nx + i;
            const dstIdx  = j * nx + i;
            const tgt = target[srcIdx];
            const rec = recon[srcIdx];
            const valT = isFinite(tgt) ? tgt : 0;
            const valR = isFinite(rec) ? rec : 0;
            const err = valT > 0 ? Math.abs(valT - valR) : Math.abs(valR);
            
            const t   = Math.min(1, err / maxErr);
            const [r, g, b] = samplePalette(t, INFERNO_PALETTE);
            
            // RGBA as little-endian 0xAABBGGRR
            pixels32[dstIdx] = 
                (255 << 24) |              // Alpha
                (Math.floor(b) << 16) |    // Blue
                (Math.floor(g) << 8) |     // Green
                Math.floor(r);             // Red
        }
    }
    ctx.putImageData(imageData, 0, 0);
}

function renderFrame() {
    if (!engine) return;
    const sliceZ = parseInt((document.getElementById('slice-slider') as HTMLInputElement).value);

    const { min: dataMin, max: dataMax } = computeDataRange();
    const { min: reconMin, max: reconMax } = computeReconRange();

    // 1. Original tensor slice — use target data range
    CanvasAdapterNeo.render(engine as any, document.getElementById('canvas-original') as HTMLCanvasElement, {
        faceIndex: 3,
        colormap: 'magma',
        minVal: dataMin,
        maxVal: dataMax,
        sliceZ: sliceZ
    });

    // 2. Reconstruction — use its own range so it's always visible
    CanvasAdapterNeo.render(engine as any, document.getElementById('canvas-recon') as HTMLCanvasElement, {
        faceIndex: 4,
        colormap: 'magma',
        minVal: reconMin,
        maxVal: reconMax,
        sliceZ: sliceZ
    });

    // 3. Error heatmap — custom draw, only at observed (target > 0) positions
    renderErrorMap(document.getElementById('canvas-error') as HTMLCanvasElement, sliceZ);

    updateSliceMetrics(sliceZ);
}

function updateSliceMetrics(sliceZ: number) {
    if (!engine) return;
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const target = views[3];
    const recon = views[4];
    const { nx, ny } = (engine.vGrid as any).config.dimensions;
    const sliceSize = nx * ny;
    const offset = sliceZ * sliceSize;

    let sliceSumSqErr = 0;
    let sliceSumAbsErr = 0;
    let sliceActiveCount = 0;

    for (let i = 0; i < sliceSize; i++) {
        const idx = offset + i;
        if (target[idx] > 0) {
            const diff = target[idx] - recon[idx];
            sliceSumSqErr += diff * diff;
            sliceSumAbsErr += Math.abs(diff);
            sliceActiveCount++;
        }
    }

    const sliceMSE = sliceActiveCount > 0 ? sliceSumSqErr / sliceActiveCount : 0;
    const sliceMAE = sliceActiveCount > 0 ? sliceSumAbsErr / sliceActiveCount : 0;
    const sliceDensity = (sliceActiveCount / sliceSize) * 100;

    document.getElementById('mse-original')!.innerText = `Active Density: ${sliceDensity.toFixed(2)}%`;
    document.getElementById('mse-recon')!.innerText = `Slice MSE: ${sliceMSE.toFixed(6)}`;
    document.getElementById('mse-error')!.innerText = `Mean Absolute Error: ${sliceMAE.toFixed(6)}`;
}

function updateFactorTables() {
    if (!engine) return;
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    const rank = engine.vGrid.config.params.rank || 10;

    const renderTable = (view: Float32Array, id: string, name: string, labels: string[]) => {
        let html = `<table><thead><tr><th>${name}</th>`;
        for (let r = 0; r < rank; r++) html += `<th>R${r}</th>`;
        html += `</tr></thead><tbody>`;
        
        const maxRows = 15;
        const nRows = view.length / rank;
        for (let i = 0; i < Math.min(nRows, maxRows); i++) {
            const label = labels[i] || `${name} #${i}`;
            html += `<tr><td style="font-weight: 600; color: #fff;">${label}</td>`;
            for (let r = 0; r < rank; r++) {
                const val = view[i * rank + r];
                const cls = val > 0.01 ? 'val-pos' : (val < -0.01 ? 'val-neg' : '');
                html += `<td class="${cls}">${val.toFixed(3)}</td>`;
            }
            html += `</tr>`;
        }
        html += `</tbody></table>`;
        document.getElementById(id)!.innerHTML = html;
    };

    renderTable(views[0], 'table-a', 'User', MOVIELENS_LABELS.users);
    renderTable(views[1], 'table-b', 'Movie', MOVIELENS_LABELS.movies);
    renderTable(views[2], 'table-c', 'Genre', MOVIELENS_LABELS.genres);
}

function exportFactors() {
    if (!engine) return;
    const bridge = engine.bridge;
    const chunkId = engine.vGrid.chunks[0].id;
    const views = bridge.getChunkViews(chunkId);
    
    const data = {
        mode_a: Array.from(views[0]),
        mode_b: Array.from(views[1]),
        mode_c: Array.from(views[2])
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tensor_factors.json';
    a.click();
}

initUI();
loadSample();
