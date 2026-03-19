import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { HypercubeNeoFactory } from '../../../core/HypercubeNeoFactory';

// --- Constants & Config ---
const NX = 64; const NY = 32; const NZ = 64;
const TANK_SIZE = 20;

enum SharkState {
    PATROL = "SEARCHING VOLUME",
    HUNT = "VOLUMETRIC CHASE",
    AMBUSH = "DEEP BRAIN AMBUSH",
    WANDER = "EXPLORING NEBULA",
    RECOVERY = "RECOVERING"
}

class LifeNebula {
    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private renderer!: THREE.WebGLRenderer;
    private controls!: OrbitControls;
    
    private engine: any = null;
    private shark!: THREE.Group;
    private preyCount = 60;
    private preyList: THREE.Group[] = [];
    private preyVels: THREE.Vector3[] = [];
    private preyStatus: number[] = [];

    private sharkVel = new THREE.Vector3(0, 0, 0);
    private sharkState: SharkState = SharkState.PATROL;
    private stateTimer = 0;
    private stateDuration = 0;
    private targetIndex = -1; // -1 = No target
    private lastPreyPos = new THREE.Vector3(0,0,0);

    private waterMesh!: THREE.Mesh;
    private waterGeo!: THREE.PlaneGeometry;
    private heatmapCtx!: CanvasRenderingContext2D;
    private stateHUD!: HTMLDivElement;
    private ambushTarget = new THREE.Vector3(0, 0, 0);
    private stuckTime = 0;

    constructor() {
        this.init3D();
        this.initEngine();
    }

    private init3D() {
        const container = document.getElementById('canvas-container')!;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x020617);
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(22, 12, 22);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        const gui = document.createElement('div');
        gui.style.cssText = 'position: absolute; top: 20px; left: 20px; color: #38bdf8; font-family: "Courier New", monospace; pointer-events: none; text-shadow: 0 0 10px #38bdf8;';
        gui.innerHTML = '<h1>NEBULA BRAIN 3D v4.0</h1><div id="shark-state" style="font-size: 20px;">INITIALIZING VOLUME...</div>';
        container.appendChild(gui);
        this.stateHUD = document.getElementById('shark-state') as HTMLDivElement;

        const canvas = document.createElement('canvas');
        canvas.width = NX; canvas.height = NZ;
        canvas.style.cssText = 'position: absolute; bottom: 20px; right: 20px; width: 140px; height: 140px; border: 2px solid #38bdf8; border-radius: 4px; opacity: 0.9;';
        container.appendChild(canvas);
        this.heatmapCtx = canvas.getContext('2d')!;

        this.scene.add(new THREE.HemisphereLight(0x38bdf8, 0x020617, 2));
        const p1 = new THREE.PointLight(0xf43f5e, 500, 50); p1.position.set(12, 10, 12); this.scene.add(p1);
        const p2 = new THREE.PointLight(0x0ea5e9, 500, 50); p2.position.set(-12, 5, -12); this.scene.add(p2);

        const sun = new THREE.DirectionalLight(0xffffff, 3.5);
        sun.position.set(10, 25, 15);
        this.scene.add(sun);

        const amb = new THREE.AmbientLight(0x0ea5e9, 0.5);
        this.scene.add(amb);

        const tankMat = new THREE.MeshPhysicalMaterial({ color: 0x38bdf8, transmission: 0.95, thickness: 1, transparent: true, opacity: 0.2, side: THREE.DoubleSide });
        this.scene.add(new THREE.Mesh(new THREE.BoxGeometry(TANK_SIZE, TANK_SIZE * 0.5, TANK_SIZE), tankMat));

        this.setupModels();
    }

    private setupModels() {
        this.shark = new THREE.Group();
        const sMat = new THREE.MeshPhysicalMaterial({ color: 0x475569, metalness: 0.8, roughness: 0.2, clearcoat: 1 });
        const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.5, 1.4, 4, 16), sMat);
        body.rotation.x = Math.PI / 2; body.scale.set(1, 1, 0.7);
        this.shark.add(body);
        this.scene.add(this.shark);

        const colors = [0xf43f5e, 0x38bdf8, 0xd9f99d, 0x818cf8];
        for (let i = 0; i < this.preyCount; i++) {
            const mat = new THREE.MeshStandardMaterial({ color: colors[i%4], emissive: colors[i%4], emissiveIntensity: 1.5 });
            const p = new THREE.Mesh(new THREE.DodecahedronGeometry(0.2), mat);
            p.scale.set(1, 0.4, 1.6);
            this.scene.add(p);
            this.preyList.push(p as any);
            p.position.set((Math.random()-0.5)*TANK_SIZE*0.8, (Math.random()-0.6)*TANK_SIZE*0.3, (Math.random()-0.5)*TANK_SIZE*0.8);
            this.preyVels.push(new THREE.Vector3((Math.random()-0.5)*0.1, (Math.random()-0.5)*0.05, (Math.random()-0.5)*0.1));
            this.preyStatus.push(0);
        }

        this.waterGeo = new THREE.PlaneGeometry(TANK_SIZE, TANK_SIZE, NX - 1, NZ - 1);
        const wMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, 
            transparent: true, 
            opacity: 0.7, 
            transmission: 0.3, 
            side: THREE.DoubleSide,
            metalness: 0.7,
            roughness: 0.1,
            clearcoat: 1.0,
            clearcoatRoughness: 0.1
        });
        this.waterMesh = new THREE.Mesh(this.waterGeo, wMat);
        this.waterMesh.rotation.x = -Math.PI / 2; this.waterMesh.position.y = TANK_SIZE * 0.25;
        this.scene.add(this.waterMesh);
    }

    private async initEngine() {
        const factory = new HypercubeNeoFactory();
        const manifest = await factory.fromManifest('./nebula-manifest.json');
        console.info("Nebula: Building Engine...");
        this.engine = await factory.build(manifest.config, manifest.engine);
        console.info("Nebula: Engine Built. Starting Animation Loop.");
        
        const loader = document.getElementById('loader');
        if (loader) loader.style.display = 'none';

        this.lastTime = performance.now();
        this.animate();
    }

    private lastTime = 0;

    private worldToGrid(v: THREE.Vector3) { 
        return {
            x: Math.floor(((v.x / TANK_SIZE) + 0.5) * (NX - 1)),
            y: Math.floor(((v.y / (TANK_SIZE * 0.5)) + 0.5) * (NY - 1)),
            z: Math.floor(((v.z / TANK_SIZE) + 0.5) * (NZ - 1))
        };
    }
    private gridToWorld(x: number, y: number, z: number) {
        return new THREE.Vector3(
            ((x / (NX - 1)) - 0.5) * TANK_SIZE,
            ((y / (NY - 1)) - 0.5) * TANK_SIZE * 0.5,
            ((z / (NZ - 1)) - 0.5) * TANK_SIZE
        );
    }

    private async updateAI() {
        if (!this.engine) return;
        const bridge = this.engine.bridge;
        const chunk = this.engine.vGrid.chunks[0];
        const nX = NX; const nY = NY; const nZ = NZ;
        const getIdx = (gx: number, gy: number, gz: number) => (gz * nY + gy) * nX + gx;

        // 1. 3D Heatmap (Pattern Memory)
        const heat = bridge.getFaceData(chunk.id, 'strategy_heatmap');
        
        // Decay patterns to keep them relevant
        for(let i=0; i<heat.length; i++) heat[i] *= 0.995;

        // Overlay current prey positions for 'Vision'
        const avgP = new THREE.Vector3(0,0,0);
        let activePrey = 0;
        this.preyList.forEach((p, i) => {
            if (this.preyStatus[i] > 0) return;
            avgP.add(p.position); activePrey++;
        });
        if (activePrey > 0) avgP.divideScalar(activePrey);

        // HUD Visualization (Projected onto XZ)
        const img = this.heatmapCtx.createImageData(NX, NZ);
        let maxV = 0, mx=nX/2, my=NY/2, mz=NZ/2;
        for(let z=0; z<NZ; z++) {
            for(let x=0; x<NX; x++) {
                let colH = 0; for(let y=0; y<NY; y++) {
                    const v = heat[getIdx(x, y, z)]; 
                    colH += v;
                    if(v > maxV) { maxV = v; mx = x; my = y; mz = z; }
                }
                const i = (z*NX+x)*4;
                img.data[i] = colH*10; img.data[i+1] = colH*5; img.data[i+2] = colH*40; img.data[i+3] = 255;
            }
        }
        this.heatmapCtx.putImageData(img, 0, 0);
        this.ambushTarget.copy(this.gridToWorld(mx, my, mz));

        // 2. Intelligence: Target Selection & Opportunism
        if (this.targetIndex === -1 || this.preyStatus[this.targetIndex] > 0) {
            // Pick new target: closest or best-pattern-matching
            let bestScore = -1;
            this.preyList.forEach((p, i) => {
                if (this.preyStatus[i] > 0) return;
                const d = p.position.distanceTo(this.shark.position);
                const g = this.worldToGrid(p.position);
                const pValue = heat[getIdx(g.x, g.y, g.z)] || 0;
                const score = (100 - d) + pValue * 50; // Distance + Pattern Value
                if (score > bestScore) { bestScore = score; this.targetIndex = i; }
            });
            if (this.targetIndex !== -1) {
                this.sharkState = SharkState.HUNT;
                this.stateDuration = 500;
            }
        } else {
            // Opportunistic Detection: Is there a BETTER target nearby in a High Success Zone?
            this.preyList.forEach((p, i) => {
                if (i === this.targetIndex || this.preyStatus[i] > 0) return;
                const d = p.position.distanceTo(this.shark.position);
                if (d < 4) { // Only switch if very close
                    const g = this.worldToGrid(p.position);
                    if (heat[getIdx(g.x, g.y, g.z)] > 2.0) { // High success pattern
                        this.targetIndex = i;
                        console.debug("Nebula AI: Seizing Opportunity - Pattern Matched!");
                    }
                }
            });
        }

        // --- States & Decisions ---
        if (this.targetIndex === -1) {
            this.sharkState = SharkState.PATROL;
        }

        // Anti-Stuck Logic
        if (this.sharkVel.lengthSq() < 0.0001) this.stuckTime++;
        else this.stuckTime = 0;
        if (this.stuckTime > 80) {
            this.sharkState = SharkState.WANDER; this.stateDuration = 100; this.stuckTime = 0;
            this.targetIndex = -1;
            this.sharkVel.add(new THREE.Vector3(Math.random()-0.5, Math.random()-0.5, Math.random()-0.5).multiplyScalar(2));
        }

        if (this.stateHUD) this.stateHUD.innerText = this.sharkState;

        // 3. Volumetric Pathfinder Seeds & Surface Injection
        let target = avgP;
        if (this.targetIndex !== -1) target = this.preyList[this.targetIndex].position;
        else if (this.sharkState === SharkState.AMBUSH) target = this.ambushTarget;
        else if (this.sharkState === SharkState.WANDER) target = new THREE.Vector3((Math.random()-0.5)*TANK_SIZE, (Math.random()-0.6)*TANK_SIZE*0.3, (Math.random()-0.5)*TANK_SIZE);
        
        // Shark Wake: Inject into Ocean if near surface
        const vh = TANK_SIZE * 0.24;
        const distFromSurface = Math.abs(this.shark.position.y - vh);
        if (distFromSurface < 2.5 && this.engine) {
            const g = this.worldToGrid(this.shark.position);
            (this.engine.dispatcher as any).setRuleParams('neo-ocean-v1', {
                objects: [
                    { pos: { x: g.x, y: g.z }, dim: { x: 6, y: 6 }, isObstacle: 0, biology: 0, objType: 1, rho: 5.0 }, // Stronger Injection
                    ...new Array(7).fill({ pos: { x: -1000, y: -1000 }, dim: { x: 1, y: 1 }, isObstacle: 0, biology: 0, objType: 0, rho: 0 })
                ]
            });
        }
        
        const sx = new Float32Array(nX*nY*nZ).fill(-10000);
        const sy = new Float32Array(nX*nY*nZ).fill(-10000);
        const sz = new Float32Array(nX*nY*nZ).fill(-10000);
        const gT = this.worldToGrid(target);
        if (gT.x >=0 && gT.x < NX) {
            const tIdx = getIdx(gT.x, gT.y, gT.z);
            sx[tIdx] = gT.x; sy[tIdx] = gT.y; sz[tIdx] = gT.z;
        }
        bridge.setFaceData(chunk.id, 'sdf_predator_x', sx); 
        bridge.setFaceData(chunk.id, 'sdf_predator_y', sy);
        bridge.setFaceData(chunk.id, 'sdf_predator_z', sz);

        // Volumetric Boundaries
        const ax = new Float32Array(nX*nY*nZ).fill(-10000);
        const ay = new Float32Array(nX*nY*nZ).fill(-10000);
        const az = new Float32Array(nX*nY*nZ).fill(-10000);
        const addWall = (x:number, y:number, z:number) => { const i = getIdx(x,y,z); ax[i]=x; ay[i]=y; az[i]=z; };
        for(let z=0; z<NZ; z++) for(let y=0; y<NY; y++) { addWall(0, y, z); addWall(NX-1, y, z); }
        for(let z=0; z<NZ; z++) for(let x=0; x<NX; x++) { addWall(x, 0, z); addWall(x, NY-1, z); }
        for(let y=0; y<NY; y++) for(let x=0; x<NX; x++) { addWall(x, y, 0); addWall(x, y, NZ-1); }
        bridge.setFaceData(chunk.id, 'sdf_avoid_x', ax);
        bridge.setFaceData(chunk.id, 'sdf_avoid_y', ay);
        bridge.setFaceData(chunk.id, 'sdf_avoid_z', az);

        // 4. Step & Sync
        console.debug("Nebula: Engine Step...");
        await this.engine.step(1);
        console.debug("Nebula: Syncing to Host...");
        await bridge.syncToHost(['sdf_predator_x', 'sdf_predator_y', 'sdf_predator_z', 'sdf_avoid_x', 'sdf_avoid_y', 'sdf_avoid_z', 'water_h', 'strategy_heatmap'].map(f => this.engine.getFaceLogicalIndex(f)));
        console.debug("Nebula: Sync Done.");

        // 5. 3D Steering
        const fields = {
            sx: bridge.getFaceData(chunk.id, 'sdf_predator_x'),
            sy: bridge.getFaceData(chunk.id, 'sdf_predator_y'),
            sz: bridge.getFaceData(chunk.id, 'sdf_predator_z'),
            ax: bridge.getFaceData(chunk.id, 'sdf_avoid_x'),
            ay: bridge.getFaceData(chunk.id, 'sdf_avoid_y'),
            az: bridge.getFaceData(chunk.id, 'sdf_avoid_z')
        };

        const steer = (pos: THREE.Vector3, vel: THREE.Vector3, isShark: boolean) => {
            const g = this.worldToGrid(pos);
            const gx = Math.max(0, Math.min(NX-1, g.x));
            const gy = Math.max(0, Math.min(NY-1, g.y));
            const gz = Math.max(0, Math.min(NZ-1, g.z));
            const i = getIdx(gx, gy, gz);
            const force = new THREE.Vector3(0,0,0);

            let distToWall = 1000;
            if (fields.ax[i] > -5000) {
                const r = new THREE.Vector3(gx - fields.ax[i], (gy - fields.ay[i])*1.5, gz - fields.az[i]);
                distToWall = r.length();
                let avoidWeight = Math.pow(Math.max(0, (12 - distToWall)/12), 2) * 2.5;

                if (isShark && this.targetIndex !== -1) {
                    const distToTarget = pos.distanceTo(this.preyList[this.targetIndex].position);
                    if (distToTarget < 3.0) avoidWeight *= 0.1; // "Wall-Scraping" mode
                }
                force.add(r.normalize().multiplyScalar(avoidWeight));
                
                // Tangential slide
                if (distToWall < 4) force.add(new THREE.Vector3(r.z, 0, -r.x).normalize().multiplyScalar(0.2));
            }

            if (isShark) {
                if (this.targetIndex !== -1) {
                    const target = this.preyList[this.targetIndex].position;
                    force.add(target.clone().sub(pos).normalize().multiplyScalar(0.5));
                } else if (fields.sx[i] > -5000) {
                    force.add(new THREE.Vector3(fields.sx[i]-gx, (fields.sy[i]-gy)*0.5, fields.sz[i]-gz).normalize().multiplyScalar(0.4));
                }
            } else {
                if (pos.distanceTo(this.shark.position) < 6) force.add(pos.clone().sub(this.shark.position).normalize().multiplyScalar(0.7));
                
                // Exponential Corner Repulsion (Corner-Fear)
                const margin = 5;
                if (gx < margin) force.x += Math.pow((margin - gx)/margin, 2) * 2.0;
                if (gx > NX - margin) force.x -= Math.pow((gx - (NX-margin))/margin, 2) * 2.0;
                if (gy < margin) force.y += Math.pow((margin - gy)/margin, 2) * 2.0;
                if (gy > NY - margin) force.y -= Math.pow((gy - (NY-margin))/margin, 2) * 2.0;
                if (gz < margin) force.z += Math.pow((margin - gz)/margin, 2) * 2.0;
                if (gz > NZ - margin) force.z -= Math.pow((gz - (NZ-margin))/margin, 2) * 2.0;

                force.y += Math.sin(Date.now()*0.003 + pos.x)*0.08;
            }

            vel.lerp(force, isShark ? 0.2 : 0.3);
            if (isNaN(vel.x)) vel.set(0,0,0);
        };

        steer(this.shark.position, this.sharkVel, true);
        this.preyList.forEach((p, i) => steer(p.position, this.preyVels[i], false));

        // 6. Water Mesh Sync (Top Layer 3D alignment)
        const hData = bridge.getFaceData(chunk.id, 'water_h');
        const pAttr = this.waterGeo.attributes.position;
        const gy = NY - 1; 
        for (let j=0; j<NZ; j++) {
            for (let i=0; i<NX; i++) {
                const vertIdx = j * NX + i;
                const physIdx = (j * NY + gy) * NX + i; // Match (z * NY + y) * NX + x
                const val = hData[physIdx];
                pAttr.setY(vertIdx, isNaN(val) ? 0 : (val - 1.0) * 45.0); 
            }
        }
        pAttr.needsUpdate = true; this.waterGeo.computeVertexNormals();

        // 7. HUD Update
        const now = performance.now();
        const dt = now - this.lastTime;
        this.lastTime = now;
        const fpsElem = document.getElementById('stat-fps');
        if (fpsElem) fpsElem.innerText = `${dt.toFixed(1)}ms`;
        const tensorElem = document.getElementById('stat-tensor');
        if (tensorElem) tensorElem.innerText = (this.stateTimer % 100 < 50) ? "Optimizing..." : "Converged";
    }

    private updating = false;
    private animate = async () => {
        if (!this.updating) {
            this.updating = true;
            await this.updateAI();
            this.updating = false;
        }
        const b = TANK_SIZE * 0.49; const vh = TANK_SIZE * 0.24;
        this.shark.position.add(this.sharkVel);
        if (this.sharkVel.lengthSq() > 0.001) this.shark.lookAt(this.shark.position.clone().add(this.sharkVel));
        this.shark.position.clamp(new THREE.Vector3(-b, -vh, -b), new THREE.Vector3(b, vh, b));

        this.preyList.forEach((p, i) => {
            p.position.add(this.preyVels[i]);
            if (this.preyVels[i].lengthSq() > 0.001) p.lookAt(p.position.clone().add(this.preyVels[i]));
            p.position.clamp(new THREE.Vector3(-b, -vh, -b), new THREE.Vector3(b, vh, b));

            if (p.position.distanceTo(this.shark.position) < 1.4) {
                this.preyStatus[i] = 12;
                const mesh = p as unknown as THREE.Mesh;
                if (mesh.material) {
                    (mesh.material as THREE.MeshStandardMaterial).color.set(0xff0000); 
                    (mesh.material as THREE.MeshStandardMaterial).emissive.set(0xff0000);
                }
                
                // --- LEARNING: Reinforce Success Pattern ---
                if (this.engine) {
                    const heat = this.engine.bridge.getFaceData(this.engine.vGrid.chunks[0].id, 'strategy_heatmap');
                    const g = this.worldToGrid(p.position);
                    const idx = (g.z * NY + g.y) * NX + g.x;
                    if (idx < heat.length) heat[idx] += 10.0; // Strong reward for pattern matching
                    console.info("Nebula AI: Catch Successful! Pattern Reinforced.");
                }
            }
            if (this.preyStatus[i] > 0) {
                this.preyStatus[i]--;
                if (this.preyStatus[i] === 0) {
                    p.position.set((Math.random()-0.5)*TANK_SIZE*0.8, (Math.random()-0.6)*TANK_SIZE*0.3, (Math.random()-0.5)*TANK_SIZE*0.8);
                    const c = [0xf43f5e, 0x38bdf8, 0xd9f99d, 0x818cf8][i%4];
                    const mesh = p as unknown as THREE.Mesh;
                    if (mesh.material) {
                        (mesh.material as THREE.MeshStandardMaterial).color.set(c); 
                        (mesh.material as THREE.MeshStandardMaterial).emissive.set(c);
                    }
                }
            }
        });

        this.renderer.render(this.scene, this.camera);
        this.controls.update();
        requestAnimationFrame(this.animate);
    }
}
new LifeNebula();
