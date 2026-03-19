import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { HypercubeNeoFactory } from '../../../core/HypercubeNeoFactory';

/**
 * LIFE NEBULA V10.3 - Extremely Robust Edition
 * Architecture: 2.5D (XY=Surface, Z=Depth)
 */
const NX = 64; const NY = 64; const NZ = 32;
const TANK_SIZE = 20;

enum SharkState {
    PATROL = "SEARCHING VOLUME",
    HUNT = "VOLUMETRIC CHASE",
    AMBUSH = "DEEP BRAIN AMBUSH",
    WANDER = "EXPLORING NEBULA"
}

class LifeNebula {
    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private renderer!: THREE.WebGLRenderer;
    private controls!: OrbitControls;
    
    private engine: any = null;
    private shark!: THREE.Group;
    private preyCount = 40; // Reduced for stability
    private preyList: THREE.Group[] = [];
    private preyVels: THREE.Vector3[] = [];
    private preyStatus: number[] = [];

    private sharkVel = new THREE.Vector3(0, 0, 0);
    private sharkState: SharkState = SharkState.PATROL;
    private stateTimer = 0;
    private ambushTarget = new THREE.Vector3(0, 0, 0);

    private waterMesh!: THREE.Mesh;
    private waterGeo!: THREE.PlaneGeometry;
    private debugHUD!: HTMLDivElement;

    constructor() {
        this.setupHUD();
        this.init3D().then(() => this.initEngine()).catch(e => {
            console.error("Nebula: Global Init Error:", e);
            this.updateHUD("CRITICAL ERROR: " + e.message);
        });
    }

    private setupHUD() {
        this.debugHUD = document.createElement('div');
        this.debugHUD.style.cssText = 'position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); color: #0ea5e9; padding: 10px; font-family: monospace; z-index: 1000; border: 1px solid #0ea5e9; border-radius: 4px;';
        this.debugHUD.innerText = "Initializing Nebula...";
        document.body.appendChild(this.debugHUD);
    }

    private updateHUD(msg: string) {
        if (this.debugHUD) this.debugHUD.innerText = `Nebula Diagnostics: ${msg}`;
        console.info(`Nebula: ${msg}`);
    }

    private async init3D() {
        this.updateHUD("Initializing Three.js...");
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

        this.scene.add(new THREE.HemisphereLight(0x38bdf8, 0x020617, 2));
        const sun = new THREE.DirectionalLight(0xffffff, 4.0);
        sun.position.set(10, 25, 15);
        this.scene.add(sun);

        this.setupModels();
    }

    private setupModels() {
        this.updateHUD("Creating Models...");
        this.shark = new THREE.Group();
        const sMat = new THREE.MeshPhysicalMaterial({ color: 0x475569, metalness: 0.8, roughness: 0.2 });
        const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.5, 1.4, 4, 16), sMat);
        body.rotation.x = Math.PI / 2; body.scale.set(1, 1, 0.7);
        this.shark.add(body);
        this.scene.add(this.shark);

        const colors = [0xf43f5e, 0x38bdf8, 0xd9f99d, 0x818cf8];
        for (let i = 0; i < this.preyCount; i++) {
            const mat = new THREE.MeshStandardMaterial({ color: colors[i%4], emissive: colors[i%4], emissiveIntensity: 1.0 });
            const p = new THREE.Mesh(new THREE.DodecahedronGeometry(0.18), mat);
            p.scale.set(1, 0.4, 1.6);
            this.scene.add(p);
            this.preyList.push(p as any);
            this.preyVels.push(new THREE.Vector3((Math.random()-0.5)*0.1, (Math.random()-0.5)*0.1, (Math.random()-0.5)*0.1));
            this.preyStatus.push(0);
        }

        this.waterGeo = new THREE.PlaneGeometry(TANK_SIZE, TANK_SIZE, NX - 1, NY - 1);
        const wMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.6, transmission: 0.5, side: THREE.DoubleSide,
            metalness: 0.9, roughness: 0.05, clearcoat: 1.0
        });
        this.waterMesh = new THREE.Mesh(this.waterGeo, wMat);
        this.waterMesh.position.y = TANK_SIZE * 0.25; 
        this.waterMesh.rotation.x = -Math.PI / 2;
        this.scene.add(this.waterMesh);
    }

    private async initEngine() {
        try {
            this.updateHUD("Building Hypercube Core...");
            const factory = new HypercubeNeoFactory();
            const manifest = await factory.fromManifest('./nebula-manifest.json');
            this.engine = await factory.build(manifest.config, manifest.engine);
            this.updateHUD("Core Online. Starting Simulation.");
            this.animate();
        } catch (e: any) {
            console.error("Nebula Engine Init Error:", e);
            this.updateHUD("ENGINE ERROR: " + e.message);
        }
    }

    private worldToGrid(v: THREE.Vector3) {
        return {
            x: Math.floor(Math.max(0, Math.min(NX - 1, (v.x / TANK_SIZE + 0.5) * NX))),
            y: Math.floor(Math.max(0, Math.min(NY - 1, (v.z / TANK_SIZE + 0.5) * NY))),
            z: Math.floor(Math.max(0, Math.min(NZ - 1, (v.y / (TANK_SIZE * 0.5) + 0.5) * NZ)))
        };
    }

    private gridToWorld(gx: number, gy: number, gz: number) {
        return new THREE.Vector3(
            ((gx / NX) - 0.5) * TANK_SIZE,
            ((gz / NZ) - 0.5) * TANK_SIZE * 0.5,
            ((gy / NY) - 0.5) * TANK_SIZE
        );
    }

    private isUpdating = false;
    private updateAI = async () => {
        if (!this.engine || this.isUpdating) return;
        this.isUpdating = true;
        try {
            const bridge = this.engine.bridge;
            const chunk = this.engine.vGrid.chunks[0];
            const nX = NX, nY = NY, nZ = NZ;
            const getIdx = (gx: number, gy: number, gz: number) => (gz * nY + gy) * nX + gx;

            // Shark Wake
            if (Math.abs(this.shark.position.y - TANK_SIZE * 0.24) < 2.5) {
                const g = this.worldToGrid(this.shark.position);
                const config = (this.engine as any).vGrid.config;
                if (!config.objects) config.objects = [];
                config.objects.push({
                    id: 'shark_wake', type: 'circle',
                    position: { x: g.x - 3, y: g.y - 3 }, dimensions: { w: 7, h: 7 },
                    properties: { rho: 8.0, biology: 1.0 }, rasterMode: "replace"
                });
            }
            
            await this.engine.step(1);
            await bridge.syncToHost();
            
            const config = (this.engine as any).vGrid.config;
            if (config.objects) config.objects = config.objects.filter((o: any) => o.id !== 'shark_wake');

            // Data Retrieval
            const views = bridge.getChunkViews(chunk.id);
            const waterIdx = this.engine.parityManager.getFaceIndices('water_h').read;
            const hData = views[waterIdx];
            
            const pAttr = this.waterGeo.attributes.position;
            for (let y=0; y<NY; y++) {
                for (let x=0; x<NX; x++) {
                    const vertIdx = y * NX + x;
                    const v = hData[vertIdx];
                    pAttr.setZ(vertIdx, isNaN(v) ? 0 : (v - 1.0) * 45.0); 
                }
            }
            pAttr.needsUpdate = true; this.waterGeo.computeVertexNormals();

            // Minimal Steering
            const predIndices = this.engine.parityManager.getFaceIndices('sdf_predator_x');
            const px = views[predIndices.read];
            const gPos = this.worldToGrid(this.shark.position);
            const tx = px[getIdx(gPos.x, gPos.y, gPos.z)];
            if (tx !== -10000 && tx !== undefined) {
                const targetPos = this.gridToWorld(tx, gPos.y, gPos.z);
                const desired = targetPos.sub(this.shark.position).normalize().multiplyScalar(0.15);
                this.sharkVel.lerp(desired, 0.1);
            }
        } catch (e) {
            console.error("Nebula update error:", e);
        } finally {
            this.isUpdating = false;
        }
    }

    private animate = async () => {
        await this.updateAI();
        const b = TANK_SIZE * 0.49; const vh = TANK_SIZE * 0.24;
        this.shark.position.add(this.sharkVel);
        if (this.sharkVel.lengthSq() > 0.001) this.shark.lookAt(this.shark.position.clone().add(this.sharkVel));
        this.shark.position.clamp(new THREE.Vector3(-b, -vh, -b), new THREE.Vector3(b, vh, b));

        this.preyList.forEach((p, i) => {
            p.position.add(this.preyVels[i]);
            if (this.preyVels[i].lengthSq() > 0.001) p.lookAt(p.position.clone().add(this.preyVels[i]));
            p.position.clamp(new THREE.Vector3(-b, -vh, -b), new THREE.Vector3(b, vh, b));
        });

        this.renderer.render(this.scene, this.camera);
        this.controls.update();
        requestAnimationFrame(this.animate);
    }
}

new LifeNebula();
