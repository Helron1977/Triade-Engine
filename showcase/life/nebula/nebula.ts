import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { HypercubeNeoFactory } from '../../../core/HypercubeNeoFactory';

/**
 * LIFE NEBULA V43.0 - THE PRECISION POLISH
 * Goal: Perfectly aligned ripples. Working HUD. Visual Eat Feedback.
 */
const NX = 64; const NY = 64;
const TANK_SIZE = 20;
const SURFACE_Y = 10;
const H_LIMIT = 0.5;

class LifeNebula {
    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private renderer!: THREE.WebGLRenderer;
    private controls!: OrbitControls;
    
    private engine: any = null;
    private shark!: THREE.Group;
    private sharkMat!: THREE.MeshPhysicalMaterial;
    private preyCount = 65;
    private preyList: THREE.Group[] = [];
    private preyVels: THREE.Vector3[] = [];

    private sharkVel = new THREE.Vector3(0, 0, 0);
    private currentTargetIndex = -1;
    private splashCooldown = 0;
    private eatFlashCounter = 0;
    
    private surfaceGeo!: THREE.PlaneGeometry;

    constructor() {
        this.init3D().then(() => this.initEngine()).catch(console.error);
    }

    private async init3D() {
        const container = document.getElementById('canvas-container')!;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x011422);
        this.camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(24, 22, 24);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        this.scene.add(new THREE.HemisphereLight(0x38bdf8, 0x011422, 2.5));
        const sun = new THREE.DirectionalLight(0xffffff, 8.0);
        sun.position.set(10, 40, 20);
        this.scene.add(sun);
        
        this.setupModels();
    }

    private setupModels() {
        // 1. Full Deep Volume
        const cubeGeo = new THREE.BoxGeometry(TANK_SIZE, TANK_SIZE, TANK_SIZE);
        const cubeMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.3, transmission: 0.95,
            metalness: 0, roughness: 0.05, ior: 1.1, thickness: 2.0
        });
        this.scene.add(new THREE.Mesh(cubeGeo, cubeMat));

        // 2. Reactive Surface
        this.surfaceGeo = new THREE.PlaneGeometry(TANK_SIZE, TANK_SIZE, NX - 1, NY - 1);
        const colorAttr = new THREE.BufferAttribute(new Float32Array(this.surfaceGeo.attributes.position.count * 3), 3);
        this.surfaceGeo.setAttribute('color', colorAttr);
        const sMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.8, transmission: 0.7,
            vertexColors: true, metalness: 0.1, roughness: 0.02, clearcoat: 1.0, side: THREE.DoubleSide
        });
        const surfaceMesh = new THREE.Mesh(this.surfaceGeo, sMat);
        surfaceMesh.rotation.x = -Math.PI / 2;
        surfaceMesh.position.y = SURFACE_Y + 0.05;
        this.scene.add(surfaceMesh);

        // 3. Shark (Hunting Machine)
        this.shark = new THREE.Group();
        this.sharkMat = new THREE.MeshPhysicalMaterial({ color: 0x334155, metalness: 0.9, roughness: 0.1, clearcoat: 1.0 });
        const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.5, 1.4, 4, 16), this.sharkMat);
        body.rotation.x = Math.PI / 2;
        this.shark.add(body); this.scene.add(this.shark);

        // 4. Prey
        const colors = [0xf43f5e, 0x38bdf8, 0x10b981];
        for (let i = 0; i < this.preyCount; i++) {
            const mat = new THREE.MeshStandardMaterial({ color: colors[i%3], emissive: colors[i%3], emissiveIntensity: 2.5 });
            const p = new THREE.Mesh(new THREE.DodecahedronGeometry(0.18), mat);
            p.scale.set(1, 0.5, 1.8);
            p.position.set((Math.random()-0.5)*18, (Math.random()-0.5)*18, (Math.random()-0.5)*18);
            this.scene.add(p);
            this.preyList.push(p as any);
            this.preyVels.push(new THREE.Vector3((Math.random()-0.5)*0.1, (Math.random()-0.5)*0.1, (Math.random()-0.5)*0.1));
        }
    }

    private async initEngine() {
        const factory = new HypercubeNeoFactory();
        const manifest = await factory.fromManifest('./nebula-manifest.json');
        this.engine = await factory.build(manifest.config, manifest.engine);
        if (document.getElementById('loader')) document.getElementById('loader')!.style.display = 'none';
        this.animate();
    }

    private worldToGrid(v: THREE.Vector3) {
        // Map world [-10, 10] to grid [0, 63]
        // INVERT Y (Z) to match Plane Geometry orientation
        return {
            x: Math.floor((v.x / TANK_SIZE + 0.5) * 63),
            y: Math.floor((1.0 - (v.z / TANK_SIZE + 0.5)) * 63)
        };
    }

    private isUpdating = false;
    private updateAI = async () => {
        if (!this.engine || this.isUpdating) return;
        this.isUpdating = true;
        try {
            const bridge = this.engine.bridge;
            const chunk = this.engine.vGrid.chunks[0];
            const gPos = this.worldToGrid(this.shark.position);
            const config = (this.engine as any).vGrid.config;
            if (!config.objects) config.objects = [];
            
            // 1. WAVE TRIGGER (Corrected Alignment)
            const distSurf = Math.abs(this.shark.position.y - SURFACE_Y);
            if (distSurf < 1.0 && this.splashCooldown-- <= 0) {
                config.objects.push({
                    id: 'impact', type: 'circle',
                    position: { x: gPos.x - 7, y: gPos.y - 7 }, 
                    dimensions: { w: 14, h: 14 },
                    properties: { rho: 1.5 }, rasterMode: "replace" 
                });
                this.splashCooldown = 15;
            }

            await this.engine.step(1);
            await bridge.syncToHost();
            config.objects = config.objects.filter((o: any) => o.id !== 'impact');

            const views = bridge.getChunkViews(chunk.id);
            const rIdx = this.engine.parityManager.getFaceIndices('rho').read;
            const rData = views[rIdx];

            const pAttr = this.surfaceGeo.attributes.position;
            const cAttr = this.surfaceGeo.attributes.color;
            const stride = 66; 

            for (let i = 0; i < pAttr.count; i++) {
                const px = pAttr.getX(i); const pz = pAttr.getY(i);
                const gx = Math.floor((px / TANK_SIZE + 0.5) * 63);
                const gy = Math.floor((1.0 - (pz / TANK_SIZE + 0.5)) * 63); // MATCH INVERSION
                const v = rData[(gy + 1) * stride + (gx + 1)];
                const win = Math.min(1.0, Math.min(gx, gy, 63-gx, 63-gy) / 4.0);
                
                let h = isNaN(v) ? 0 : (v - 1.0) * 15.0 * win; 
                h = Math.max(-H_LIMIT, Math.min(H_LIMIT, h));
                pAttr.setZ(i, h);
                
                if (h > H_LIMIT * 0.8) cAttr.setXYZ(i, 1.0, 1.0, 1.0);
                else if (h < -0.05) cAttr.setXYZ(i, 0.05, 0.2, 0.45);
                else cAttr.setXYZ(i, 0.2, 0.6, 0.95);
            }
            pAttr.needsUpdate = true; cAttr.needsUpdate = true; this.surfaceGeo.computeVertexNormals();

            // 2. HUNTING LOGIC (With Eating Flash)
            if (this.currentTargetIndex === -1 || Math.random() < 0.005) {
                let mD = 1000;
                this.preyList.forEach((p, idx) => {
                    const d = p.position.distanceTo(this.shark.position);
                    if (d < mD) { mD = d; this.currentTargetIndex = idx; }
                });
            }
            if (this.currentTargetIndex !== -1) {
                const target = this.preyList[this.currentTargetIndex];
                const delta = target.position.clone().sub(this.shark.position);
                const dist = delta.length();
                if (dist < 1.3) {
                    target.position.set((Math.random()-0.5)*18, (Math.random()-0.5)*18, (Math.random()-0.5)*18);
                    this.currentTargetIndex = -1; 
                    this.eatFlashCounter = 12; // Start Flash
                } else {
                    this.sharkVel.lerp(delta.normalize().multiplyScalar(0.24), 0.07);
                }
            }
            
            // Visual Flash Logic
            if (this.eatFlashCounter > 0) {
                this.sharkMat.emissive.setHex(0xf43f5e);
                this.sharkMat.emissiveIntensity = this.eatFlashCounter / 6.0;
                this.eatFlashCounter--;
            } else {
                this.sharkMat.emissiveIntensity = 0;
            }

        } catch (e) { console.error("Nebula V43 Error:", e); }
        finally { this.isUpdating = false; }
    }

    private animate = async () => {
        const start = performance.now();
        await this.updateAI();
        const b = 9.8; 
        
        this.shark.position.add(this.sharkVel);
        if (this.sharkVel.lengthSq() > 0.001) this.shark.lookAt(this.shark.position.clone().add(this.sharkVel));
        this.shark.position.clamp(new THREE.Vector3(-b, -b, -b), new THREE.Vector3(b, SURFACE_Y, b));

        this.preyList.forEach((p, i) => {
            const v = this.preyVels[i];
            v.add(new THREE.Vector3((Math.random()-0.5)*0.015, (Math.random()-0.5)*0.015, (Math.random()-0.5)*0.015));
            const pos = p.position;
            if (Math.abs(pos.x) > b || Math.abs(pos.y) > b || Math.abs(pos.z) > b) v.add(pos.clone().multiplyScalar(-0.25));
            const dS = p.position.distanceTo(this.shark.position);
            if (dS < 4.5) v.lerp(p.position.clone().sub(this.shark.position).normalize().multiplyScalar(0.35), 0.22);
            v.clampLength(0.04, 0.22); p.position.add(v);
            if (v.lengthSq() > 0.001) p.lookAt(p.position.clone().add(v));
        });

        this.renderer.render(this.scene, this.camera);
        this.controls.update();
        
        const ms = performance.now() - start;
        const hud = document.getElementById('stat-fps'); // CORRECTED ID
        if (hud) hud.innerHTML = `${ms.toFixed(1)}ms`;
        requestAnimationFrame(this.animate);
    }
}

new LifeNebula();
