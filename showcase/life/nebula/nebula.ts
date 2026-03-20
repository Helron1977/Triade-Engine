import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { HypercubeNeoFactory } from '../../../core/HypercubeNeoFactory';

/**
 * LIFE NEBULA V23.0 - THE LIQUID SOUL
 * Feature: Velocity-based Rings (Outward thrust)
 * Feature: No Spikes (Stable rho)
 * Feature: Erupting Splash (8-point star)
 */
const NX = 64; const NY = 64;
const TANK_SIZE = 20;
const SURFACE_Y = 8;

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

    private sharkVel = new THREE.Vector3(0, 0, 0);
    private currentTargetIndex = -1;
    private splashCooldown = 0;
    
    private waterMesh!: THREE.Mesh;
    private waterGeo!: THREE.PlaneGeometry;
    private underwaterMesh!: THREE.Mesh;

    constructor() {
        this.init3D().then(() => this.initEngine()).catch(console.error);
    }

    private async init3D() {
        const container = document.getElementById('canvas-container')!;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x010409);
        this.camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(22, 14, 22);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        this.scene.add(new THREE.HemisphereLight(0x38bdf8, 0x020617, 3));
        const sun = new THREE.DirectionalLight(0xffffff, 5.0);
        sun.position.set(10, 30, 20);
        this.scene.add(sun);

        this.setupModels();
    }

    private setupModels() {
        this.shark = new THREE.Group();
        const sMat = new THREE.MeshPhysicalMaterial({ color: 0x475569, metalness: 0.9, roughness: 0.1, clearcoat: 1.0 });
        const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.5, 1.4, 4, 16), sMat);
        body.rotation.x = Math.PI / 2;
        this.shark.add(body);
        this.shark.position.set(0, SURFACE_Y, 0);
        this.scene.add(this.shark);

        const colors = [0xf43f5e, 0x38bdf8, 0x10b981];
        for (let i = 0; i < this.preyCount; i++) {
            const mat = new THREE.MeshStandardMaterial({ color: colors[i%3], emissive: colors[i%3], emissiveIntensity: 2.0 });
            const p = new THREE.Mesh(new THREE.DodecahedronGeometry(0.18), mat);
            p.scale.set(1, 0.5, 1.8);
            p.position.set((Math.random()-0.5)*18, Math.random() * SURFACE_Y, (Math.random()-0.5)*18);
            this.scene.add(p);
            this.preyList.push(p as any);
            this.preyVels.push(new THREE.Vector3((Math.random()-0.5)*0.1, 0, (Math.random()-0.5)*0.1));
        }

        this.waterGeo = new THREE.PlaneGeometry(TANK_SIZE, TANK_SIZE, NX - 1, NY - 1);
        const colorAttr = new THREE.BufferAttribute(new Float32Array(NX * NY * 3), 3);
        this.waterGeo.setAttribute('color', colorAttr);
        const wMat = new THREE.MeshPhysicalMaterial({ 
            color: 0x0ea5e9, transparent: true, opacity: 0.65, transmission: 0.5, side: THREE.DoubleSide,
            metalness: 0.9, roughness: 0.05, clearcoat: 1.0, vertexColors: true
        });
        this.waterMesh = new THREE.Mesh(this.waterGeo, wMat);
        this.waterMesh.position.y = SURFACE_Y; this.waterMesh.rotation.x = -Math.PI / 2;
        this.scene.add(this.waterMesh);

        const uwMat = new THREE.MeshStandardMaterial({ color: 0x075985, transparent: true, opacity: 0.4 });
        this.underwaterMesh = new THREE.Mesh(new THREE.BoxGeometry(TANK_SIZE, SURFACE_Y, TANK_SIZE), uwMat);
        this.underwaterMesh.position.y = SURFACE_Y / 2;
        this.scene.add(this.underwaterMesh);
    }

    private async initEngine() {
        const factory = new HypercubeNeoFactory();
        const manifest = await factory.fromManifest('./nebula-manifest.json');
        this.engine = await factory.build(manifest.config, manifest.engine);
        if (document.getElementById('loader')) document.getElementById('loader')!.style.display = 'none';
        this.animate();
    }

    private worldToGrid(v: THREE.Vector3) {
        return {
            gx: (v.x / TANK_SIZE + 0.5) * 64,
            gy: (v.z / TANK_SIZE + 0.5) * 64
        };
    }

    private isUpdating = false;
    private updateAI = async () => {
        if (!this.engine || this.isUpdating) return;
        this.isUpdating = true;
        try {
            const bridge = this.engine.bridge;
            const chunk = this.engine.vGrid.chunks[0];
            const pos = this.worldToGrid(this.shark.position);

            const config = (this.engine as any).vGrid.config;
            if (!config.objects) config.objects = [];
            
            // --- VELOCITY RINGS: Erupting from the body ---
            const distToSurface = Math.abs(this.shark.position.y - SURFACE_Y);
            this.splashCooldown--;
            if (distToSurface < 1.0 && this.splashCooldown <= 0) {
                // 8-Point Eruption Star
                const thrust = 0.4;
                for (let i = 0; i < 8; i++) {
                    const angle = (i / 8) * Math.PI * 2;
                    config.objects.push({
                        id: `ring_${i}`, type: 'circle',
                        position: { x: pos.gx + Math.cos(angle)*3 - 1, y: pos.gy + Math.sin(angle)*3 - 1 },
                        dimensions: { w: 2, h: 2 },
                        properties: { vx: Math.cos(angle)*thrust, vy: Math.sin(angle)*thrust, rho: 1.1 },
                        rasterMode: "replace"
                    });
                }
                this.splashCooldown = 12;
            }

            await this.engine.step(1);
            await bridge.syncToHost();
            config.objects = config.objects.filter((o: any) => !o.id.startsWith('ring_'));

            const views = bridge.getChunkViews(chunk.id);
            const rIdx = this.engine.parityManager.getFaceIndices('rho').read;
            const rData = views[rIdx];

            const pAttr = this.waterGeo.attributes.position;
            const cAttr = this.waterGeo.attributes.color;
            const stride = NX + 2;
            for (let y = 0; y < NY; y++) {
                for (let x = 0; x < NX; x++) {
                    const vIdx = y * NX + x;
                    const v = rData[(y + 1) * stride + (x + 1)];
                    const win = Math.min(1.0, Math.min(x, y, NX-1-x, NY-1-y) / 4.0);
                    
                    const height = isNaN(v) ? 0 : (v - 1.0) * 180.0 * win; 
                    pAttr.setZ(vIdx, height);

                    // Physical White Crests
                    if (height > 0.4) {
                        cAttr.setXYZ(vIdx, 1.0, 1.0, 1.0);
                    } else if (height < -0.4) {
                        cAttr.setXYZ(vIdx, 0.0, 0.1, 0.3);
                    } else {
                        cAttr.setXYZ(vIdx, 0.05, 0.5, 0.82);
                    }
                }
            }
            pAttr.needsUpdate = true; cAttr.needsUpdate = true; this.waterGeo.computeVertexNormals();

            // AI Focus
            if (this.currentTargetIndex === -1 || Math.random() < 0.005) {
                let mD = 1000;
                this.preyList.forEach((p, idx) => {
                    const d = p.position.distanceTo(this.shark.position);
                    if (d < mD) { mD = d; this.currentTargetIndex = idx; }
                });
            }
            if (this.currentTargetIndex !== -1) {
                const target = this.preyList[this.currentTargetIndex];
                const delta = target.position.clone().sub(this.shark.position).normalize();
                this.sharkVel.lerp(delta.multiplyScalar(0.25), 0.06);
            }

        } catch (e) { console.error("Nebula Soul Error:", e); }
        finally { this.isUpdating = false; }
    }

    private animate = async () => {
        await this.updateAI();
        const b = 9.8; 
        
        this.shark.position.add(this.sharkVel);
        if (this.sharkVel.lengthSq() > 0.001) this.shark.lookAt(this.shark.position.clone().add(this.sharkVel));
        this.shark.position.clamp(new THREE.Vector3(-b, 0, -b), new THREE.Vector3(b, SURFACE_Y + 1, b));

        this.preyList.forEach((p, i) => {
            const v = this.preyVels[i];
            v.add(new THREE.Vector3((Math.random()-0.5)*0.012, (Math.random()-0.5)*0.008, (Math.random()-0.5)*0.012));
            
            const px = Math.abs(p.position.x);
            const pz = Math.abs(p.position.z);
            if (px > 5.0 || pz > 5.0) {
                v.add(new THREE.Vector3(-p.position.x, 0, -p.position.z).normalize().multiplyScalar(0.2));
            }

            const dS = p.position.distanceTo(this.shark.position);
            if (dS < 4.5) v.lerp(p.position.clone().sub(this.shark.position).normalize().multiplyScalar(0.3), 0.2);

            v.clampLength(0.04, 0.22);
            p.position.add(v);
            if (v.lengthSq() > 0.001) p.lookAt(p.position.clone().add(v));
            p.position.clamp(new THREE.Vector3(-9.2, 0.5, -9.2), new THREE.Vector3(9.2, SURFACE_Y + 0.5, 9.2));
        });

        this.renderer.render(this.scene, this.camera);
        this.controls.update();
        requestAnimationFrame(this.animate);
    }
}

new LifeNebula();
