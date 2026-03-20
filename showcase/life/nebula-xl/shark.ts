import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';

export class SharkEntity {
    public group: THREE.Group;
    private model: THREE.Object3D | null = null;
    private speed = 0.35; 
    public speedMultiplier = 1.0;
    
    // 3D HUNTING (Pitch enabled)
    private steerCommitment = 0;
    private currentCommittedTarget = new THREE.Vector3(0, 0, 1);

    private animUniforms = {
        uTime: { value: 0 },
        uTurnLean: { value: 0 }
    };

    constructor(scene: THREE.Scene, private material: THREE.Material) {
        this.group = new THREE.Group();
        scene.add(this.group);
        
        // BIOLOGICAL 8-NODE VERTEBRAL DEFORMATION (V75.3)
        this.material.onBeforeCompile = (shader) => {
            shader.uniforms.uTime = this.animUniforms.uTime;
            shader.uniforms.uTurnLean = this.animUniforms.uTurnLean;
            shader.vertexShader = `
                uniform float uTime;
                uniform float uTurnLean;
            ` + shader.vertexShader;
            shader.vertexShader = shader.vertexShader.replace(
                '#include <begin_vertex>',
                `
                vec3 transformed = vec3(position);
                float t = clamp((position.x - (-123.0)) / (123.0 - (-123.0)), 0.0, 1.0);
                float flex = pow(t, 3.5) * 20.0; 
                float wavePhase = uTime * 5.5 - t * 4.0;
                float wag = sin(wavePhase);
                transformed.y += wag * flex;
                transformed.x += abs(wag) * flex * 0.2;
                transformed.y += uTurnLean * flex * 0.2;
                `
            );
        };

        const loader = new OBJLoader();
        loader.load('assets/shark.obj', (obj) => {
            obj.traverse((child) => {
                if ((child as THREE.Mesh).isMesh) {
                    (child as THREE.Mesh).material = this.material;
                }
            });
            
            obj.scale.set(0.015, 0.015, 0.015);
            obj.rotation.x = -Math.PI / 2;
            obj.rotation.z = Math.PI * 0.5;
            this.group.add(obj);
            this.model = obj;
        });
    }

    public update(suggestedDir: THREE.Vector3) {
        if (isNaN(this.group.position.x) || isNaN(this.group.quaternion.x)) {
            this.group.position.set(0, 2, 0);
            this.group.quaternion.set(0, 0, 0, 1);
            this.currentCommittedTarget.set(0, 0, 1);
        }

        const lastQuat = this.group.quaternion.clone();
        
        if (this.steerCommitment-- <= 0 || suggestedDir.lengthSq() < 0.1) {
            if (suggestedDir.lengthSq() > 0.0001) {
                this.currentCommittedTarget.copy(suggestedDir).normalize();
            }
            this.steerCommitment = 40 + Math.random() * 20;
        }

        // FULL 3D STEERING (V75.5) - Removed setY(0) to allow diving/rising
        if (this.currentCommittedTarget.lengthSq() > 0.0001) {
            const targetQuat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1), this.currentCommittedTarget);
            this.group.quaternion.slerp(targetQuat, 0.035); 
        }

        // Forward movement in direction of orientation
        const currentSpeed = this.speed * this.speedMultiplier;
        this.group.position.addScaledVector(new THREE.Vector3(0,0,1).applyQuaternion(this.group.quaternion), currentSpeed);
        
        const halfW = 39.5; 
        const halfH = 9.8;
        this.group.position.x = THREE.MathUtils.clamp(this.group.position.x, -halfW, halfW);
        this.group.position.z = THREE.MathUtils.clamp(this.group.position.z, -halfW, halfW);
        this.group.position.y = THREE.MathUtils.clamp(this.group.position.y, -halfH, halfH);

        const turnRate = 1.0 - Math.abs(this.group.quaternion.dot(lastQuat));
        const sideMultiplier = this.group.quaternion.y > lastQuat.y ? 1 : -1;
        this.animUniforms.uTurnLean.value = THREE.MathUtils.lerp(this.animUniforms.uTurnLean.value, turnRate * sideMultiplier * 45.0, 0.1);
        
        if (this.model && !isNaN(this.animUniforms.uTurnLean.value)) {
            this.model.rotation.z = Math.PI * 0.5 + this.animUniforms.uTurnLean.value * 0.6;
        }
    }

    public animate(time: number) {
        this.animUniforms.uTime.value = time;
    }

    public getSnoutPosition(): THREE.Vector3 {
        const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(this.group.quaternion);
        return this.group.position.clone().addScaledVector(forward, 1.85);
    }

    public getPosition() { return this.group.position; }
}
