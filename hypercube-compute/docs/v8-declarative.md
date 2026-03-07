# Hypercube V8: Declarative Architecture Guide 🧪⚙️⚖️

L'architecture **V8 (Declarative)** est l'aboutissement de l'évolution d'Hypercube, visant à séparer totalement la **définition des lois physiques** de leur **implémentation matérielle**.

---

## 🏗️ Les 3 Piliers du V8

### 1. Le Manifeste (EngineDescriptor)
C'est le contrat unique. Il définit :
- **Faces** : Les couches de données (Ex: Température, Obstacles).
- **Parameters** : Les constantes physiques (Ex: Diffusion rate).
- **Physics Rules** : Le comportement sémantique (Rôles Inlets, Outlets, Walls).

### 2. Le Proxy (V8EngineProxy)
Le cerveau de l'application. Au lieu de manipuler des buffers, le développeur manipule des **Shapes** (Circle, Box).
- `proxy.addShape()` : Injecte instantanément de la physique dans le monde.
- `proxy.compute()` : Détermine le backend (CPU/GPU) et orchestre le tick.

### 3. Le Backend (Agnostic Flow)
- **GPU (WGSL)** : Utilise un système de "Ping-Pong" Zero-Copy et une synchronisation VRAM directe entre chunks.
- **CPU (SharedArrayBuffer)** : Utilise des WebWorkers multithreadés pour paralléliser le calcul sur le MasterBuffer.

---

## 🔄 Flux de Synchronisation

### "Ghost Cells" & Hardware Sync
En mode GPU, les bordures des chunks sont échangées directement via `copyBufferToBuffer` dans la VRAM. En mode CPU, elles sont échangées via `SharedArrayBuffer`.

> [!IMPORTANT]
> **Règles de Continuité V8** :
> 1. **Dual-Sync** : Si une face utilise un système de Ping-Pong (ex: `Temperature` et `TemperatureNext`), **les deux faces** doivent être marquées `isSynchronized: true` dans le manifeste. Sinon, la continuité physique sera perdue une frame sur deux.
> 2. **Parity propagation** : Le moteur V8 doit être "Parity-Aware". La parité est automatiquement synchronisée entre le thread principal et les WebWorkers pour garantir que tout le monde calcule la même phase du ping-pong.
> 3. **Worker Config** : Utilisez `getConfig()` et `applyConfig()` dans votre shim pour propager des états personnalisés à travers la grille multithreadée.

### Injection Dynamique (GPU-Aware)
Le `Rasterizer` est désormais "GPU-Aware". Chaque modification sur le CPU (via une Shape) déclenche un `syncFromHost` immédiat vers les buffers GPU actifs, permettant une interactivité fluide même en calcul intensif.

---

## 🚀 Comment créer un nouveau moteur V8 ?

1. **Définir le Manifeste** : Créez un fichier `.ts` exportant un `EngineDescriptor`.
2. **Coder le Kernel WGSL** (Optionnel pour GPU) : Écrivez le shader en respectant le `V8Uniforms` layout.
3. **Instancier via la Factory** :
```typescript
const proxy = await HypercubeFactory.instantiate(MyDescriptor, config, MyGpuKernel);
```

---

## 🧪 Showcase
- [06: V8 Heat GPU](file:///examples/06-heat-gpu-v8.ts) : Performance maximale avec double-buffering matériel.
- [07: V8 Heat CPU](file:///examples/07-heat-cpu-v8.ts) : Preuve du contrat agnostique (zéro changement de code logique).
