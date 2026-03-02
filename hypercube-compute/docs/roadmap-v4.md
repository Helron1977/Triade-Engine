# Roadmap Hypercube V4 : Scalabilité et Démonstration

Cette feuille de route s'inscrit dans la continuité de la V3. Le socle technologique (multithreading CPU, WebGPU, shaders physiques LBM) étant validé, la V4 se concentre sur **l'exploitabilité**, **la visibilité** et **l'extension 3D**.

## 1. Visibilité et Preuves (Démonstration & Benchmarks)

Le projet nécessite de passer du statut de POC expérimental à celui de Framework public :
- **Hub de Démos Unifié** : Un point d'entrée unique (GH Pages) permettant de switcher sans rechargement entre les implémentations phares :
  - *LBM D2Q9* : Aérodynamique en temps réel.
  - *Ocean Simulator* : Fluide macroscopique et friction.
  - *Swarm Intelligence* : Pathfinding Flow-Field multi-chunks.
- **Benchmarks chiffrés** : Publication d'un tableau comparatif dans la documentation (JS Objet Classique vs Hypercube CPU Workers vs Hypercube WebGPU).
- **Documentation de l'API** : Snippets rapides ("Quick Start") illustrant la simplicité de créer un moteur vectoriel (`FlowFieldEngine`, `FluidEngine`).

## 2. Rendu Multi-Tuiles WebGPU (Compositor V2)

Actuellement, l'architecture permet le calcul multi-chunks asynchrone, mais l'affichage WebGPU (le `HypercubeCompositor`) est encore limité au premier bloc (`cubes[0][0]`).
- **Évolution** : Refonte du pipeline WebGPU pour envoyer l'intégralité d'une `HypercubeGrid` (NxM chunks) directement au Fragment Shader.
- **Techniques visées** : Mega-Textures WebGPU ou tableau de BindGroups dynamiques pour assembler sans coutures les blocs à l'écran.

## 3. Dimension Supérieure : Foundation 3D (Z-Axis)

Le moteur "Hypercube" porte enfin son nom. L'architecture $O(1)$ à base de tableau 1D contigus permet théoriquement une extension gratuite en 3D (`index = z * (width * height) + y * width + x`).
- **Variables Z** : Introduction du paramètre `depth` dans l'allocation dynamique du `HypercubeMasterBuffer`.
- **Stencil 3D** : Création d'un premier stub `LBM D3Q19` (19 vecteurs de propagation au lieu de 9 pour la 2D).
- L'objectif n'est pas de faire un ray-tracer complexe, mais de prouver que la *mécanique tensorielle* peut gérer 20 millions de voxels avec les mêmes avantages de cache CPU.

---
*L'accomplissement de ces axes permettra de publier le projet sur HackerNews ou r/webgpu avec un effet "Waouh" indéniable et un code source robuste.*
