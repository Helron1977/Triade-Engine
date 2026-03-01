# Rapport d'Analyse : Hypercube-Compute & Roadmap V3

Ce document détaille les axes d'amélioration pour le noyau de calcul et propose de nouveaux modules fonctionnels basés sur l'architecture des 6+1 Faces.

## 1. Analyse Stratégique : WebGL / WebGPU vs Frameworks

Le passage à une accélération matérielle pose la question de la "réinvention de la roue".

**Pourquoi continuer en "Bespoke" (Sur-mesure) ?**

*   **Architecture Unique :** Aucun framework majeur n'implémente nativement le pipeline des 6 Faces basé sur le SAT (Summed Area Table).
*   **Légèreté :** Hypercube-Compute vise l'efficacité brute sans le "bloat" (poids inutile) des moteurs comme Three.js ou Babylon.js.
*   **GPGPU Logique :** Contrairement aux bibliothèques d'effets visuels, votre moteur utilise le GPU pour la décision logicielle et non juste pour l'esthétique.

---

## 2. Optimisations du Core : Approche Hybride

Afin de garantir une flexibilité maximale, **le moteur laissera au développeur le choix explicite** entre l'accélération matérielle (WebGL/WebGPU) et le traitement logiciel pur (CPU).

### A. Passage au WebGPU / WebGL (Calcul Spatial)

Le calcul de la Face 5 (SAT) est le candidat idéal pour WebGPU.

*   **Prefix Sum Parallèle :** Implémenter l'intégration SAT via des Compute Shaders réduirait le temps de calcul de $O(N)$ à $O(\log N)$ sur le plan matériel.
*   **Zero-Copy :** WebGPU permet de lire les résultats des calculs directement pour le rendu sans faire d'allers-retours coûteux entre la RAM et la VRAM.

### B. Memory Pooling & Workers (Fallback CPU Optatif)

Pour les environnements ne supportant pas l'accélération matérielle, ou par choix explicite du développeur, le calcul CPU sera optimisé :

*   **SharedArrayBuffer :** Utiliser des Workers pour calculer les faces en parallèle sans duplication de mémoire.
*   **SIMD :** Utiliser les instructions vectorielles de WebAssembly pour accélérer les additions de l'intégration SAT.

---

## 3. Nouveaux Modules (Engines) Proposés

### 🏹 Flow-Field Pathfinding

*   **Face 3 (Influence) :** Stocke la distance à la cible (Dijkstra Map).
*   **Face 6 (Vector Field) :** Stocke les vecteurs de pente.
*   **Avantage :** Diriger 10 000 agents pour le coût d'une seule lecture de case mémoire par agent.

### 🌊 Fluid-Dynamics Simplifiée

*   **Mécanique :** Advection de densité entre les faces.
*   **Usage SAT :** Calculer instantanément la pression moyenne d'une zone pour simuler la flottabilité d'entités sur le 7ème plan.

---

## 4. Évolution vers le 7ème Plan (F7)

Le 7ème Plan devient un "Compositeur" :

*   Il reçoit les 6 faces comme des textures d'entrée.
*   Un shader de fragment effectue la synthèse finale (ex: `F3 * F4 + F1`) pour l'affichage utilisateur.

*Note : Cette architecture est idéale pour les simulations de type "God Games", Urbanisme ou RTS massivement peuplés.*





































