# LBM Ocean 2.5D (Showcase 02)

Ce showcase démontre les capacités multi-physiques du moteur **Hypercube Neo** en croisant deux domaines simultanément :
1. **La Dynamique des Fluides (LBM D2Q9)** pour simuler la surface de l'eau et sa propagation d'ondes (Shallow Water Equation).
2. **L'Advection-Diffusion** pour simuler la propagation de matière biologique ou thermique *portée* par le courant d'eau.

---

## 🌊 Concepts de Base (Vulgarisation)

La méthode de Boltzmann sur réseau (LBM) ne simule pas l'eau comme des "gouttes" (particules) ou de vastes "blocs carrés" (volumes finis). Elle simule des champs de **probabilité de collision**.

Imaginez une maille de grille. Au lieu de dire "cette case contient 2 litres d'eau allant vers la droite", on dit "dans cette case, la probabilité que de l'eau voyage vers le NORD est $f_2$, vers l'EST est $f_1$", etc. 

Il y a 9 directions possibles (D2Q9) :
- `f0` : Eau stagnante (au repos)
- `f1` à `f4` : Eau voyageant vers les 4 points cardinaux (Haut, Bas, Gauche, Droite)
- `f5` à `f8` : Eau voyageant vers les 4 diagonales.

Chaque étape de la simulation (Frame) se décompose en 2 phases très simples :
1. **Streaming (Le Voyage)** : Les probabilités `f` se déplacent mathématiquement dans la case voisine vers laquelle elles pointent.
2. **Collision (Le Choc)** : Les probabilités de la case se mélangent ("Relaxation") et s'équilibrent pour recréer la macroscopie de l'eau (Vitesse Globale $\vec{v}$ et Hauteur de l'eau $\rho$).

### Différence avec l'Aérodynamique (Air vs Eau)
Dans le showcase Aérodynamique, $\rho$ (Rho) représente la *pression/densité* de l'air de votre soufflerie.
Dans ce showcase Océan (Shallow Water), $\rho$ (Rho) est mathématiquement interprétée comme la **HAUTEUR** de la surface de l'eau ! C'est ce qui nous permet de simuler des vagues en 2.5 dimensions sur une simple grille 2D.

---

## 🛠️ Comment Paramétrer l'Océan

L'océan est configuré via un **Manifeste Déclaratif V4** (JSON).

### 1. Les Visages ("Faces")
Contrairement à la V3, la V4 demande de déclarer ouvertement la RAM que le moteur va dédier à l'Océan.

```json
        "faces": [
            { "name": "f0", "type": "scalar", "isSynchronized": true },
            { "name": "f1", "type": "scalar", "isSynchronized": true },
            // ... jusqu'à f8
            { "name": "obstacles", "type": "mask", "isSynchronized": true, "isReadOnly": true },
            
            // Les variables macroscopiques locales (Pas besoin de se synchroniser entre les chunks !)
            { "name": "vx", "type": "scalar", "isSynchronized": false },
            { "name": "vy", "type": "scalar", "isSynchronized": false },
            { "name": "rho", "type": "scalar", "isSynchronized": false },
            
            // Le calque passif (Biologique)
            { "name": "biology", "type": "scalar", "isSynchronized": true }
        ]
```
*Note sur la performance : Les variables macroscopiques (`vx`, `vy`, `rho`) n'ont pas de `isSynchronized: true` car elles se recalculent localement. LBM n'a besoin d'échanger avec ses voisins que les populations frontalières (`f0-f8`).*

### 2. Les Paramètres Physiques (`params`)
Dans le bloc `rules`, vous contrôlez le comportement mathématique des équations.

```json
        "rules": [
            {
                "type": "neo-ocean-v1",
                "method": "OceanPhysics",
                "source": "f0",
                "params": {
                    "tau_0": 0.8,
                    "cflLimit": 0.38,
                    "bioDiffusion": 0.05,
                    "bioGrowth": 0.0005
                }
            }
        ]
```

- **`tau_0` (Le Viscosité)** : Le temps de relaxation (Tau). Plus il est proche de `0.5`, plus l'eau est fluide (très instable mathématiquement). Plus il est grand (`0.8` à `1.5`), plus l'eau est visqueuse comme du miel, et les ondes se dissipent vite.
- **`cflLimit` (La Limite de Courant)** : Le ratio Courant-Friedrichs-Lewy. C'est un limiteur mathématique qui empêche les fluides de voyager plus d'une case par frame. S'il est dépassé, la simulation explose.
- **`bioDiffusion`** : À quelle vitesse l'algue biologique se répand d'elle-même dans une eau totalement stagnante.
- **`bioGrowth`** : Le taux de croissance ("Reproduction") de l'algue. A `0.0005`, l'algue se multiplie légèrement à chaque frame jusqu'à atteindre un plafond naturel (1.0).

### 3. Les Objets (`objects`)
C'est ici que vous "Dessinez" l'état initial de l'eau.

```json
            {
                "id": "splash_init",
                "type": "circle",
                "position": { "x": 240, "y": 240 },
                "dimensions": { "w": 30, "h": 30 },
                "properties": {
                    "rho": 2.2,
                    "f0": 0.9,
                    //... autres populations
                    "biology": 1.0
                },
                "rasterMode": "add"
            }
```
Créer un "Splash" consiste à poser explicitement une zone (cercle) où la hauteur locale $\rho$ est anormalement élevée (ex: `2.2`). Au T=0, la physique va s'empresser d'écraser cet excédent, créant de gigantesques vagues circulaires. Le mode `"add"` au lieu de `"replace"` permet d'additionner l'eau par-dessus l'eau déjà existante au repos (`1.0`).

---

Cet océan CPU tire parti du moteur **Hypercube Neo**. Si vous indiquez `"chunks": { "x": 2, "y": 2 }` et `"executionMode": "parallel"`, l'océan de `512x512` sera découpé en 4 mers de `128x128`.

Hypercube va instancier **4 Web Workers** dédiés en parallèle. Le `BoundarySynchronizer` injectera nativement les populations `f` frontalières d'un serveur à l'autre sans que le noyau mathématique `NeoOceanKernel` ne s'en aperçoive.
Ceci vous permet de simuler des océans LBM à une vitesse impossible pour le fil principal du navigateur Javascript.

---

## ⚡ Mode GPU & Rendu Zero-Stall (High-Res)

Pour les simulations massives (512x512 et plus), le mode **GPU (WebGPU)** est recommandé.

![Démonstration de l'Océan GPU à 60 FPS](./media/ocean-gpu-demo.webp)
*Simulation 512x512 tournant intégralement en VRAM avec le renderer WebGPU natif.*

### 1. Pourquoi le mode GPU ?
Contrairement au mode CPU qui doit synchroniser des morceaux de mémoire (chunks), le mode GPU traite l'océan comme un **bloc monolithique en VRAM**. 
- **Zéro Goulot d'étranglement** : Le calcul physique (LBM) et le rendu 2.5D se font sur la carte graphique. Aucune donnée ne circule vers le processeur (CPU) pendant la boucle de simulation.
- **Rendu Instancié** : Chaque cellule de l'océan est dessinée comme un cube 3D indépendant par le GPU, permettant une netteté parfaite sans sacrifier le framerate.

### 2. Configuration GPU
Il suffit de changer le `mode` dans votre manifeste :
```json
{
    "config": {
        "mode": "gpu",
        "chunks": { "x": 1, "y": 1 }
    }
}
```

---

## 🏰 Import de Topologie (Game Developers)

Hypercube Neo n'est pas limité à des ondes circulaires. Vous pouvez importer des **mondes complexes**.

### L'API `setFaceData`
Pour les développeurs souhaitant intégrer Hypercube dans un jeu vidéo, vous pouvez injecter directement une image de topographie (Heightmap) :

```typescript
// On récupère les données d'une image de 512x512
const obstaclesMap = myImageLoader.getBuffer(); 

// On "peint" la topologie directement dans le MasterBuffer via le Bridge
engine.bridge.setFaceData('chunk_0', 'obstacles', obstaclesMap);
```

**Applications :**
- **Labyrinthes sous-marins** : L'eau rebondira sur chaque mur dessiné.
- **Îles et Côtes** : Importez votre carte du monde pour voir comment les vagues de tempête impactent vos rivages.
- **Bathymétrie** : Définissez les zones de profondeur pour influencer physiquement la vitesse des ondes.
