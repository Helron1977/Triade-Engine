# Hypercube Neo: Tensor-CP Showcase 🧮

Ce guide explique comment utiliser le moteur **Hypercube Neo** pour effectuer une décomposition de tenseur en facteurs de rang faible (CP Decomposition) via l'algorithme **ALS (Alternating Least Squares)**.

---

## 🏗️ Principe de Fonctionnement
Contrairement aux simulations physiques classiques (fluides, chaleur), le showcase **Tensor-CP** traite les données comme un cube 3D (ex: Utilisateurs × Items × Temps). 

L'algorithme ALS cherche à approximer ce tenseur $X$ par une somme de produits extérieurs de trois matrices de facteurs $A, B$ et $C$ :
$$X \approx \sum_{r=1}^{R} a_r \otimes b_r \otimes c_r$$

### Avantages de l'implémentation Neo :
- **Zero-Allocation** : Les matrices de facteurs sont stockées directement dans les faces du `MasterBuffer`.
- **Backend Agnostique** : Le même manifeste pilote le calcul sur CPU (WebWorkers) ou GPU (WebGPU).
- **Interactivité** : Visualisation en temps réel de la courbe de convergence et des facteurs extraits.

---

## 📄 Manifeste Type
```json
{
  "name": "Tensor CP/ALS Explorer",
  "type": "tensor-cp",
  "rank": 3,
  "params": {
    "maxIterations": 50,
    "tolerance": 1e-4,
    "regularization": 0.01
  },
  "faces": [
    { "name": "mode_a", "isSynchronized": true },
    { "name": "mode_b", "isSynchronized": true },
    { "name": "mode_c", "isSynchronized": true },
    { "name": "target", "isSynchronized": true }
  ]
}
```

---

## 🚀 Utilisation
1. Accédez à `showcase/tensor-cp/index.html`.
2. Chargez un fichier CSV (format: index1, index2, index3, valeur).
3. Choisissez le **Rank** (nombre de facteurs latents).
4. Lancez la décomposition.
5. Observez l'erreur quadratique moyenne (MSE) diminuer à chaque itération.
