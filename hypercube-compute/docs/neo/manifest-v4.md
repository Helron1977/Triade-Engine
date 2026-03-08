# Hypercube Neo: Manifest V4 Guide (SSOT)

Welcome to the **Hypercube Neo** documentation. From V4 onwards, the engine is entirely driven by a **Hierarchical JSON Manifest**. This file is the "Single Source of Truth" (SSOT) defining both the physical properties of the engine and the simulation configuration.

## Table of Contents

1. [The Root Manifest](#1-the-root-manifest)
2. [Engine Layer (`engine`)](./engine-layer.md)
   - Faces, Rules, Requirements
3. [Configuration Layer (`config`)](./config-layer.md)
   - Dimensions, Chunks, Mode, Boundaries
4. [Objects & Physics Layer (`objects`)](./objects-layer.md)
   - Shapes (Circle, Poly, Rect), Properties, Animation
5. [Code Integration Snippets](./integration.md)

---

## 1. The Root Manifest

Every simulation starts with a root object. 

### Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `$schema` | `string` | Optional URL to the JSON schema for IDE autocompletion. |
| `name` | `string` | Human readable name of the simulation case. |
| `version` | `string` | Version of the manifest (e.g., `1.0.0`). |
| `engine` | `object` | **Engine Layer**: Physics definition (Faces, Rules). |
| `config` | `object` | **Config Layer**: Geometry and Execution (Chunks, Resolution). |

### Example Snippet

```json
{
  "$schema": "https://hypercube-neo.dev/schema/v4.json",
  "name": "My LBM Simulation",
  "version": "1.0.0",
  "engine": { ... },
  "config": { ... }
}
```

Next step: [Discover the Engine Layer](./engine-layer.md)
