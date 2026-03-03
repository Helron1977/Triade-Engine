# User Specific Rules - Antigravity AI

## Environment & Shell (Windows PowerShell)
- **NEVER** use `&&` to chain commands. This operator is not natively supported in many versions of PowerShell or causes issues in the current execution environment.
- **ALWAYS** use `;` to separate commands (e.g., `git add . ; git commit -m "msg"`).

## Performance & Compute (Hypercube Engine)
- **SharedArrayBuffer Warning**: GitHub Pages and most standard web hosts do not serve the required `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers by default. 
- **Consequence**: `SharedArrayBuffer` will be unavailable on these platforms, leading to a fallback to standard `ArrayBuffer`.
- **Optimization**: Ensure engines are robust to CPU sequential fallback when multi-threading is unavailable.

## Multi-Chunk Sync & Artifacts
- **LBM Boundaries**: Always respect the 1-pixel ghost cell margin. Loops must iterate from `1` to `N-2`.
- **getSyncFaces**: Ensure all engines implement `getSyncFaces` to synchronize microscopic populations (faces 0-8) and macro variables (faces 18-20).

## CI/CD & Deployment
- **Documentation Demos**: Whenever you modify an example demo that is listed in the documentation (e.g. `01` to `07`), you **MUST** recompile it (`npm run build`) and correctly copy/deploy its built assets into the root `/docs/` folder of the repository. If you skip this step, the online GitHub Pages links will point to stale, broken, or older versions of the examples.
