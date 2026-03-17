# User Specific Rules - Antigravity AI

## 🧠 Cognitive Workflow
- **Recursive Analysis**: Always analyze the repository recursively and stay up-to-date with the latest structure, code, docs, and manifests.
- **Honest Critique**: Provide direct and constructive feedback—highlighting strengths but never hiding real structural or architectural flaws.
- **Manifest-First Philosophy**: Propose solutions that are clean, elegant, and consistent with the core principles: manifest-first, zero-allocation, maximum factorization, and reuse of the `HypercubeNeoFactory`.
- **Excellence Standards**: Prioritize clarity, maintainability, and the elimination of duplication (especially between CPU and GPU modes).
- **Showcase & UX**: Ensure a premium user experience with beautiful `index.html` dashboards and crystal-clear documentation.

## 💻 Coding Standards
- **Clean Code**: Provide well-commented, professional code.
- **Structure**: Always include the corresponding manifest and the proposed folder structure when delivering code.
- **Architectural Evolution**: Proactively suggest improvements to the separation of core, kernels, and showcases.
- **Tone**: Professional, precise, motivating, and "no bullshit".

## 🛠️ Environment & Shell (Windows PowerShell)
- **NEVER** use `&&` to chain commands. Use `;` instead.
- **Git Persistence**: Always follow through with `git add .`, `commit`, and `push` after major changes.

## 🌪️ Performance & Compute (Hypercube Engine)
- **SharedArrayBuffer Warning**: Ensure robust sequential fallback when COOP/COEP headers are missing.
- **LBM Boundaries**: Always respect the 1-pixel ghost cell margin (loops from `1` to `N-2`).
- **Sync Integrity**: All engines must implement `getSyncFaces` for correct microscopic/macroscopic synchronization.

## 📦 CI/CD & Deployment
- **Documentation Demos**: Recompile demos (`npm run build`) and sync assets to `/docs/` when modifying examples to keep GitHub Pages live.
