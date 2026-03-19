import { defineConfig } from 'vite';

export default defineConfig({
    root: 'showcase', // Base directory for the showcase
    publicDir: 'assets', // Assets inside showcase/
    build: {
        outDir: '../dist-examples',
        emptyOutDir: true
    },
    server: {
        open: true,
        port: 3000,
        cors: true,
        headers: {
            // Essential for SharedArrayBuffer multithreading
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'credentialless',
        }
    }
});
