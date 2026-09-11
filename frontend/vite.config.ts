import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend serves the built assets from its own origin, so no CORS setup is
// needed in production. In development the dev server proxies to it, which
// keeps the frontend code identical in both cases.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
