import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Die Oberfläche ist bewusst nur örtlich erreichbar - über die Unterlagen
// laufen Gesundheitsdaten. Anfragen an den Dienst werden weitergereicht,
// damit im Entwicklungsbetrieb keine Ursprungsbeschränkungen greifen.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/session": "http://127.0.0.1:8000",
      "/actions": "http://127.0.0.1:8000",
      "/status": "http://127.0.0.1:8000",
    },
  },
  preview: { host: "127.0.0.1", port: 4173 },
});
