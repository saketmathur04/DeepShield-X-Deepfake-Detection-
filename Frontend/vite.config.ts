// DeepShield-X Vite + TanStack Start configuration
// Configured with TanStack Start, React, Tailwind CSS, TypeScript path aliases, and Nitro server build.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

export default defineConfig({
  tanstackStart: {
    // Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
    // nitro/vite builds from this
    server: { entry: "server" },
  },
});
