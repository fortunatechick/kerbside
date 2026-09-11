import type { Config } from "tailwindcss";

// Every colour is a CSS custom property defined in tokens.css. Naming them here
// means a component never writes a raw hex, and a light theme is a token swap
// rather than a sweep through the components.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "var(--bg-base)",
        surface: "var(--bg-surface)",
        raised: "var(--bg-raised)",
        overlay: "var(--bg-overlay)",
        subtle: "var(--border-subtle)",
        line: "var(--border-default)",
        strong: "var(--border-strong)",
        primary: "var(--text-primary)",
        secondary: "var(--text-secondary)",
        muted: "var(--text-muted)",
        inverse: "var(--text-inverse)",
        accent: "var(--accent)",
        "accent-hover": "var(--accent-hover)",
        "accent-dim": "var(--accent-dim)",
        free: "var(--flow-free)",
        slow: "var(--flow-slow)",
        stop: "var(--flow-stop)",
        shadowhl: "var(--shadow-hl)",
        motionhl: "var(--motion-hl)",
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SF Mono", "Menlo", "monospace"],
      },
      borderRadius: { ctl: "6px", panel: "8px", dialog: "10px" },
      boxShadow: { overlay: "0 8px 24px rgba(0,0,0,.5)" },
      transitionDuration: { fast: "140ms" },
    },
  },
  plugins: [],
} satisfies Config;
