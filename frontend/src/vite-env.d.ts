/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "browser" in the web build: run the Python pipeline in-page under Pyodide. */
  readonly VITE_ENGINE?: "browser" | "server";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
