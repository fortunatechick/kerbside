/**
 * Pack the Python package for the in-browser engine.
 *
 * Collects every module under backend/kerbside into one JSON file of
 * { "kerbside/…/module.py": "source" }, which the engine's worker writes into
 * Pyodide's virtual filesystem before importing it. One request instead of
 * sixty, and no archive format to depend on.
 *
 * The standalone documents in docs/ are copied alongside so the deployed site
 * carries the overview and the presentation as well as the working console.
 *
 * Run by `npm run build:web`; both outputs are build products and gitignored.
 */

import { readdirSync, readFileSync, writeFileSync, mkdirSync, statSync, copyFileSync } from "node:fs";
import { join, relative, sep, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontend = join(here, "..");
const repo = join(frontend, "..");
const packageRoot = join(repo, "backend", "kerbside");
const backendRoot = join(repo, "backend");

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    if (name === "__pycache__") continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...walk(path));
    else if (name.endsWith(".py")) out.push(path);
  }
  return out;
}

const files = walk(packageRoot).sort();
if (files.length < 25) {
  // A glob that silently matches nothing would ship an engine that fails at
  // import time in the visitor's browser, which is the worst place to find out.
  throw new Error(`only ${files.length} Python modules found under ${packageRoot}`);
}

const bundle = {};
for (const file of files) {
  bundle[relative(backendRoot, file).split(sep).join("/")] = readFileSync(file, "utf8");
}

const engineDir = join(frontend, "public", "engine");
mkdirSync(engineDir, { recursive: true });
writeFileSync(join(engineDir, "kerbside.json"), JSON.stringify(bundle));

const docsSrc = join(repo, "docs");
const docsOut = join(frontend, "public", "docs");
mkdirSync(docsOut, { recursive: true });
let docs = 0;
for (const name of readdirSync(docsSrc)) {
  if (name.endsWith(".html")) {
    copyFileSync(join(docsSrc, name), join(docsOut, name));
    docs += 1;
  }
}

const kb = Math.round(JSON.stringify(bundle).length / 1024);
console.log(`bundled ${files.length} Python modules (${kb} KB) and ${docs} documents`);
