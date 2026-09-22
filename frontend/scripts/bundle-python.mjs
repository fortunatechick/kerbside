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

// Every subpackage the engine imports must be present. A missing one does not
// fail the build on its own - it fails at import time, in the visitor's browser,
// after a 35 MB download. That happened once: an unanchored "data/" in
// .vercelignore stripped backend/kerbside/data/ from the deployment.
const REQUIRED = ["api", "core", "data", "eval", "geometry", "operators", "stages"];
const present = new Set(files.map((f) => relative(packageRoot, f).split(sep)[0]));
const missing = REQUIRED.filter((pkg) => !present.has(pkg));
if (missing.length) {
  throw new Error(
    `kerbside subpackage(s) missing from the build: ${missing.join(", ")}. ` +
      `Check .vercelignore - an unanchored pattern matches at any depth.`,
  );
}
for (const pkg of REQUIRED) {
  if (!files.some((f) => relative(packageRoot, f).split(sep).join("/") === `${pkg}/__init__.py`)) {
    throw new Error(`kerbside/${pkg}/__init__.py is missing from the build`);
  }
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
