/*
 * The in-browser engine.
 *
 * Runs the real Python pipeline - the same `kerbside` package the server runs -
 * under Pyodide, CPython compiled to WebAssembly. It lives in a worker so that
 * a two-second analysis does not freeze the page while it runs.
 *
 * Protocol (all messages are plain objects):
 *   page -> worker  { id, method, path, query, body }
 *   worker -> page  { type: "response", id, status, contentType, body }
 *                   { type: "progress", phase, done, total }
 *                   { type: "ready" } | { type: "idle" } | { type: "fatal", message }
 *
 * Requests are handled strictly one at a time. The session is a single mutable
 * object, and letting two requests interleave at an await would let one read
 * state the other is halfway through changing.
 */

const PYODIDE_VERSION = "0.29.5";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const PACKAGES = ["numpy", "scipy", "opencv-python", "pydantic", "fastapi"];

let dispatchText = null;
let queue = Promise.resolve();
let pending = 0;

function progress(phase, done, total) {
  self.postMessage({ type: "progress", phase, done, total });
}

async function boot() {
  progress("runtime", 0, 1);
  importScripts(`${PYODIDE_URL}pyodide.js`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  progress("runtime", 1, 1);

  let loaded = 0;
  progress("packages", 0, PACKAGES.length);
  await pyodide.loadPackage(PACKAGES, {
    messageCallback: (message) => {
      if (/^Loaded /.test(message)) progress("packages", Math.min(++loaded, PACKAGES.length), PACKAGES.length);
    },
    errorCallback: (message) => console.warn("[engine]", message),
  });
  progress("packages", PACKAGES.length, PACKAGES.length);

  progress("source", 0, 1);
  const response = await fetch(new URL("kerbside.json", self.location.href));
  if (!response.ok) throw new Error(`could not fetch the Python source (${response.status})`);
  const bundle = await response.json();
  const root = "/home/pyodide";
  for (const [path, source] of Object.entries(bundle)) {
    const full = `${root}/${path}`;
    pyodide.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
    pyodide.FS.writeFile(full, source);
  }
  pyodide.runPython(`import sys\nif "${root}" not in sys.path: sys.path.insert(0, "${root}")`);
  await pyodide.runPythonAsync("from kerbside.api.inprocess import dispatch_text, set_progress");
  progress("source", 1, 1);

  pyodide.globals.get("set_progress")((phase, done, total) => progress(phase, done, total));
  dispatchText = pyodide.globals.get("dispatch_text");
  self.postMessage({ type: "ready" });
}

const booted = boot().catch((error) => {
  self.postMessage({ type: "fatal", message: String(error && error.message ? error.message : error) });
  throw error;
});

async function handle({ id, method, path, query, body }) {
  await booted;
  let result = null;
  try {
    result = await dispatchText(method, path, JSON.stringify(query || {}), body == null ? null : body);
    const [status, contentType, bytesProxy] = result.toJs();
    // Always copy. toJs can return a view onto the WebAssembly heap, and
    // transferring that buffer to the page would detach Python's own memory.
    const bytes = new Uint8Array(bytesProxy);
    self.postMessage({ type: "response", id, status, contentType, body: bytes }, [bytes.buffer]);
  } catch (error) {
    const text = new TextEncoder().encode(JSON.stringify({ detail: `engine error: ${error && error.message ? error.message : error}` }));
    self.postMessage({ type: "response", id, status: 500, contentType: "application/json", body: text }, [text.buffer]);
  } finally {
    if (result && typeof result.destroy === "function") result.destroy();
  }
}

self.onmessage = (event) => {
  pending += 1;
  queue = queue
    .then(() => handle(event.data))
    .finally(() => {
      pending -= 1;
      if (pending === 0) self.postMessage({ type: "idle" });
    });
};
