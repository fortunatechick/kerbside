/**
 * Where requests go: to the Python server, or to the in-browser engine.
 *
 * The console is built one of two ways. The default build talks to the FastAPI
 * server over HTTP. The web build (`npm run build:web`, which is what the Vercel
 * deployment serves) runs the same Python pipeline inside the page under
 * Pyodide, in a worker, and hands it each request instead. Everything above this
 * module is identical in both builds; the only places that care are the four
 * that bypass `fetch` - images, and the two downloads.
 */

import { useEffect, useRef, useState } from "react";

export const IN_BROWSER = import.meta.env.VITE_ENGINE === "browser";

export interface EngineResponse {
  status: number;
  contentType: string;
  body: Uint8Array<ArrayBuffer>;
}

export interface EngineProgress {
  phase: string;
  done: number;
  total: number;
}

type Listener = (event: EngineProgress | { phase: "ready" | "idle" | "fatal"; message?: string }) => void;

let worker: Worker | null = null;
let nextId = 1;
const waiting = new Map<number, (response: EngineResponse) => void>();
const listeners = new Set<Listener>();
let readyResolve: () => void = () => undefined;
let readyReject: (error: Error) => void = () => undefined;
export const engineReady: Promise<void> = new Promise((resolve, reject) => {
  readyResolve = resolve;
  readyReject = reject;
});

function start(): Worker {
  if (worker) return worker;
  worker = new Worker(`${import.meta.env.BASE_URL}engine/worker.js`);
  worker.onmessage = (event: MessageEvent) => {
    const data = event.data;
    if (data.type === "response") {
      waiting.get(data.id)?.({ status: data.status, contentType: data.contentType, body: data.body });
      waiting.delete(data.id);
    } else if (data.type === "progress") {
      listeners.forEach((l) => l({ phase: data.phase, done: data.done, total: data.total }));
    } else if (data.type === "ready") {
      readyResolve();
      listeners.forEach((l) => l({ phase: "ready" }));
    } else if (data.type === "idle") {
      listeners.forEach((l) => l({ phase: "idle" }));
    } else if (data.type === "fatal") {
      readyReject(new Error(data.message));
      listeners.forEach((l) => l({ phase: "fatal", message: data.message }));
    }
  };
  worker.onerror = (event) => {
    readyReject(new Error(event.message || "the engine worker failed to start"));
  };
  return worker;
}

/** Start downloading the runtime as early as possible. */
export function bootEngine(): void {
  if (IN_BROWSER) start();
}

export function onEngineEvent(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Send one request to the in-browser engine. `path` may carry a query string. */
export function engineRequest(method: string, path: string, body?: BodyInit | null): Promise<EngineResponse> {
  const w = start();
  const url = new URL(path, "http://engine.local");
  const query: Record<string, string> = {};
  url.searchParams.forEach((value, key) => {
    query[key] = value;
  });
  const id = nextId++;
  return new Promise((resolve) => {
    waiting.set(id, resolve);
    w.postMessage({
      id,
      method,
      path: url.pathname,
      query,
      body: typeof body === "string" ? body : null,
    });
  });
}

export function decodeJson<T>(response: EngineResponse): T {
  return JSON.parse(new TextDecoder().decode(response.body)) as T;
}

/**
 * An `<img src>` for an API image. On the server build this is the URL itself;
 * in the browser build the bytes come from the engine and are shown through an
 * object URL, which is revoked when the next one replaces it.
 */
export function useApiImage(url: string | null): string | undefined {
  const [src, setSrc] = useState<string | undefined>(IN_BROWSER ? undefined : url ?? undefined);
  const current = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (!IN_BROWSER) {
      setSrc(url ?? undefined);
      return undefined;
    }
    if (!url) return undefined;
    let live = true;
    void engineRequest("GET", url).then((response) => {
      if (!live || response.status >= 400) return;
      const next = URL.createObjectURL(new Blob([response.body], { type: response.contentType }));
      if (current.current) URL.revokeObjectURL(current.current);
      current.current = next;
      setSrc(next);
    });
    return () => {
      live = false;
    };
  }, [url]);

  useEffect(
    () => () => {
      if (current.current) URL.revokeObjectURL(current.current);
    },
    [],
  );

  return src;
}

/** Fetch an image through the engine once and return an object URL for it. */
export async function engineImage(url: string): Promise<string | null> {
  const response = await engineRequest("GET", url);
  if (response.status >= 400) return null;
  return URL.createObjectURL(new Blob([response.body], { type: response.contentType }));
}

/**
 * Save an API response as a file. A plain `<a download href>` works against the
 * server; in the browser build there is no URL to point at, so the bytes are
 * fetched from the engine and saved through a temporary object URL.
 */
export async function downloadFromApi(path: string, filename: string): Promise<void> {
  let href = path;
  let revoke = false;
  if (IN_BROWSER) {
    const response = await engineRequest("GET", path);
    if (response.status >= 400) {
      throw new Error(decodeJson<{ detail?: string }>(response).detail ?? `download failed (${response.status})`);
    }
    href = URL.createObjectURL(new Blob([response.body], { type: response.contentType }));
    revoke = true;
  }
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  if (revoke) window.setTimeout(() => URL.revokeObjectURL(href), 10_000);
}

/**
 * Resolution and length the browser build uses. Measured, not guessed: the
 * browser runs the pipeline about 4.5x slower than native, and at 480x270 for
 * four seconds it counted no vehicles at all - they are too small at the
 * counting line to be confirmed as tracks. 560x315 for five seconds counts five
 * and analyses in about a minute on a laptop, which beat 640x360 on both.
 */
export const BROWSER_CLIP = { width_px: 560, height_px: 315, duration_s: 5 };
