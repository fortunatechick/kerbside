/**
 * Startup for the browser build.
 *
 * In the browser build there is no server: the first visit downloads Python
 * (about 35 MB, cached afterwards), loads the pipeline, renders a traffic scene
 * and analyses it - all on the visitor's device. That takes long enough that a
 * blank page would look broken, so each step is named and measured as it
 * happens. Once it finishes, the console opens on a clip that is already
 * analysed, which is the state a demonstration needs to start from.
 *
 * On the server build this renders its children and does nothing else.
 */

import { useEffect, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { BROWSER_CLIP, engineReady, IN_BROWSER, onEngineEvent, type EngineProgress } from "../lib/engine";

const STEPS: { phase: string; label: string }[] = [
  { phase: "runtime", label: "Download the Python runtime (WebAssembly)" },
  { phase: "packages", label: "Load NumPy, SciPy and OpenCV" },
  { phase: "source", label: "Load the Kerbside pipeline" },
  { phase: "render", label: "Render a traffic scene with exact ground truth" },
  { phase: "analyse", label: "Run the nine-stage pipeline on every frame" },
];

export function EngineBoot({ children }: { children: ReactNode }) {
  const client = useQueryClient();
  const [booted, setBooted] = useState(!IN_BROWSER);
  const [progress, setProgress] = useState<Record<string, EngineProgress>>({});
  const [error, setError] = useState<string | null>(null);
  const [started] = useState(() => performance.now());
  const [now, setNow] = useState(() => performance.now());

  useEffect(() => {
    if (!IN_BROWSER) return undefined;
    const off = onEngineEvent((event) => {
      if ("done" in event) setProgress((p) => ({ ...p, [event.phase]: event }));
      if (event.phase === "fatal") {
        setError(("message" in event && event.message) || "the engine failed to start");
      }
    });
    const tick = window.setInterval(() => setNow(performance.now()), 250);

    let cancelled = false;
    (async () => {
      try {
        await engineReady;
        await api.loadScenario({ name: "free_flow", seed: 4242, ...BROWSER_CLIP });
        await api.analyse(false);
        if (cancelled) return;
        await client.invalidateQueries();
        setBooted(true);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
      off();
      window.clearInterval(tick);
    };
  }, [client]);

  if (booted) return <>{children}<EngineBusy /></>;

  const activeIndex = STEPS.findIndex((s) => {
    const p = progress[s.phase];
    return !p || p.done < p.total - (s.phase === "render" || s.phase === "analyse" ? 1 : 0);
  });

  return (
    <div className="flex h-full items-center justify-center overflow-auto bg-base p-6 text-primary">
      <div className="w-full max-w-[560px]">
        <p className="num mb-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-accent">
          Kerbside · starting up
        </p>
        <h1 className="mb-2 text-[28px] font-semibold tracking-tight">
          Loading the traffic analytics pipeline
        </h1>
        <p className="mb-6 text-[14px] leading-relaxed text-secondary">
          There is no server behind this page. The real Python pipeline is being
          loaded into your browser and will run on this device, so nothing you do
          here is sent anywhere. The first visit downloads about 35&nbsp;MB; after
          that it is cached.
        </p>

        <ol className="space-y-3">
          {STEPS.map((step, i) => {
            const p = progress[step.phase];
            const done = activeIndex === -1 || i < activeIndex;
            const active = i === activeIndex;
            const fraction = p && p.total > 0 ? Math.min(1, (p.done + (done ? 1 : 0)) / p.total) : 0;
            return (
              <li key={step.phase}>
                <div className="flex items-baseline justify-between gap-3">
                  <span
                    className={`text-[13.5px] ${
                      done ? "text-free" : active ? "font-medium text-primary" : "text-muted"
                    }`}
                  >
                    {done ? "✓ " : active ? "● " : "○ "}
                    {step.label}
                  </span>
                  {active && p && p.total > 1 ? (
                    <span className="num text-[11.5px] text-muted">
                      {Math.min(p.done + 1, p.total)} / {p.total}
                    </span>
                  ) : null}
                </div>
                {active ? (
                  <div className="mt-1.5 h-[3px] w-full overflow-hidden rounded bg-raised">
                    <div
                      className="h-full bg-accent transition-[width] duration-fast"
                      style={{ width: `${Math.max(4, fraction * 100)}%` }}
                    />
                  </div>
                ) : null}
              </li>
            );
          })}
        </ol>

        <p className="num mt-6 text-[11.5px] text-muted">
          {((now - started) / 1000).toFixed(0)} s elapsed
        </p>

        <p className="mt-4 text-[13px] text-secondary">
          While it loads:{" "}
          <a href="docs/" target="_blank" rel="noopener" className="text-accent hover:underline">
            read the one-page project overview
          </a>{" "}
          in a new tab. This one keeps loading; leaving it would start the download again.
        </p>

        {error ? (
          <div className="mt-4 rounded-panel border border-stop/40 bg-stop/10 px-4 py-3 text-[13px] leading-relaxed text-stop">
            <strong>The pipeline could not start in this browser.</strong> {error}
            <br />
            It needs a recent browser with WebAssembly and about 1&nbsp;GB of free memory.
            Reloading usually fixes a failed download.
          </div>
        ) : null}
      </div>
    </div>
  );
}

/**
 * After startup, choosing another scenario or re-analysing runs the same
 * render-and-analyse work. This shows it, so a busy engine never looks frozen.
 */
function EngineBusy() {
  const [state, setState] = useState<EngineProgress | null>(null);

  useEffect(() => {
    if (!IN_BROWSER) return undefined;
    return onEngineEvent((event) => {
      if (event.phase === "idle") setState(null);
      else if ("done" in event && (event.phase === "render" || event.phase === "analyse")) {
        setState(event);
      }
    });
  }, []);

  if (!state) return null;
  const label = state.phase === "render" ? "Rendering the scene" : "Running the pipeline";
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-10 z-50 flex justify-center px-4">
      <div className="pointer-events-auto w-full max-w-[360px] rounded-panel border border-line bg-surface px-4 py-3 shadow-overlay">
        <div className="mb-1.5 flex items-baseline justify-between text-[12.5px]">
          <span className="font-medium text-primary">{label} in your browser</span>
          <span className="num text-muted">
            {Math.min(state.done + 1, state.total)} / {state.total}
          </span>
        </div>
        <div className="h-[3px] overflow-hidden rounded bg-raised">
          <div
            className="h-full bg-accent"
            style={{ width: `${((state.done + 1) / Math.max(1, state.total)) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
