/**
 * The application shell.
 *
 * Three regions and a status bar, fixed: a left rail that names the views and
 * carries the stage parameters, a centre that belongs to whatever view is
 * selected, and a right inspector that always answers "what does the pipeline
 * believe right now". The bottom console is a drawer because a log that is
 * always open trains people to stop reading it.
 *
 * The layout never reflows when a view changes. A console whose controls move
 * between screens is a console the user has to re-learn on every tab.
 */

import { useCallback, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError, type Status } from "./lib/api";
import { seconds } from "./lib/format";
import { useStore, type ViewId } from "./lib/store";
import { useTelemetry } from "./lib/ws";
import { Button, ErrorNote } from "./components/primitives";
import { ParamRail } from "./components/ParamRail";
import { Inspector } from "./components/Inspector";

import { ConsoleView } from "./views/ConsoleView";
import { CalibrationView } from "./views/CalibrationView";
import { PipelineView } from "./views/PipelineView";
import { OperatorsView } from "./views/OperatorsView";
import { ShadowLab } from "./views/ShadowLab";
import { MotionLab } from "./views/MotionLab";
import { AnalyticsView } from "./views/AnalyticsView";
import { AblationLab } from "./views/AblationLab";
import { ReportView } from "./views/ReportView";

const VIEWS: { id: ViewId; label: string; glyph: string; hint: string }[] = [
  { id: "console", label: "Console", glyph: "▶", hint: "Live video, overlays, counts" },
  { id: "calibration", label: "Calibration", glyph: "⊹", hint: "Four points, then metres" },
  { id: "pipeline", label: "Pipeline", glyph: "⧉", hint: "Every stage's output" },
  { id: "operators", label: "Operators", glyph: "⌗", hint: "All 35, by syllabus unit" },
  { id: "shadow", label: "Shadow lab", glyph: "◑", hint: "The HSV test, decomposed" },
  { id: "motion", label: "Motion lab", glyph: "⇄", hint: "Block matching strategies" },
  { id: "analytics", label: "Analytics", glyph: "⌸", hint: "Flow, speeds, headways" },
  { id: "ablation", label: "Ablations", glyph: "⚖", hint: "Nine studies" },
  { id: "report", label: "Report", glyph: "⎙", hint: "One self-contained file" },
];

export default function App() {
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const railCollapsed = useStore((s) => s.railCollapsed);
  const toggleRail = useStore((s) => s.toggleRail);
  const consoleOpen = useStore((s) => s.consoleOpen);
  const toggleConsole = useStore((s) => s.toggleConsole);
  const stages = useStore((s) => s.stages);
  const setStages = useStore((s) => s.setStages);
  const setTelemetry = useStore((s) => s.setTelemetry);
  const pushLog = useStore((s) => s.pushLog);

  const [theme, setTheme] = useState<"dark" | "light">("dark");
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useTelemetry(useCallback((t) => setTelemetry(t), [setTelemetry]));

  const status = useQuery({
    queryKey: ["status"],
    queryFn: api.status,
    refetchInterval: 4000,
  });

  const pipeline = useQuery({ queryKey: ["pipeline"], queryFn: api.pipeline });
  useEffect(() => {
    if (pipeline.data) setStages(pipeline.data.stages);
  }, [pipeline.data, setStages]);

  // The server keeps its own log of what each stage did; mirroring it into the
  // console means the drawer shows pipeline events, not just UI events. The
  // server's log has no cursor, so the last timestamp seen is what stops the
  // same line being appended on every poll.
  useEffect(() => {
    let stop = false;
    let seen = 0;
    const poll = async () => {
      try {
        const { entries } = await api.log(60);
        if (stop) return;
        for (const entry of entries) {
          if (entry.t > seen) {
            pushLog({ t: entry.t * 1000, source: entry.source, message: entry.message });
          }
        }
        const newest = entries[entries.length - 1];
        if (newest) seen = Math.max(seen, newest.t);
      } catch {
        /* the log is a convenience; its absence must not raise an error banner */
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 3000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, [pushLog]);

  const loaded = Boolean(status.data?.state.frame_count);

  return (
    <div className="flex h-full flex-col bg-base text-primary">
      <div className="flex min-h-0 flex-1">
        {/* ------------------------------------------------- left rail -- */}
        <aside
          className={`flex shrink-0 flex-col border-r border-subtle bg-surface transition-[width] duration-fast ${
            railCollapsed ? "w-[48px]" : "w-[216px]"
          }`}
        >
          <div className="flex h-[42px] shrink-0 items-center gap-2 border-b border-subtle px-2">
            {railCollapsed ? null : (
              <span className="flex-1 truncate text-[13px] font-semibold tracking-tight">
                Kerbside
              </span>
            )}
            <button
              type="button"
              onClick={toggleRail}
              aria-label={railCollapsed ? "Expand rail" : "Collapse rail"}
              className="rounded-ctl px-1.5 py-1 text-muted hover:bg-raised hover:text-primary"
            >
              {railCollapsed ? "»" : "«"}
            </button>
          </div>

          <nav className="shrink-0 border-b border-subtle p-1.5">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                type="button"
                onClick={() => setView(v.id)}
                title={railCollapsed ? `${v.label} — ${v.hint}` : v.hint}
                className={`mb-0.5 flex w-full items-center gap-2 rounded-ctl px-2 py-1.5 text-left text-[12.5px] transition-colors duration-fast ${
                  view === v.id
                    ? "bg-accent-dim font-medium text-accent"
                    : "text-secondary hover:bg-raised hover:text-primary"
                }`}
              >
                <span className="w-4 shrink-0 text-center text-[13px]">{v.glyph}</span>
                {railCollapsed ? null : <span className="truncate">{v.label}</span>}
              </button>
            ))}
          </nav>

          {railCollapsed ? null : (
            <div className="min-h-0 flex-1 overflow-hidden">
              <ParamRail stages={stages} />
            </div>
          )}
        </aside>

        {/* ---------------------------------------------------- centre -- */}
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-hidden p-3">
            {status.isError ? (
              <ErrorNote message={String((status.error as ApiError)?.message ?? status.error)} />
            ) : null}
            <ViewSwitch view={view} loaded={loaded} status={status.data ?? null} />
          </div>

          {/* --------------------------------------- bottom console -- */}
          <BottomConsole open={consoleOpen} onToggle={toggleConsole} />
        </main>

        {/* ------------------------------------------------- inspector -- */}
        <aside className="hidden w-[300px] shrink-0 flex-col overflow-hidden border-l border-subtle bg-surface xl:flex">
          <div className="flex h-[42px] shrink-0 items-center border-b border-subtle px-3">
            <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">
              Inspector
            </span>
          </div>
          <div className="min-h-0 flex-1 overflow-hidden">
            <Inspector />
          </div>
        </aside>
      </div>

      <StatusBar
        status={status.data ?? null}
        theme={theme}
        onTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
      />
    </div>
  );
}

function ViewSwitch({
  view,
  loaded,
  status,
}: {
  view: ViewId;
  loaded: boolean;
  status: Status | null;
}) {
  switch (view) {
    case "console":
      return <ConsoleView loaded={loaded} status={status} />;
    case "calibration":
      return <CalibrationView loaded={loaded} />;
    case "pipeline":
      return <PipelineView loaded={loaded} />;
    case "operators":
      // The registry is static, so this view needs no source loaded - which is
      // also why it is the one thing worth reading before picking a clip.
      return <OperatorsView />;
    case "shadow":
      return <ShadowLab loaded={loaded} />;
    case "motion":
      return <MotionLab loaded={loaded} />;
    case "analytics":
      return <AnalyticsView loaded={loaded} />;
    case "ablation":
      return <AblationLab />;
    case "report":
      return <ReportView loaded={loaded} />;
    default:
      return null;
  }
}

function BottomConsole({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const log = useStore((s) => s.log);
  const paused = useStore((s) => s.logPaused);
  const setPaused = useStore((s) => s.setLogPaused);

  return (
    <section
      className={`shrink-0 border-t border-subtle bg-surface transition-[height] duration-fast ${
        open ? "h-[160px]" : "h-[32px]"
      }`}
    >
      <header className="flex h-[32px] items-center gap-2 px-3">
        <button
          type="button"
          onClick={onToggle}
          className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted hover:text-primary"
        >
          {open ? "▾" : "▸"} Console
        </button>
        <span className="num text-[11px] text-muted">{log.length}</span>
        <div className="flex-1" />
        {open ? (
          <Button size="sm" variant="ghost" onClick={() => setPaused(!paused)}>
            {paused ? "Resume" : "Pause"}
          </Button>
        ) : null}
      </header>
      {open ? (
        <div className="h-[128px] overflow-auto px-3 pb-2">
          {log.length === 0 ? (
            <p className="text-[11.5px] text-muted">
              Pipeline events appear here — stage errors, parameter changes, and the
              reason a frame produced no detections.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {log.slice(-120).map((e, i) => (
                <li key={i} className="num flex gap-2 text-[11px] leading-relaxed">
                  <span className="shrink-0 text-muted">
                    {new Date(e.t).toLocaleTimeString([], { hour12: false })}
                  </span>
                  <span className="shrink-0 text-accent">{e.source}</span>
                  <span className="min-w-0 text-secondary">{e.message}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : null}
    </section>
  );
}

function StatusBar({
  status,
  theme,
  onTheme,
}: {
  status: Status | null;
  theme: string;
  onTheme: () => void;
}) {
  const telemetry = useStore((s) => s.telemetry);
  const state = status?.state;
  const slowest = status?.slowest_stage;

  const cell = (label: string, value: string, hint?: string) => (
    <span className="flex items-baseline gap-1.5" title={hint}>
      <span className="text-[10px] uppercase tracking-[0.06em] text-muted">{label}</span>
      <span className="num text-[11.5px] text-secondary">{value}</span>
    </span>
  );

  return (
    <footer className="flex h-[26px] shrink-0 items-center gap-4 overflow-x-auto border-t border-subtle bg-surface px-3">
      {cell("source", state?.source_name ?? "none")}
      {state
        ? cell(
            "fps",
            `${state.measured_fps.toFixed(3)} measured / ${state.container_fps.toFixed(3)} declared`,
            "The declared rate is never used for timing. If these differ, every speed computed from the declared rate would carry that error.",
          )
        : null}
      {state ? cell("size", `${state.width_px}×${state.height_px}`) : null}
      {telemetry ? cell("t", seconds(telemetry.time_s)) : null}
      {telemetry ? cell("frame", `${telemetry.frame_ms.toFixed(0)} ms`) : null}
      {slowest ? cell("slowest", `${slowest[0]} ${slowest[1].toFixed(1)} ms`) : null}
      {state && !state.calibrated ? (
        <span className="rounded px-1.5 text-[11px] text-slow" title="Speeds and sizes are not physical until a calibration exists.">
          uncalibrated
        </span>
      ) : null}
      <div className="flex-1" />
      <button
        type="button"
        onClick={onTheme}
        title="Toggle theme"
        className="text-[11px] text-muted hover:text-primary"
      >
        {theme === "dark" ? "☾" : "☀"}
      </button>
      <span className="text-[10.5px] text-muted">measurement only — not an enforcement device</span>
    </footer>
  );
}
