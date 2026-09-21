/**
 * The live console: the view a person opens first.
 *
 * If no source is loaded it does not show an empty video box with a spinner —
 * it shows the eight scenarios and what each one is for, because "which failure
 * mode do you want to look at" is the actual first question.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Status } from "../lib/api";
import { BROWSER_CLIP, IN_BROWSER } from "../lib/engine";
import { Button, Empty, ErrorNote, Panel, Stat } from "../components/primitives";
import { Viewport } from "../components/Viewport";
import { useStore } from "../lib/store";

export function ConsoleView({ loaded, status }: { loaded: boolean; status: Status | null }) {
  if (!loaded) return <SourcePicker />;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <Viewport frames={status?.state.frame_count ?? 0} />
      <CountStrip status={status} />
    </div>
  );
}

function CountStrip({ status }: { status: Status | null }) {
  const telemetry = useStore((s) => s.telemetry);
  const byLane = telemetry?.counts ?? status?.counts_by_lane ?? {};
  const lanes = Object.entries(byLane);

  return (
    <div className="flex shrink-0 gap-2 overflow-x-auto">
      <Stat
        label="Total counted"
        value={String(telemetry?.total_count ?? status?.total_count ?? 0)}
        tone="accent"
      />
      {lanes.map(([lane, count]) => (
        <Stat key={lane} label={`Lane ${lane}`} value={String(count)} />
      ))}
      {lanes.length === 0 ? (
        <div className="flex items-center px-2 text-[12px] text-muted">
          No lane counts yet — lanes come from the calibration.
        </div>
      ) : null}
    </div>
  );
}

function SourcePicker() {
  const client = useQueryClient();
  const scenarios = useQuery({ queryKey: ["scenarios"], queryFn: api.scenarios });

  const load = useMutation({
    mutationFn: (name: string) =>
      api.loadScenario(
        IN_BROWSER ? { name, seed: 7, ...BROWSER_CLIP } : { name, duration_s: 12.0, seed: 7 },
      ),
    onSuccess: () => {
      void client.invalidateQueries();
    },
  });

  const analyse = useMutation({
    mutationFn: () => api.analyse(false),
    onSuccess: () => void client.invalidateQueries(),
  });

  const [videoPath, setVideoPath] = useState("");
  const loadVideo = useMutation({
    mutationFn: () => api.loadVideo(videoPath.trim()),
    onSuccess: () => void client.invalidateQueries(),
  });

  return (
    <div className="h-full overflow-auto">
      <Panel title="Choose a source">
        <p className="mb-4 max-w-prose text-[13px] text-secondary">
          Each scenario is rendered on demand with exact ground truth, so every
          number the console reports can be scored rather than eyeballed. Each one
          isolates a different failure mode.
        </p>

        {scenarios.isError ? (
          <ErrorNote message={String(scenarios.error)} />
        ) : scenarios.isLoading ? (
          <p className="text-[13px] text-muted">Loading scenarios…</p>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {scenarios.data?.scenarios.map((s) => (
              <button
                key={s.name}
                type="button"
                disabled={load.isPending}
                onClick={() => load.mutate(s.name)}
                className="rounded-panel border border-subtle bg-raised p-3 text-left transition-colors duration-fast hover:border-accent disabled:opacity-50"
              >
                <div className="num mb-1 text-[12.5px] font-semibold text-accent">{s.name}</div>
                <div className="mb-1.5 text-[13px] font-medium text-primary">{s.label}</div>
                <p className="m-0 text-[11.5px] leading-snug text-muted">{s.description}</p>
                <p className="mt-1.5 text-[11px] leading-snug text-secondary">
                  <span className="text-muted">Exercises: </span>
                  {s.exercises}
                </p>
              </button>
            ))}
          </div>
        )}

        {load.isPending ? (
          <p className="mt-4 text-[13px] text-accent">
            Rendering the clip… this takes a few seconds, because the
            renderer is producing ground truth as well as pixels.
          </p>
        ) : null}
        {load.isError ? <div className="mt-4"><ErrorNote message={String(load.error)} /></div> : null}
      </Panel>

      {IN_BROWSER ? (
        <div className="mt-3">
          <Panel title="Running in your browser">
            <p className="max-w-prose text-[13px] leading-relaxed text-secondary">
              This copy of Kerbside has no server behind it. The Python pipeline is
              running inside this page, on this device, under WebAssembly &mdash; which
              is why scenarios render at {BROWSER_CLIP.width_px}&times;{BROWSER_CLIP.height_px}{" "}
              for {BROWSER_CLIP.duration_s} seconds rather than at 720p. Opening a real
              video file needs the local version, which reads from disk.
            </p>
          </Panel>
        </div>
      ) : (
      <div className="mt-3">
        <Panel title="Or a real clip">
          <p className="mb-3 max-w-prose text-[13px] leading-relaxed text-secondary">
            Any video a codec on this machine can decode. Real footage has no ground
            truth, so the console will show counts, speeds and classes but will not
            score them — and a synthetic result and a real one are never averaged
            together anywhere in this system.
          </p>
          <div className="flex items-end gap-2">
            <label className="flex-1">
              <span className="mb-1 block text-[12px] font-medium text-secondary">
                Path to a video file
              </span>
              <input
                value={videoPath}
                placeholder="D:\footage\junction.mp4"
                onChange={(e) => setVideoPath(e.target.value)}
                className="num h-8 w-full rounded-ctl border border-line bg-raised px-2 text-[13px] text-primary placeholder:text-muted"
              />
            </label>
            <Button
              onClick={() => loadVideo.mutate()}
              disabled={loadVideo.isPending || !videoPath.trim()}
            >
              {loadVideo.isPending ? "Opening…" : "Open"}
            </Button>
          </div>
          {loadVideo.isError ? (
            <div className="mt-3">
              <ErrorNote message={String(loadVideo.error)} />
            </div>
          ) : null}
        </Panel>
      </div>
      )}

      <div className="mt-3">
        <Empty
          message="Already loaded a clip and want the whole thing measured? Analysing runs all nine stages over every frame and fills in the counts, speeds and analytics."
          action={
            <Button variant="primary" onClick={() => analyse.mutate()} disabled={analyse.isPending}>
              {analyse.isPending ? "Analysing…" : "Analyse all frames"}
            </Button>
          }
        />
      </div>
    </div>
  );
}
