/**
 * The pipeline inspector: every stage's actual output, side by side.
 *
 * This is the view that makes the rest of the project checkable. A claim like
 * "shadow suppression is what splits those two vehicles" is either visible here
 * between S4 and S5 or it is not true.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type StageFrame } from "../lib/api";
import { ms } from "../lib/format";
import { useStore } from "../lib/store";
import { Empty, ErrorNote, Panel, UnitChips } from "../components/primitives";

export function PipelineView({ loaded }: { loaded: boolean }) {
  const telemetry = useStore((s) => s.telemetry);
  const [frozen, setFrozen] = useState<number | null>(null);
  const index = frozen ?? telemetry?.frame_index ?? 0;

  const stages = useQuery({
    queryKey: ["frame-stages", index],
    queryFn: () => api.frameStages(index, true),
    enabled: loaded,
  });

  if (!loaded) return <Empty message="Load a source to inspect the pipeline." />;

  const total = stages.data?.stages.reduce((a, s) => a + s.elapsed_ms, 0) ?? 0;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex shrink-0 items-center gap-3 rounded-panel border border-subtle bg-surface px-3 py-2">
        <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
          Frame
        </span>
        <span className="num text-[13px] text-primary">{index}</span>
        <button
          type="button"
          onClick={() => setFrozen(frozen == null ? index : null)}
          className={`rounded-ctl border px-2 py-0.5 text-[11.5px] ${
            frozen == null
              ? "border-line bg-raised text-muted"
              : "border-accent bg-accent-dim text-accent"
          }`}
        >
          {frozen == null ? "Follow playback" : "Frozen"}
        </button>
        <div className="flex-1" />
        <span className="num text-[11.5px] text-muted">
          {ms(total)} ms total · {total > 0 ? (1000 / total).toFixed(1) : "—"} fps
        </span>
      </div>

      {stages.isError ? (
        <ErrorNote message={String(stages.error)} />
      ) : (
        <div className="min-h-0 flex-1 overflow-auto">
          <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
            {stages.data?.stages.map((stage) => (
              <StageCard key={stage.stage_id} stage={stage} total={total} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StageCard({ stage, total }: { stage: StageFrame; total: number }) {
  const share = total > 0 ? stage.elapsed_ms / total : 0;
  const metrics = Object.entries(stage.metrics);

  return (
    <Panel
      title={stage.name}
      badge={<UnitChips units={stage.units} />}
      actions={
        <span
          className="num text-[11px] text-muted"
          title={`${(share * 100).toFixed(0)}% of this frame's pipeline time`}
        >
          {ms(stage.elapsed_ms)} ms
        </span>
      }
    >
      {stage.image_png ? (
        <img
          src={`data:image/png;base64,${stage.image_png}`}
          alt={`${stage.name} output`}
          className="mb-2 w-full rounded-panel border border-subtle bg-black object-contain"
        />
      ) : (
        <div className="mb-2 rounded-panel border border-dashed border-line px-3 py-4 text-[11.5px] text-muted">
          This stage has no image output — it produces numbers, not pixels.
        </div>
      )}

      {/* A share bar rather than a chart: the only question here is "which stage
          is eating the frame", and one bar answers it without a legend. */}
      <div className="mb-2 h-[3px] w-full overflow-hidden rounded bg-raised">
        <div
          className={share > 0.35 ? "h-full bg-stop" : "h-full bg-accent"}
          style={{ width: `${Math.min(100, share * 100)}%` }}
        />
      </div>

      <p className="mb-2 text-[11.5px] leading-snug text-muted">{stage.rationale}</p>

      {metrics.length ? (
        <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11.5px]">
          {metrics.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2">
              <dt className="truncate text-muted" title={k}>
                {k}
              </dt>
              <dd className="num m-0 text-primary">
                {Number.isInteger(v) ? v : v.toFixed(3)}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}

      {stage.modified.length ? (
        <p className="mt-2 text-[11px] text-accent">
          {stage.modified.length} parameter{stage.modified.length === 1 ? "" : "s"} changed
          from default: {stage.modified.join(", ")}
        </p>
      ) : null}
    </Panel>
  );
}
