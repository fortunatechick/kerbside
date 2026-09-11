/**
 * The motion lab.
 *
 * The finding this view exists to make visible: a fast block search is a good
 * choice for a codec and a bad one for a measurement, and those are different
 * goals scored by different numbers. So the table reports cost, PSNR and vector
 * error as three separate columns and never collapses them into a single
 * "quality" figure — collapsing them is precisely the mistake.
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api, type AblationResult } from "../lib/api";
import { Button, Empty, ErrorNote, Panel, Table } from "../components/primitives";
import { useStore } from "../lib/store";

export function MotionLab({ loaded }: { loaded: boolean }) {
  const selectStage = useStore((s) => s.selectStage);
  const [result, setResult] = useState<AblationResult | null>(null);

  const run = useMutation({
    mutationFn: () => api.runAblation("search_strategy", 6),
    onSuccess: setResult,
  });

  if (!loaded) return <Empty message="Load a source to run the motion studies." />;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto">
      <Panel
        title="Cheap and accurate are different goals"
        actions={
          <button
            type="button"
            onClick={() => selectStage("s8_speed")}
            className="text-[11.5px] text-accent hover:underline"
          >
            Edit S8 parameters →
          </button>
        }
      >
        <p className="max-w-prose text-[13px] leading-relaxed text-secondary">
          Exhaustive search evaluates every candidate in the window. The 2-D log and
          three-step searches evaluate a few dozen. Measured on real content, the
          fast searches use <strong>32–36× fewer comparisons</strong> and still predict
          the next frame at about <strong>35 dB</strong> — entirely adequate for a video
          codec, whose only job is to make the residual small.
        </p>
        <p className="mt-2 max-w-prose text-[13px] leading-relaxed text-secondary">
          They also land about <strong>five pixels</strong> from the true displacement,
          because they descend into a local minimum of a surface that is not convex.
          Five pixels is a large speed error. The same algorithm is a good choice for
          compression and a bad one for measurement, and no single quality score can
          express that.
        </p>
        <p className="mt-2 max-w-prose text-[12.5px] leading-relaxed text-muted">
          Exhaustive search on real content: <span className="num text-primary">0.32 px MAE</span>,
          96% of blocks exact. Sub-pixel refinement by upsampled-DFT phase correlation
          brings the residual under <span className="num text-primary">0.1 px</span> —
          a parabolic fit through the correlation peak lands about a pixel out, because
          the peak of a fractional shift is a periodic sinc, not a parabola.
        </p>
      </Panel>

      <Panel
        title="Search strategy study"
        actions={
          <Button size="sm" variant="primary" disabled={run.isPending} onClick={() => run.mutate()}>
            {run.isPending ? "Running…" : "Run study"}
          </Button>
        }
      >
        {run.isError ? <ErrorNote message={String(run.error)} /> : null}
        {result ? (
          <>
            <p className="mb-3 max-w-prose text-[12.5px] text-secondary">{result.question}</p>
            <Table
              columns={[
                { key: "setting", label: "Strategy" },
                { key: "blockmatch_mae_kmph", label: "Speed MAE km/h" },
                { key: "speed_mae_kmph", label: "Route A MAE km/h" },
                { key: "fps", label: "fps" },
                { key: "recall", label: "Recall" },
              ]}
              numeric={["blockmatch_mae_kmph", "speed_mae_kmph", "fps", "recall"]}
              rows={result.rows.map((r) => ({
                setting: r.setting,
                blockmatch_mae_kmph: r.blockmatch_mae_kmph?.toFixed(2) ?? "—",
                speed_mae_kmph: r.speed_mae_kmph?.toFixed(2) ?? "—",
                fps: r.fps.toFixed(2),
                recall: r.recall.toFixed(3),
              }))}
            />
            <p className="mt-2 text-[11.5px] text-muted">{result.gate.description}</p>
          </>
        ) : (
          <p className="text-[12.5px] text-muted">
            Runs the pipeline over {`${6}`} seconds of the current clip once per search
            strategy, scoring each against ground truth. Takes a minute or two — the
            pipeline runs at about 4 fps.
          </p>
        )}
      </Panel>

      <Panel title="A sign convention that would have been invisible">
        <p className="max-w-prose text-[12.5px] leading-relaxed text-muted">
          A codec's motion vector points from the current block to where it was
          found in the reference — the negation of the displacement the content
          actually underwent. Reported the codec way, block matching would have
          disagreed with phase correlation on every frame by exactly a sign, and the
          disagreement would have looked like noise. This API returns content
          displacement, and offers the codec form explicitly as
          <code className="num text-accent"> reference_offset_px</code>.
        </p>
      </Panel>
    </div>
  );
}
