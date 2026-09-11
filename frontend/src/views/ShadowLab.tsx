/**
 * The shadow lab.
 *
 * The HSV shadow test is three conditions ANDed together, and the interesting
 * question is never "does it work" but "which condition is doing the work, and
 * which one is throwing away correct pixels". So the lab shows the stage's own
 * per-condition metrics next to the controls that set them, and states the one
 * fact that makes the hue condition behave counter-intuitively.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { pct } from "../lib/format";
import { useStore } from "../lib/store";
import { Empty, ErrorNote, Panel, Stat } from "../components/primitives";

export function ShadowLab({ loaded }: { loaded: boolean }) {
  const telemetry = useStore((s) => s.telemetry);
  const selectStage = useStore((s) => s.selectStage);
  const index = telemetry?.frame_index ?? 0;

  const stages = useQuery({
    queryKey: ["frame-stages", index],
    queryFn: () => api.frameStages(index, true),
    enabled: loaded,
  });

  if (!loaded) return <Empty message="Load a source — ideally the long_shadows scenario." />;

  const shadow = stages.data?.stages.find((s) => s.stage_id.startsWith("s5"));
  const foreground = stages.data?.stages.find((s) => s.stage_id.startsWith("s4"));
  const m = shadow?.metrics ?? {};

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto">
      <Panel
        title="Why this stage exists"
        actions={
          <button
            type="button"
            onClick={() => selectStage("s5_shadow")}
            className="text-[11.5px] text-accent hover:underline"
          >
            Edit its parameters →
          </button>
        }
      >
        <p className="max-w-prose text-[13px] leading-relaxed text-secondary">
          A cast shadow moves with the vehicle, so every motion-based test calls it
          foreground. Left in, it inflates the footprint — which is what the class
          decision is made from — and it <strong>bridges adjacent vehicles into one
          blob</strong>, which is what the count is made from. Removing it is not
          cosmetic.
        </p>
        <p className="mt-2 max-w-prose text-[12.5px] leading-relaxed text-muted">
          The test: a shadowed pixel keeps its hue, loses value by a bounded ratio,
          and changes saturation only slightly. The trap is the hue condition —
          hue is the arctangent of two colour differences, so where saturation is
          near zero it is undefined. Asphalt is close to neutral. Applying the
          textbook 10-unit hue tolerance on a background at saturation 0.075
          rejected <strong>half of a correctly identified shadow</strong>, which is why
          the hue test here is gated behind a saturation floor.
        </p>
      </Panel>

      {stages.isError ? <ErrorNote message={String(stages.error)} /> : null}

      <div className="grid shrink-0 grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="Shadow pixels"
          value={String(m.shadow_px ?? 0)}
          tone="accent"
          hint="Pixels S5 removed from the S4 mask on this frame."
        />
        <Stat
          label="Share of foreground"
          value={pct(
            foreground?.metrics.foreground_px
              ? (m.shadow_px ?? 0) / foreground.metrics.foreground_px
              : null,
            1,
          )}
        />
        <Stat label="Hue-gated pixels" value={String(m.hue_gated_px ?? 0)}
              hint="Pixels where saturation was too low for the hue test to mean anything, so it was skipped." />
        <Stat label="Stage time" value={(shadow?.elapsed_ms ?? 0).toFixed(1)} unit="ms" />
      </div>

      <div className="grid min-h-0 gap-3 lg:grid-cols-2">
        <Panel title="Mask before S5 — foreground incl. shadow">
          {foreground?.image_png ? (
            <img
              src={`data:image/png;base64,${foreground.image_png}`}
              alt="Foreground mask"
              className="w-full rounded-panel border border-subtle bg-black"
            />
          ) : (
            <p className="text-[12px] text-muted">No image for this stage on this frame.</p>
          )}
        </Panel>
        <Panel title="After S5 — shadow removed">
          {shadow?.image_png ? (
            <img
              src={`data:image/png;base64,${shadow.image_png}`}
              alt="Mask after shadow suppression"
              className="w-full rounded-panel border border-subtle bg-black"
            />
          ) : (
            <p className="text-[12px] text-muted">No image for this stage on this frame.</p>
          )}
        </Panel>
      </div>

      <Panel title="How this is scored">
        <p className="max-w-prose text-[12.5px] leading-relaxed text-secondary">
          Not by eye. The renderer produces two clips identical in every respect
          except <code className="num text-accent">cast_shadows</code>, so the true
          shadow pixels are known exactly by differencing them. Against that,
          suppression removes <strong>87–97%</strong> of true shadow pixels.
        </p>
        <p className="mt-2 max-w-prose text-[12.5px] leading-relaxed text-muted">
          That number is also why shadow was ruled out as the cause of the Phase 3
          failure: rendering with no shadows at all barely moved frame-count
          agreement. If shadow were the constraint, removing it perfectly would have
          fixed the number. It did not — the constraint is fragmentation.
        </p>
      </Panel>
    </div>
  );
}
