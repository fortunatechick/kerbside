/**
 * The report view.
 *
 * One button, one self-contained HTML file. Self-contained is the requirement
 * that shapes it: every figure is an inline data URI and every style is inline,
 * so the file can be emailed, opened offline or printed without a directory of
 * assets behind it. The parameters and the calibration that produced the numbers
 * are embedded verbatim, because a result without its settings cannot be
 * reproduced or argued with — which is the entire purpose of the document.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Button, Empty, ErrorNote, Panel, Stat } from "../components/primitives";
import { kmph } from "../lib/format";

interface ReportShape {
  generated_utc: string;
  source: Record<string, string | number | boolean>;
  analytics: {
    total_count: number;
    speed: { p85_kmph: number; mean_kmph: number };
    disclaimer: string;
    flagged: unknown[];
  };
  calibration: { verdict: string; camera_height_m: number; camera_tilt_deg: number } | null;
  figures: { caption: string; uri: string }[];
  timings: Record<string, number>;
}

export function ReportView({ loaded }: { loaded: boolean }) {
  const report = useQuery({
    queryKey: ["report"],
    queryFn: () => api.report() as Promise<unknown>,
    enabled: loaded,
  });

  if (!loaded) return <Empty message="Load and analyse a clip before generating a report." />;
  if (report.isError) return <ErrorNote message={String(report.error)} />;
  if (!report.data) {
    return (
      <p className="text-[13px] text-muted">
        Building the report — this analyses every frame if it has not been done yet.
      </p>
    );
  }

  const r = report.data as ReportShape;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto">
      <Panel
        title="Export"
        actions={
          <a href="/api/report.html" download>
            <Button variant="primary" size="sm">
              Download HTML report
            </Button>
          </a>
        }
      >
        <p className="max-w-prose text-[13px] leading-relaxed text-secondary">
          One file. Every figure is an inline data URI and every style is inline, so
          it opens offline, prints, and can be emailed without dragging a directory of
          assets behind it.
        </p>
        <p className="mt-2 max-w-prose text-[12.5px] leading-relaxed text-muted">
          It embeds the full parameter state of all nine stages and the calibration
          matrix used. A number without the settings that produced it cannot be
          reproduced or disagreed with, and a report you cannot disagree with is not
          evidence of anything.
        </p>
        <p className="num mt-2 text-[11.5px] text-muted">generated {r.generated_utc}</p>
      </Panel>

      <div className="grid shrink-0 grid-cols-2 gap-2 md:grid-cols-4">
        <Stat label="Vehicles counted" value={String(r.analytics.total_count)} tone="accent" />
        <Stat label="85th percentile" value={kmph(r.analytics.speed.p85_kmph, 1)} unit="km/h" />
        <Stat label="Mean speed" value={kmph(r.analytics.speed.mean_kmph, 1)} unit="km/h" />
        <Stat label="Flagged for review" value={String(r.analytics.flagged.length)} tone="slow" />
      </div>

      <div className="rounded-panel border border-slow/40 bg-slow/10 px-3 py-2 text-[12px] leading-snug text-slow">
        {r.analytics.disclaimer}
      </div>

      {r.calibration ? (
        <Panel title="Calibration recorded in the report">
          <p className="text-[12.5px] text-secondary">{r.calibration.verdict}</p>
          <p className="num mt-1 text-[12px] text-muted">
            implied camera {r.calibration.camera_height_m?.toFixed(2)} m high,{" "}
            {r.calibration.camera_tilt_deg?.toFixed(2)}° below horizontal
          </p>
        </Panel>
      ) : (
        <ErrorNote message="No calibration is loaded, so no real-world quantity in this report can be trusted." />
      )}

      {r.figures.length ? (
        <Panel title="Figures">
          <div className="grid gap-3 lg:grid-cols-2">
            {r.figures.map((f, i) => (
              <figure key={i} className="m-0">
                <img
                  src={f.uri}
                  alt={f.caption}
                  className="w-full rounded-panel border border-subtle bg-black"
                />
                <figcaption className="mt-1.5 text-[11.5px] leading-snug text-muted">
                  {f.caption}
                </figcaption>
              </figure>
            ))}
          </div>
        </Panel>
      ) : null}

      <Panel title="Where the time goes">
        <ul className="space-y-1">
          {Object.entries(r.timings)
            .sort((a, b) => b[1] - a[1])
            .map(([stage, msValue]) => (
              <li key={stage} className="flex items-center gap-2 text-[12px]">
                <span className="w-[140px] shrink-0 truncate text-secondary">{stage}</span>
                <span className="h-[3px] flex-1 overflow-hidden rounded bg-raised">
                  <span
                    className="block h-full bg-accent"
                    style={{
                      width: `${Math.min(
                        100,
                        (msValue / Math.max(...Object.values(r.timings))) * 100,
                      )}%`,
                    }}
                  />
                </span>
                <span className="num w-[64px] shrink-0 text-right text-primary">
                  {msValue.toFixed(1)} ms
                </span>
              </li>
            ))}
        </ul>
      </Panel>
    </div>
  );
}
