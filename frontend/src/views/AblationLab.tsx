/**
 * The ablation lab.
 *
 * Each study states its question before it runs. That ordering is deliberate:
 * a study whose question is written afterwards is a study whose question was
 * chosen to match the answer. The first row is the baseline and every other row
 * shows its delta against it, so an improvement and a regression are equally
 * legible.
 */

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, type AblationResult, type AblationRow } from "../lib/api";
import { BROWSER_CLIP, IN_BROWSER } from "../lib/engine";
import { Button, Empty, ErrorNote, Panel } from "../components/primitives";

export function AblationLab() {
  const list = useQuery({ queryKey: ["ablations"], queryFn: api.ablations });
  // In the browser every setting of a study is a full render and analysis on
  // this device, so the default clip is short there.
  const [duration, setDuration] = useState(IN_BROWSER ? 2 : 6);
  const [results, setResults] = useState<Record<string, AblationResult>>({});

  const run = useMutation({
    mutationFn: (key: string) =>
      api.runAblation(key, duration, IN_BROWSER ? BROWSER_CLIP : undefined),
    onSuccess: (data) => setResults((r) => ({ ...r, [data.key]: data })),
  });

  if (list.isError) return <ErrorNote message={String(list.error)} />;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto">
      <Panel
        title="Studies"
        actions={
          <label className="flex items-center gap-2 text-[11.5px] text-muted">
            duration
            <input
              type="number"
              min={2}
              max={30}
              step={1}
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="num h-7 w-14 rounded-ctl border border-line bg-raised px-1.5 text-[12px] text-primary"
            />
            s
          </label>
        }
      >
        <p className="mb-3 max-w-prose text-[12.5px] text-secondary">
          Each study varies exactly one axis and scores every setting against the same
          ground truth with the same metric set. The full pipeline runs at about 1 fps
          at 720p, so a six-second study with four settings takes several minutes.
          {IN_BROWSER
            ? ` Here it runs in your browser at ${BROWSER_CLIP.width_px}×${BROWSER_CLIP.height_px}, so figures are comparable with each other but not with the 720p results in the report.`
            : null}
        </p>
        <div className="grid gap-2 md:grid-cols-2 2xl:grid-cols-3">
          {list.data?.ablations.map((a) => (
            <div
              key={a.key}
              className="flex flex-col justify-between rounded-panel border border-subtle bg-raised p-3"
            >
              <div>
                <div className="num mb-1 text-[11.5px] text-accent">{a.key}</div>
                <div className="mb-1 text-[13px] font-medium text-primary">{a.label}</div>
                <p className="m-0 text-[11.5px] leading-snug text-muted">{a.question}</p>
              </div>
              <div className="mt-3">
                <Button
                  size="sm"
                  variant={results[a.key] ? "default" : "primary"}
                  disabled={run.isPending}
                  onClick={() => run.mutate(a.key)}
                >
                  {run.isPending && run.variables === a.key
                    ? "Running…"
                    : results[a.key]
                      ? "Re-run"
                      : "Run"}
                </Button>
              </div>
            </div>
          ))}
        </div>
        {run.isError ? (
          <div className="mt-3">
            <ErrorNote message={String(run.error)} />
          </div>
        ) : null}
      </Panel>

      {Object.values(results).length === 0 ? (
        <Empty message="No study has been run in this session yet. Results appear below, and stay until the page is reloaded so two studies can be compared." />
      ) : null}

      {Object.values(results).map((result) => (
        <ResultTable key={result.key} result={result} />
      ))}
    </div>
  );
}

const COLUMNS: { key: keyof AblationRow; label: string; better: "higher" | "lower" }[] = [
  { key: "count_accuracy", label: "Count acc.", better: "higher" },
  { key: "recall", label: "Recall", better: "higher" },
  { key: "precision", label: "Precision", better: "higher" },
  { key: "false_merge_rate", label: "False merge", better: "lower" },
  { key: "frame_agreement", label: "Frame agree", better: "higher" },
  { key: "speed_mae_kmph", label: "Route A MAE", better: "lower" },
  { key: "blockmatch_mae_kmph", label: "Route B MAE", better: "lower" },
  { key: "lane_accuracy", label: "Lane acc.", better: "higher" },
  { key: "class_accuracy", label: "Class acc.", better: "higher" },
  { key: "fps", label: "fps", better: "higher" },
];

function ResultTable({ result }: { result: AblationResult }) {
  const baseline = result.rows[0];

  return (
    <Panel
      title={result.label}
      badge={<span className="num text-[11px] text-muted">{result.axis}</span>}
      actions={
        <span className="num text-[11px] text-muted">
          {result.scenario} · {result.duration_s}s
        </span>
      }
    >
      <p className="mb-3 max-w-prose text-[12.5px] text-secondary">{result.question}</p>
      <div className="overflow-auto rounded-panel border border-subtle">
        <table className="w-full border-collapse text-[12px]">
          <thead>
            <tr>
              <th className="sticky left-0 border-b border-subtle bg-raised px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
                Setting
              </th>
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  className="border-b border-subtle bg-raised px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-[0.08em] text-muted"
                >
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row, i) => (
              <tr key={row.setting} className="border-b border-subtle last:border-0">
                <td
                  className={`sticky left-0 bg-surface px-3 py-1.5 text-left ${
                    i === 0 ? "font-medium text-primary" : "text-secondary"
                  }`}
                >
                  {row.setting}
                  {i === 0 ? (
                    <span className="ml-1.5 text-[10px] text-muted">baseline</span>
                  ) : null}
                </td>
                {COLUMNS.map((c) => (
                  <Cell
                    key={c.key}
                    value={row[c.key] as number | null}
                    base={i === 0 ? null : ((baseline?.[c.key] ?? null) as number | null)}
                    better={c.better}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11.5px] text-muted">{result.gate.description}</p>
    </Panel>
  );
}

function Cell({
  value,
  base,
  better,
}: {
  value: number | null;
  base: number | null;
  better: "higher" | "lower";
}) {
  if (value == null || !Number.isFinite(value)) {
    return <td className="num px-3 py-1.5 text-right text-muted">—</td>;
  }
  const delta = base == null || !Number.isFinite(base) ? null : value - base;
  const improved = delta == null ? null : better === "higher" ? delta > 0 : delta < 0;
  return (
    <td className="num whitespace-nowrap px-3 py-1.5 text-right">
      <span className="text-primary">{value.toFixed(value >= 100 ? 0 : 2)}</span>
      {delta != null && Math.abs(delta) > 1e-6 ? (
        <span className={`ml-1.5 text-[10.5px] ${improved ? "text-free" : "text-stop"}`}>
          {delta > 0 ? "+" : ""}
          {delta.toFixed(2)}
        </span>
      ) : null}
    </td>
  );
}
