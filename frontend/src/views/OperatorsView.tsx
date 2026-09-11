/**
 * The operator registry, browsable.
 *
 * "Every operator carries a rationale and its syllabus unit, surfaced in the
 * UI" is a course requirement, and the parameter rail only satisfies it for the
 * nine stages — which between them drive a fraction of the library. This view
 * is the rest: all 35 operators, grouped by the unit they belong to, each with
 * the reason it exists and the parameters it takes.
 *
 * It is deliberately a reference rather than a playground. An operator invoked
 * here in isolation would be running on a frame with no pipeline state behind
 * it, and a result produced that way would not be the result the pipeline gets.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type OperatorSpec } from "../lib/api";
import { ErrorNote, Panel, UnitChips } from "../components/primitives";

const UNIT_NAMES: Record<number, string> = {
  1: "Digital image fundamentals, sampling, quantisation",
  2: "Intensity transforms, histograms, colour models",
  3: "Spatial filtering, smoothing, sharpening",
  4: "Frequency domain: DFT, DCT, filtering",
  5: "Restoration and degradation models",
  6: "Morphology and connected components",
  7: "Segmentation, thresholding, edges, watershed",
  8: "Video: motion estimation and compression",
};

export function OperatorsView() {
  const operators = useQuery({ queryKey: ["operators"], queryFn: api.operators });
  const [query, setQuery] = useState("");
  const [unit, setUnit] = useState<number | null>(null);

  const filtered = useMemo(() => {
    const all = operators.data?.operators ?? [];
    const needle = query.trim().toLowerCase();
    return all.filter((op) => {
      if (unit != null && !op.units.includes(unit)) return false;
      if (!needle) return true;
      return (
        op.name.toLowerCase().includes(needle) ||
        op.operator_id.toLowerCase().includes(needle) ||
        op.rationale.toLowerCase().includes(needle) ||
        op.tags.some((t) => t.toLowerCase().includes(needle))
      );
    });
  }, [operators.data, query, unit]);

  const byUnit = useMemo(() => {
    const groups = new Map<number, OperatorSpec[]>();
    for (const op of filtered) {
      // An operator that exercises two units appears under both, because the
      // question a reader has is "what covers unit 6", not "where does this
      // one thing live".
      for (const u of op.units) {
        const list = groups.get(u) ?? [];
        list.push(op);
        groups.set(u, list);
      }
    }
    return [...groups.entries()].sort((a, b) => a[0] - b[0]);
  }, [filtered]);

  if (operators.isError) return <ErrorNote message={String(operators.error)} />;

  const total = operators.data?.operators.length ?? 0;

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto">
      <Panel
        title="Operator registry"
        badge={
          <span className="num text-[11px] text-muted">
            {filtered.length} of {total}
          </span>
        }
        actions={
          <input
            value={query}
            placeholder="Search name, id, rationale, tag"
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search operators"
            className="h-7 w-[220px] rounded-ctl border border-line bg-raised px-2 text-[12px] text-primary placeholder:text-muted"
          />
        }
      >
        <p className="mb-3 max-w-prose text-[12.5px] leading-relaxed text-secondary">
          Every operator in the library, with the syllabus unit it belongs to and the
          reason it exists. The pipeline drives a subset of these; the rest are here
          because the comparison studies need them, and because an operator nobody
          can find is an operator nobody can check.
        </p>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setUnit(null)}
            className={`rounded-ctl border px-2 py-0.5 text-[11.5px] transition-colors duration-fast ${
              unit == null
                ? "border-accent bg-accent-dim text-accent"
                : "border-line bg-raised text-muted hover:text-secondary"
            }`}
          >
            All units
          </button>
          {[1, 2, 3, 4, 5, 6, 7, 8].map((u) => (
            <button
              key={u}
              type="button"
              title={UNIT_NAMES[u]}
              onClick={() => setUnit(unit === u ? null : u)}
              className={`num rounded-ctl border px-2 py-0.5 text-[11.5px] transition-colors duration-fast ${
                unit === u
                  ? "border-accent bg-accent-dim text-accent"
                  : "border-line bg-raised text-muted hover:text-secondary"
              }`}
            >
              U{u}
            </button>
          ))}
        </div>
      </Panel>

      {operators.isLoading ? (
        <p className="text-[13px] text-muted">Loading the registry…</p>
      ) : null}

      {byUnit.length === 0 && !operators.isLoading ? (
        <p className="text-[13px] text-muted">
          Nothing matches “{query}”. The registry holds {total} operators.
        </p>
      ) : null}

      {byUnit.map(([u, ops]) => (
        <section key={u}>
          <h3 className="mb-2 flex items-baseline gap-2 text-[13px] font-semibold text-primary">
            <span className="num text-accent">Unit {u}</span>
            <span className="text-[12px] font-normal text-muted">{UNIT_NAMES[u]}</span>
            <span className="num ml-auto text-[11px] text-muted">{ops.length}</span>
          </h3>
          <div className="grid gap-2 md:grid-cols-2 2xl:grid-cols-3">
            {ops.map((op) => (
              <OperatorCard key={`${u}-${op.operator_id}`} op={op} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function OperatorCard({ op }: { op: OperatorSpec }) {
  const [open, setOpen] = useState(false);
  return (
    <article className="rounded-panel border border-subtle bg-surface p-3">
      <header className="mb-1.5 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h4 className="truncate text-[13px] font-semibold text-primary">{op.name}</h4>
          <p className="num m-0 truncate text-[10.5px] text-muted">{op.operator_id}</p>
        </div>
        <UnitChips units={op.units} />
      </header>

      <p className="m-0 text-[11.5px] leading-relaxed text-secondary">{op.rationale}</p>

      {op.reference ? (
        <p className="mt-1.5 text-[11px] italic leading-snug text-muted">{op.reference}</p>
      ) : null}

      {op.params.length ? (
        <>
          <button
            type="button"
            onClick={() => setOpen(!open)}
            className="mt-2 text-[11.5px] text-accent hover:underline"
          >
            {open ? "Hide" : "Show"} {op.params.length} parameter
            {op.params.length === 1 ? "" : "s"}
          </button>
          {open ? (
            <dl className="mt-1.5 space-y-1.5 border-t border-subtle pt-2">
              {op.params.map((p) => (
                <div key={p.name}>
                  <dt className="flex items-baseline justify-between gap-2">
                    <span className="num text-[11px] text-primary">{p.name}</span>
                    <span className="num text-[10.5px] text-muted">
                      {p.kind}
                      {p.minimum != null && p.maximum != null
                        ? ` ${p.minimum}–${p.maximum}`
                        : ""}
                      {p.unit ? ` ${p.unit}` : ""}
                    </span>
                  </dt>
                  {p.description ? (
                    <dd className="m-0 text-[11px] leading-snug text-muted">{p.description}</dd>
                  ) : null}
                </div>
              ))}
            </dl>
          ) : null}
        </>
      ) : (
        <p className="mt-2 text-[11px] text-muted">No parameters — it does one thing.</p>
      )}

      {op.tags.length ? (
        <div className="mt-2 flex flex-wrap gap-1">
          {op.tags.map((t) => (
            <span
              key={t}
              className="rounded border border-line bg-raised px-1.5 text-[10px] leading-[16px] text-muted"
            >
              {t}
            </span>
          ))}
        </div>
      ) : null}
    </article>
  );
}
