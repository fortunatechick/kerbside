/**
 * The parameter rail.
 *
 * Nothing here is hand-written per stage. The controls are generated from the
 * backend's operator registry — its parameter kind, range, step, unit and
 * rationale — so adding an operator to the pipeline makes it appear in the
 * interface without touching the interface. That is also what guarantees the
 * label a user reads matches the parameter the pipeline actually applies.
 */

import { useCallback, useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type ParamSpec, type StageInfo } from "../lib/api";
import { useStore } from "../lib/store";
import { Button, Select, Slider, Toggle, UnitChips } from "./primitives";

export function ParamRail({ stages }: { stages: StageInfo[] }) {
  const selected = useStore((s) => s.selectedStage);
  const selectStage = useStore((s) => s.selectStage);
  const setStages = useStore((s) => s.setStages);
  const pushLog = useStore((s) => s.pushLog);
  const client = useQueryClient();

  const stage = stages.find((s) => s.stage_id === selected) ?? stages[0];

  const apply = useMutation({
    mutationFn: ({ id, params }: { id: string; params: Record<string, unknown> }) =>
      api.setParams(id, params),
    onSuccess: (data) => {
      setStages(stages.map((s) => (s.stage_id === data.stage.stage_id ? data.stage : s)));
      void client.invalidateQueries({ queryKey: ["status"] });
      void client.invalidateQueries({ queryKey: ["frame-stages"] });
    },
    onError: (error) =>
      pushLog({ t: Date.now(), source: "params", message: String(error) }),
  });

  const reset = useMutation({
    mutationFn: (id?: string) => api.resetParams(id),
    onSuccess: (data) => setStages(data.stages),
  });

  const commit = useCallback(
    (name: string, value: unknown) => {
      if (!stage) return;
      apply.mutate({ id: stage.stage_id, params: { [name]: value } });
    },
    [apply, stage],
  );

  if (!stage) {
    return <p className="p-3 text-[12.5px] text-muted">Load a source to see the pipeline.</p>;
  }

  return (
    <div className="flex min-h-0 flex-col">
      {/* stage picker */}
      <div className="border-b border-subtle p-2">
        <div className="grid grid-cols-3 gap-1">
          {stages.map((s) => (
            <button
              key={s.stage_id}
              type="button"
              title={s.name}
              onClick={() => selectStage(s.stage_id)}
              className={`relative rounded-ctl border px-1 py-1 text-[11px] font-semibold transition-colors duration-fast ${
                s.stage_id === stage.stage_id
                  ? "border-accent bg-accent-dim text-accent"
                  : "border-line bg-raised text-muted hover:text-secondary"
              }`}
            >
              {s.stage_id.split("_")[0]?.toUpperCase()}
              {s.modified.length > 0 ? (
                <span
                  className="absolute right-1 top-0.5 text-accent"
                  title={`${s.modified.length} parameter(s) changed from default`}
                >
                  •
                </span>
              ) : null}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-3">
        <div className="mb-2 flex items-start justify-between gap-2">
          <h3 className="text-[13px] font-semibold text-primary">{stage.name}</h3>
          <UnitChips units={stage.units} />
        </div>
        {/* The rationale is not a tooltip. A stage whose reason for existing is
            hidden behind a hover is a stage the user will mis-tune. */}
        <p className="mb-4 text-[11.5px] leading-relaxed text-muted">{stage.rationale}</p>

        <div className="space-y-3.5">
          {stage.params.map((spec) => (
            <Control
              key={spec.name}
              spec={spec}
              value={stage.values[spec.name]}
              modified={stage.modified.includes(spec.name)}
              onCommit={(v) => commit(spec.name, v)}
            />
          ))}
        </div>

        <div className="mt-5 flex gap-1.5">
          <Button size="sm" onClick={() => reset.mutate(stage.stage_id)}>
            Reset stage
          </Button>
          <Button size="sm" variant="ghost" onClick={() => reset.mutate(undefined)}>
            Reset all
          </Button>
        </div>
      </div>
    </div>
  );
}

function Control({
  spec,
  value,
  modified,
  onCommit,
}: {
  spec: ParamSpec;
  value: unknown;
  modified: boolean;
  onCommit: (v: unknown) => void;
}) {
  if (spec.kind === "bool") {
    return (
      <Toggle
        label={spec.label}
        description={spec.description}
        checked={Boolean(value)}
        onChange={onCommit}
      />
    );
  }
  if (spec.kind === "enum") {
    return (
      <Select
        label={spec.label}
        description={spec.description}
        value={String(value)}
        options={spec.choices}
        onChange={onCommit}
      />
    );
  }
  // An odd_int parameter is a structuring element or kernel size: stepping it by
  // one would offer even values that the operator would silently round, so the
  // control steps by two and the user never sees a value the pipeline rejects.
  const step = spec.kind === "odd_int" ? 2 : spec.kind === "int" ? 1 : (spec.step ?? 0.01);
  return (
    <SliderControl
      spec={spec}
      step={step}
      value={Number(value)}
      modified={modified}
      onCommit={onCommit}
    />
  );
}

function SliderControl({
  spec,
  step,
  value,
  modified,
  onCommit,
}: {
  spec: ParamSpec;
  step: number;
  value: number;
  modified: boolean;
  onCommit: (v: unknown) => void;
}) {
  // Local state while dragging, committed on release: a slider that re-runs the
  // pipeline on every pointer move at 4 fps is a slider that fights the user.
  const [local, setLocal] = useLocalNumber(value);
  return (
    <Slider
      label={spec.label}
      description={spec.description}
      unit={spec.unit}
      min={spec.minimum ?? 0}
      max={spec.maximum ?? 1}
      step={step}
      value={local}
      isDefault={!modified}
      onChange={setLocal}
      onCommit={(v) => onCommit(v)}
    />
  );
}

function useLocalNumber(external: number): [number, (v: number) => void] {
  const [local, setLocal] = useState(external);
  useEffect(() => setLocal(external), [external]);
  return [local, setLocal];
}
