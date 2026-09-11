/**
 * The component primitives.
 *
 * Built rather than imported, per the brief: no MUI, Ant or Chakra. Each one
 * encodes a rule from the design system so a view cannot accidentally break it
 * — panels always have an uppercase header and a hairline divider, sliders
 * always show their value in mono to the right of the label, numbers are never
 * centred.
 */

import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

/* ---------------------------------------------------------------- Panel -- */

export function Panel({
  title,
  actions,
  children,
  className = "",
  badge,
  scroll = false,
}: {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  badge?: ReactNode;
  scroll?: boolean;
}) {
  return (
    <section
      className={`flex min-h-0 flex-col rounded-panel border border-subtle bg-surface ${className}`}
    >
      <header className="flex items-center justify-between gap-3 border-b border-subtle px-4 py-2.5">
        <div className="flex items-center gap-2">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">
            {title}
          </h2>
          {badge}
        </div>
        {actions ? <div className="flex items-center gap-1.5">{actions}</div> : null}
      </header>
      <div className={`min-h-0 p-4 ${scroll ? "overflow-auto" : ""}`}>{children}</div>
    </section>
  );
}

/* --------------------------------------------------------------- Button -- */

export function Button({
  children,
  onClick,
  variant = "default",
  size = "md",
  disabled,
  title,
  active,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "ghost" | "danger";
  size?: "sm" | "md";
  disabled?: boolean;
  title?: string;
  active?: boolean;
}) {
  const base =
    "inline-flex items-center justify-center gap-1.5 rounded-ctl border font-medium " +
    "transition-colors duration-fast disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap";
  const sizes = size === "sm" ? "h-7 px-2.5 text-[12px]" : "h-8 px-3 text-[13px]";
  const variants = {
    default: "border-line bg-raised text-primary hover:bg-overlay",
    primary: "border-accent bg-accent text-inverse hover:bg-accent-hover",
    ghost: "border-transparent bg-transparent text-secondary hover:bg-raised hover:text-primary",
    danger: "border-line bg-raised text-stop hover:bg-overlay",
  }[variant];
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`${base} ${sizes} ${variants} ${active ? "!border-accent !text-accent" : ""}`}
    >
      {children}
    </button>
  );
}

/* --------------------------------------------------------------- Slider -- */

export function Slider({
  label,
  value,
  min,
  max,
  step,
  unit,
  description,
  isDefault,
  onChange,
  onCommit,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  description?: string;
  isDefault: boolean;
  onChange: (v: number) => void;
  onCommit: (v: number) => void;
}) {
  const decimals = step < 1 ? String(step).split(".")[1]?.length ?? 2 : 0;
  return (
    <label className="block" title={description}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[12px] font-medium text-secondary">
          {label}
          {isDefault ? null : <span className="ml-1.5 text-accent" title="changed from default">•</span>}
        </span>
        {/* The value always sits in mono to the right of the label. */}
        <span className="num text-[13px] font-medium text-primary">
          {value.toFixed(decimals)}
          {unit ? <span className="ml-0.5 text-muted">{unit}</span> : null}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(e) => onChange(Number(e.target.value))}
        onPointerUp={(e) => onCommit(Number((e.target as HTMLInputElement).value))}
        onKeyUp={(e) => onCommit(Number((e.target as HTMLInputElement).value))}
      />
    </label>
  );
}

/* ---------------------------------------------------------------- Field -- */

export function Toggle({
  label,
  checked,
  onChange,
  description,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  description?: string;
}) {
  return (
    <label
      className="flex cursor-pointer items-center justify-between gap-3 py-1"
      title={description}
    >
      <span className="text-[12px] font-medium text-secondary">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={`relative h-[18px] w-8 shrink-0 rounded-full border transition-colors duration-fast ${
          checked ? "border-accent bg-accent-dim" : "border-line bg-raised"
        }`}
      >
        <span
          className={`absolute top-[2px] h-[12px] w-[12px] rounded-full transition-all duration-fast ${
            checked ? "left-[16px] bg-accent" : "left-[2px] bg-muted"
          }`}
        />
      </button>
    </label>
  );
}

export function Select({
  label,
  value,
  options,
  onChange,
  description,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  description?: string;
}) {
  return (
    <label className="block" title={description}>
      <span className="mb-1 block text-[12px] font-medium text-secondary">{label}</span>
      <select
        value={value}
        aria-label={label}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 w-full rounded-ctl border border-line bg-raised px-2 text-[13px] text-primary"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

/* ------------------------------------------------------------ Stat tile -- */

export function Stat({
  label,
  value,
  unit,
  tone = "default",
  hint,
  big = false,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: "default" | "free" | "slow" | "stop" | "accent";
  hint?: string;
  big?: boolean;
}) {
  const tones = {
    default: "text-primary",
    free: "text-free",
    slow: "text-slow",
    stop: "text-stop",
    accent: "text-accent",
  }[tone];
  return (
    <div className="rounded-panel border border-subtle bg-raised px-3 py-2" title={hint}>
      <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
        {label}
      </div>
      <div className={`num font-semibold ${tones} ${big ? "text-[28px] leading-tight" : "text-[17px]"}`}>
        {value}
        {unit ? <span className="ml-1 text-[11px] font-normal text-muted">{unit}</span> : null}
      </div>
    </div>
  );
}

/** A number that flashes green or red when it changes — improvement or
 *  regression. One of the two deliberate exceptions to the motion budget. */
export function DeltaValue({
  value,
  format,
  betterWhen = "lower",
}: {
  value: number;
  format: (v: number) => string;
  betterWhen?: "lower" | "higher";
}) {
  const previous = useRef(value);
  const [delta, setDelta] = useState<number | null>(null);

  useEffect(() => {
    if (previous.current !== value && Number.isFinite(previous.current)) {
      setDelta(value - previous.current);
      const timer = window.setTimeout(() => setDelta(null), 1500);
      previous.current = value;
      return () => window.clearTimeout(timer);
    }
    previous.current = value;
    return undefined;
  }, [value]);

  const improved =
    delta == null ? false : betterWhen === "lower" ? delta < 0 : delta > 0;
  return (
    <span className="num inline-flex items-baseline gap-1.5">
      <span>{format(value)}</span>
      {delta != null && Math.abs(delta) > 1e-9 ? (
        <span className={`delta-flash text-[11px] ${improved ? "text-free" : "text-stop"}`}>
          {delta > 0 ? "+" : ""}
          {format(delta)}
        </span>
      ) : null}
    </span>
  );
}

/* ----------------------------------------------------------- Unit chips -- */

export function UnitChips({ units }: { units: number[] }) {
  return (
    <span className="flex gap-1">
      {units.map((u) => (
        <span
          key={u}
          title={`Syllabus unit ${u}`}
          className="num rounded border border-line bg-raised px-1 text-[10px] leading-[15px] text-muted"
        >
          U{u}
        </span>
      ))}
    </span>
  );
}

/* -------------------------------------------------------- Empty & error -- */

export function Empty({ message, action }: { message: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-start gap-3 rounded-panel border border-dashed border-line bg-raised px-4 py-5">
      <p className="m-0 max-w-prose text-[13px] text-secondary">{message}</p>
      {action}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="rounded-panel border border-stop/40 bg-stop/10 px-3 py-2 text-[12.5px] text-stop">
      {message}
    </div>
  );
}

export function Table({
  columns,
  rows,
  numeric = [],
  onRowClick,
}: {
  columns: { key: string; label: string }[];
  rows: Record<string, string | number | null>[];
  numeric?: string[];
  onRowClick?: (row: Record<string, string | number | null>) => void;
}) {
  return (
    <div className="overflow-auto rounded-panel border border-subtle">
      <table className="w-full border-collapse text-[12.5px]">
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                className={`sticky top-0 border-b border-subtle bg-raised px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted ${
                  numeric.includes(c.key) ? "text-right" : "text-left"
                }`}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              onClick={() => onRowClick?.(row)}
              className={`border-b border-subtle last:border-0 ${
                onRowClick ? "cursor-pointer hover:bg-raised" : ""
              }`}
            >
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={`px-3 py-1.5 ${
                    numeric.includes(c.key) ? "num text-right" : "text-left"
                  }`}
                >
                  {row[c.key] ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
