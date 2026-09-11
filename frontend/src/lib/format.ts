/** Formatting helpers. Every number in the interface goes through one of these,
 *  so units and precision are decided once rather than at each call site. */

export const kmph = (v: number | null | undefined, digits = 1): string =>
  v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(digits)}`;

export const metres = (v: number | null | undefined, digits = 2): string =>
  v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(digits)}`;

export const ms = (v: number | null | undefined, digits = 1): string =>
  v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(digits)}`;

export const pct = (v: number | null | undefined, digits = 1): string =>
  v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const int = (v: number | null | undefined): string =>
  v == null || !Number.isFinite(v) ? "—" : String(Math.round(v));

export const seconds = (v: number): string => {
  const m = Math.floor(v / 60);
  const s = v - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
};

/** A stage id like "s4_foreground" as the short label the rail shows. */
export const stageLabel = (id: string): string => id.split("_")[0]?.toUpperCase() ?? id;
