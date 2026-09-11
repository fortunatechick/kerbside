/**
 * The typed API client.
 *
 * Every failure the backend can explain arrives as a 400 with a message written
 * for a person, so this layer surfaces `detail` verbatim rather than replacing
 * it with a status code. "The parameter set produced zero detections" is
 * actionable; "Request failed" is not.
 */

export interface SourceState {
  source_name: string;
  source_kind: string;
  width_px: number;
  height_px: number;
  frame_count: number;
  container_fps: number;
  measured_fps: number;
  codec: string;
  bit_depth: number;
  calibrated: boolean;
  has_ground_truth: boolean;
  warnings: string[];
}

export interface ParamSpec {
  name: string;
  label: string;
  kind: "float" | "int" | "bool" | "enum" | "odd_int";
  default: unknown;
  minimum: number | null;
  maximum: number | null;
  step: number | null;
  choices: string[];
  unit: string;
  description: string;
}

export interface StageInfo {
  stage_id: string;
  name: string;
  units: number[];
  rationale: string;
  params: ParamSpec[];
  values: Record<string, unknown>;
  modified: string[];
}

export interface StageFrame {
  stage_id: string;
  name: string;
  units: number[];
  rationale: string;
  elapsed_ms: number;
  metrics: Record<string, number>;
  extras: Record<string, unknown>;
  modified: string[];
  image_png?: string;
}

export interface TrackSnapshot {
  track_id: number;
  frame_index: number;
  time_s: number;
  state: string;
  lane_id: number | null;
  direction: string;
  counted: boolean;
  vehicle_class: string;
  colour_name: string;
  world_pos_m: [number, number] | null;
  ground_point_px: [number, number] | null;
  bbox_px: [number, number, number, number] | null;
  footprint_m: [number, number, number] | null;
  /** Route A: the ground point differenced through the road homography. */
  speed_kmph_track: number;
  /** Route B: median block displacement, mapped through the elevated-plane homography. */
  speed_kmph_blockmatch: number;
  hits: number;
  age_frames: number;
}

export interface Telemetry {
  frame_index: number;
  time_s: number;
  frame_ms: number;
  frame_type: string;
  blobs: number;
  tracks: TrackSnapshot[];
  counts: Record<string, number>;
  total_count: number;
  shake_px: [number, number];
  stages: Record<string, { ms: number; metrics: Record<string, number> }>;
}

export interface Status {
  state: SourceState;
  frames: number;
  analysed: boolean;
  total_count: number;
  counts_by_lane: Record<string, number>;
  timings: Record<string, number>;
  slowest_stage: [string, number] | null;
  cache: Record<string, Record<string, number>>;
}

export interface Analytics {
  total_count: number;
  duration_s: number;
  counts_by_lane: Record<string, number>;
  counts_by_class: Record<string, number>;
  speed: { mean_kmph: number; median_kmph: number; p85_kmph: number; limit_kmph: number };
  speed_histogram: { counts: number[]; edges: number[] };
  intervals: { t_s: number; count: number; flow_veh_per_hour: number; mean_speed_kmph: number }[];
  fundamental_diagram: { flow_veh_per_hour: number; density_veh_per_km: number; mean_speed_kmph: number }[];
  headways_s: number[];
  vehicles: Record<string, string | number | null>[];
  flagged: Record<string, string | number | null>[];
  disclaimer: string;
}

export interface CalibrationInfo {
  calibrated: boolean;
  homography?: number[][];
  rmse_px?: number;
  verdict?: string;
  exactly_determined?: boolean;
  warnings?: string[];
  camera?: {
    focal_px: number | null;
    height_m: number | null;
    tilt_deg: number | null;
    plausible: boolean;
    reasons: string[];
  };
  horizon_row_px?: number;
  covered_y_range_m?: [number, number] | null;
  lanes?: unknown[];
  suggestions: { name: string; nominal_m: number; range_m: number[]; caution: string }[];
}

export interface AblationRow {
  setting: string;
  count_accuracy: number;
  counted: number;
  expected: number;
  recall: number;
  precision: number;
  false_merge_rate: number;
  frame_agreement: number;
  speed_mae_kmph: number | null;
  blockmatch_mae_kmph: number | null;
  lane_accuracy: number;
  class_accuracy: number;
  fps: number;
}

export interface AblationResult {
  key: string;
  label: string;
  question: string;
  axis: string;
  scenario: string;
  gate: { description: string };
  duration_s: number;
  rows: AblationRow[];
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* a non-JSON error body is still worth surfacing as the status line */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  scenarios: () =>
    request<{ scenarios: { name: string; label: string; description: string; exercises: string }[] }>(
      "/api/scenarios",
    ),

  loadScenario: (body: Record<string, unknown>) =>
    request<SourceState>("/api/source/scenario", { method: "POST", body: JSON.stringify(body) }),

  loadVideo: (path: string) =>
    request<SourceState>("/api/source/video", { method: "POST", body: JSON.stringify({ path }) }),

  status: () => request<Status>("/api/status"),

  analyse: (force = false) =>
    request<Status>(`/api/analyse?force=${force}`, { method: "POST" }),

  pipeline: () => request<{ stages: StageInfo[] }>("/api/pipeline"),

  setParams: (stage_id: string, params: Record<string, unknown>) =>
    request<{ stage: StageInfo }>("/api/pipeline/params", {
      method: "POST",
      body: JSON.stringify({ stage_id, params }),
    }),

  resetParams: (stageId?: string) =>
    request<{ stages: StageInfo[] }>(
      `/api/pipeline/reset${stageId ? `?stage_id=${stageId}` : ""}`,
      { method: "POST" },
    ),

  frameStages: (index: number, includeImages = true) =>
    request<{ frame_index: number; stages: StageFrame[] }>(
      `/api/frame/${index}/stages?include_images=${includeImages}`,
    ),

  probe: (index: number, x: number, y: number) =>
    request<Record<string, unknown>>(`/api/frame/${index}/probe?x=${x}&y=${y}`),

  transport: (body: Record<string, unknown>) =>
    request<{ playing: boolean; cursor: number; speed: number; frames: number }>(
      "/api/transport",
      { method: "POST", body: JSON.stringify(body) },
    ),

  calibration: () => request<CalibrationInfo>("/api/calibration"),

  calibrate: (body: Record<string, unknown>) =>
    request<CalibrationInfo>("/api/calibration", { method: "POST", body: JSON.stringify(body) }),

  measure: (a: [number, number], b: [number, number]) =>
    request<{ distance_m: number | null; note: string }>("/api/calibration/measure", {
      method: "POST",
      body: JSON.stringify({ a_px: a, b_px: b }),
    }),

  grid: () => request<{ polylines: [number, number][][]; step_m: number }>("/api/calibration/grid"),

  saveCalibration: (key: string) =>
    request<{ saved: string }>("/api/calibration/save", {
      method: "POST",
      body: JSON.stringify({ key }),
    }),

  analytics: () => request<Analytics>("/api/analytics"),

  ablations: () =>
    request<{ ablations: { key: string; label: string; question: string }[] }>("/api/ablations"),

  runAblation: (key: string, duration = 6) =>
    request<AblationResult>(`/api/ablations/${key}?duration_s=${duration}`, { method: "POST" }),

  report: () => request<Record<string, unknown>>("/api/report"),

  log: (limit = 200) =>
    request<{ entries: { t: number; source: string; message: string }[] }>(`/api/log?limit=${limit}`),
};

/** The MJPEG stream URL, with the overlay layers encoded as query parameters. */
export function streamUrl(layers: Record<string, boolean>, nonce: number): string {
  const query = new URLSearchParams(
    Object.fromEntries(Object.entries(layers).map(([k, v]) => [k, v ? "1" : "0"])),
  );
  query.set("t", String(nonce));
  return `/api/stream.mjpg?${query.toString()}`;
}

export function frameUrl(index: number, layers: Record<string, boolean>): string {
  const query = new URLSearchParams(
    Object.fromEntries(Object.entries(layers).map(([k, v]) => [k, v ? "1" : "0"])),
  );
  return `/api/frame/${index}.jpg?${query.toString()}`;
}
