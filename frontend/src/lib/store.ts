import { create } from "zustand";
import type { StageInfo, Telemetry } from "./api";

export type ViewId =
  | "console"
  | "calibration"
  | "pipeline"
  | "operators"
  | "shadow"
  | "motion"
  | "analytics"
  | "ablation"
  | "report";

export interface LogEntry {
  t: number;
  source: string;
  message: string;
  ms?: number;
}

interface AppState {
  view: ViewId;
  setView: (v: ViewId) => void;

  railCollapsed: boolean;
  toggleRail: () => void;
  consoleOpen: boolean;
  toggleConsole: () => void;
  inspectorOpen: boolean;
  toggleInspector: () => void;

  selectedStage: string;
  selectStage: (id: string) => void;
  selectedTrack: number | null;
  selectTrack: (id: number | null) => void;

  stages: StageInfo[];
  setStages: (s: StageInfo[]) => void;

  telemetry: Telemetry | null;
  setTelemetry: (t: Telemetry) => void;

  /** Overlay layers, persisted so a user's choices survive a reload. */
  layers: Record<string, boolean>;
  toggleLayer: (k: string) => void;

  /** Bumped each time the pipeline is re-run, so frames are fetched afresh. */
  revision: number;
  bumpRevision: () => void;

  log: LogEntry[];
  pushLog: (e: LogEntry) => void;
  logPaused: boolean;
  setLogPaused: (v: boolean) => void;
}

const LAYER_KEY = "kerbside.layers";

/** True on a phone or tablet. `any-pointer` rather than `pointer`, because an
 *  iPad with a keyboard and trackpad attached reports a fine primary pointer
 *  while still being used mostly by touch. */
export function isTouchDevice(): boolean {
  try {
    return window.matchMedia("(any-pointer: coarse)").matches || navigator.maxTouchPoints > 1;
  } catch {
    return false;
  }
}

function loadLayers(): Record<string, boolean> {
  const fallback = {
    boxes: true,
    ground_points: true,
    trails: true,
    labels: true,
    lanes: true,
    counting_lines: true,
    mask: false,
    shadow: false,
    blobs: false,
  };
  try {
    const raw = localStorage.getItem(LAYER_KEY);
    return raw ? { ...fallback, ...(JSON.parse(raw) as Record<string, boolean>) } : fallback;
  } catch {
    // Private windows and blocked site data both throw here; the defaults are
    // a perfectly good answer and losing a toggle preference is not an error.
    return fallback;
  }
}

export const useStore = create<AppState>((set, get) => ({
  view: "console",
  setView: (v) => set({ view: v }),

  railCollapsed: false,
  toggleRail: () => set((s) => ({ railCollapsed: !s.railCollapsed })),
  // Collapsed by default on a touch screen: on a tablet the vertical space is
  // worth more to the video than to a log, and it is one tap to open.
  consoleOpen: !isTouchDevice(),
  toggleConsole: () => set((s) => ({ consoleOpen: !s.consoleOpen })),
  inspectorOpen: false,
  toggleInspector: () => set((s) => ({ inspectorOpen: !s.inspectorOpen })),

  selectedStage: "s4_foreground",
  selectStage: (id) => set({ selectedStage: id }),
  selectedTrack: null,
  selectTrack: (id) => set({ selectedTrack: id }),

  stages: [],
  setStages: (stages) => set({ stages }),

  telemetry: null,
  setTelemetry: (telemetry) => set({ telemetry }),

  layers: loadLayers(),
  toggleLayer: (k) => {
    const layers = { ...get().layers, [k]: !get().layers[k] };
    set({ layers });
    try {
      localStorage.setItem(LAYER_KEY, JSON.stringify(layers));
    } catch {
      /* preference only; not worth surfacing */
    }
  },

  revision: 0,
  bumpRevision: () => set((s) => ({ revision: s.revision + 1 })),

  log: [],
  pushLog: (e) => set((s) => ({ log: [...s.log.slice(-300), e] })),
  logPaused: false,
  setLogPaused: (v) => set({ logPaused: v }),
}));
