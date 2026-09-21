/**
 * The viewport: the annotated video, its transport, and the pixel probe.
 *
 * Frames arrive as MJPEG on a plain <img>. That is deliberate — the browser's
 * own image decoder is doing the work, there is no codec negotiation, and each
 * frame is an independent JPEG the pipeline just produced. The overlay is drawn
 * server-side into the same JPEG so that what the user sees is exactly what the
 * pipeline saw; an overlay drawn client-side from telemetry would drift by one
 * frame and nobody would notice.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, frameUrl, streamUrl } from "../lib/api";
import { engineImage, IN_BROWSER, useApiImage } from "../lib/engine";
import { useStore } from "../lib/store";
import { Button } from "./primitives";

const LAYER_LABELS: [string, string][] = [
  ["boxes", "Boxes"],
  ["ground_points", "Ground points"],
  ["trails", "World trails"],
  ["labels", "Labels"],
  ["lanes", "Lanes"],
  ["counting_lines", "Counting lines"],
  ["mask", "Foreground mask"],
  ["shadow", "Shadow pixels"],
  ["blobs", "Blob outlines"],
];

interface ProbeResult {
  x: number;
  y: number;
  rgb: number[];
  hsv: number[];
  background_rgb: number[];
  difference: number;
  threshold: number;
  [key: string]: unknown;
}

export function Viewport({ frames, live = true }: { frames: number; live?: boolean }) {
  const layers = useStore((s) => s.layers);
  const toggleLayer = useStore((s) => s.toggleLayer);
  const telemetry = useStore((s) => s.telemetry);
  const pushLog = useStore((s) => s.pushLog);
  const revision = useStore((s) => s.revision);

  const [playing, setPlaying] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [nonce, setNonce] = useState(() => Date.now());
  const imgRef = useRef<HTMLImageElement>(null);

  // The stream URL carries the layer flags, so a layer toggle has to restart
  // the stream. The nonce is what forces the browser to drop the old one
  // instead of reusing a connection that is still sending the previous layers.
  useEffect(() => setNonce(Date.now()), [layers]);

  // Against the server, the stream advances the cursor and the telemetry says
  // where it is. In the browser build the page advances it, so the page's own
  // cursor is the truth and telemetry trails it by a poll.
  const index = IN_BROWSER ? cursor : telemetry?.frame_index ?? cursor;

  const send = useCallback(
    async (body: Record<string, unknown>) => {
      try {
        const result = await api.transport(body);
        setPlaying(result.playing);
        setCursor(result.cursor);
        setSpeed(result.speed);
      } catch (error) {
        pushLog({ t: Date.now(), source: "transport", message: String(error) });
      }
    },
    [pushLog],
  );

  const onProbe = useCallback(
    async (event: React.MouseEvent<HTMLImageElement>) => {
      const img = imgRef.current;
      if (!img || !img.naturalWidth) return;
      const rect = img.getBoundingClientRect();
      // The displayed image is letterboxed inside its box by object-contain, so
      // the click has to be mapped through the *rendered* rectangle, not the
      // element's. Getting this wrong probes a pixel a few rows away, which is
      // exactly the kind of error that looks like a pipeline bug.
      const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
      const drawnW = img.naturalWidth * scale;
      const drawnH = img.naturalHeight * scale;
      const x = Math.round((event.clientX - rect.left - (rect.width - drawnW) / 2) / scale);
      const y = Math.round((event.clientY - rect.top - (rect.height - drawnH) / 2) / scale);
      if (x < 0 || y < 0 || x >= img.naturalWidth || y >= img.naturalHeight) return;
      try {
        setProbe((await api.probe(index, x, y)) as unknown as ProbeResult);
      } catch (error) {
        pushLog({ t: Date.now(), source: "probe", message: String(error) });
      }
    },
    [index, pushLog],
  );

  const serverSrc = useMemo(
    () => (live && playing ? streamUrl(layers, nonce) : frameUrl(index, layers, revision)),
    [live, playing, layers, nonce, index, revision],
  );

  // Browser build: a still frame comes through the engine; playback is a loop
  // here that steps the cursor, renders that frame, and only then steps again.
  // Stepping on a timer instead would queue frames faster than the engine can
  // draw them, and the video would fall further behind the controls each second.
  const stillSrc = useApiImage(IN_BROWSER && !playing ? frameUrl(index, layers, revision) : null);
  const [playSrc, setPlaySrc] = useState<string | undefined>(undefined);
  useEffect(() => {
    if (!IN_BROWSER || !playing) return undefined;
    let stopped = false;
    let shown: string | undefined;
    const run = async () => {
      // Pressing Play on the last frame starts again from the top, as the
      // server's stream does, rather than stopping on the spot.
      const at = await api.transport({});
      if (at.cursor >= at.frames - 1) await api.transport({ cursor: 0 });
      while (!stopped) {
        const started = performance.now();
        const t = await api.transport({ step: 1 });
        if (stopped) break;
        setCursor(t.cursor);
        const next = await engineImage(frameUrl(t.cursor, layers, revision));
        if (stopped) {
          if (next) URL.revokeObjectURL(next);
          break;
        }
        if (next) {
          setPlaySrc(next);
          if (shown) URL.revokeObjectURL(shown);
          shown = next;
        }
        if (t.cursor >= t.frames - 1) {
          await send({ playing: false });
          break;
        }
        const budget = 1000 / 25 / Math.max(t.speed, 0.05);
        const wait = budget - (performance.now() - started);
        if (wait > 0) await new Promise((r) => window.setTimeout(r, wait));
      }
    };
    void run();
    return () => {
      stopped = true;
    };
    // `send` is stable; `layers` restarts the loop so a toggle applies at once.
  }, [playing, layers, send, revision]);

  const src = IN_BROWSER ? (playing ? playSrc ?? stillSrc : stillSrc) : serverSrc;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-panel border border-subtle bg-black">
        <img
          ref={imgRef}
          src={src}
          alt="Annotated frame"
          onClick={onProbe}
          className="max-h-full max-w-full cursor-crosshair object-contain"
        />
        {probe ? (
          <ProbeCard probe={probe} onClose={() => setProbe(null)} />
        ) : null}
      </div>

      {/* transport */}
      <div className="flex flex-wrap items-center gap-2 rounded-panel border border-subtle bg-surface px-3 py-2">
        <Button size="sm" onClick={() => void send({ step: -1 })} title="Previous frame">
          ⏮
        </Button>
        <Button
          size="sm"
          variant={playing ? "default" : "primary"}
          onClick={() => void send({ playing: !playing })}
        >
          {playing ? "Pause" : "Play"}
        </Button>
        <Button size="sm" onClick={() => void send({ step: 1 })} title="Next frame">
          ⏭
        </Button>
        <input
          type="range"
          min={0}
          max={Math.max(0, frames - 1)}
          value={index}
          aria-label="Frame"
          onChange={(e) => void send({ cursor: Number(e.target.value) })}
          className="mx-2 min-w-[120px] flex-1"
        />
        <span className="num w-[96px] text-right text-[12px] text-secondary">
          {index + 1} / {frames}
        </span>
        <select
          value={String(speed)}
          aria-label="Playback speed"
          onChange={(e) => void send({ speed: Number(e.target.value) })}
          className="h-7 rounded-ctl border border-line bg-raised px-1.5 text-[12px] text-primary"
        >
          {[0.25, 0.5, 1, 2, 4].map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </div>

      {/* layers */}
      <div className="flex flex-wrap gap-1.5 rounded-panel border border-subtle bg-surface px-3 py-2">
        <span className="mr-1 self-center text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
          Layers
        </span>
        {LAYER_LABELS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            aria-pressed={layers[key] ?? false}
            onClick={() => toggleLayer(key)}
            className={`rounded-ctl border px-2 py-0.5 text-[11.5px] transition-colors duration-fast ${
              layers[key]
                ? "border-accent bg-accent-dim text-accent"
                : "border-line bg-raised text-muted hover:text-secondary"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}

function ProbeCard({ probe, onClose }: { probe: ProbeResult; onClose: () => void }) {
  const passed = probe.difference >= probe.threshold;
  const row = (k: string, v: string) => (
    <div className="flex justify-between gap-4">
      <span className="text-muted">{k}</span>
      <span className="num text-primary">{v}</span>
    </div>
  );
  return (
    <div className="absolute right-3 top-3 w-[224px] rounded-panel border border-line bg-surface/95 p-3 text-[11.5px] shadow-overlay backdrop-blur">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
          Pixel probe
        </span>
        <button type="button" onClick={onClose} className="text-muted hover:text-primary">
          ✕
        </button>
      </div>
      <div className="space-y-1">
        {row("position", `${probe.x}, ${probe.y}`)}
        {row("rgb", probe.rgb.join(", "))}
        {row("background", probe.background_rgb.join(", "))}
        {row("hsv", probe.hsv.join(", "))}
        {row("difference", probe.difference.toFixed(2))}
        {row("threshold", probe.threshold.toFixed(2))}
        {probe.world_m ? row("world (m)", (probe.world_m as number[]).join(", ")) : null}
      </div>
      {/* The decision, not just the numbers. Reporting a difference and a
          threshold and leaving the comparison to the reader is half a probe. */}
      <div
        className={`mt-2 rounded-ctl px-2 py-1 text-[11px] font-medium ${
          passed ? "bg-free/15 text-free" : "bg-raised text-muted"
        }`}
      >
        {passed ? "foreground — difference exceeds threshold" : "background — below threshold"}
      </div>
    </div>
  );
}
