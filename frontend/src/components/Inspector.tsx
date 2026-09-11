/**
 * The inspector: what the pipeline currently believes about each vehicle.
 *
 * The list is the evidence, so it shows both speed routes side by side for
 * every track rather than one "the" speed. Route A differences the ground point
 * through the road homography; Route B measures where texture went. Seeing the
 * two disagree by 19 km/h on a specific vehicle is what makes the comparison
 * concrete instead of theoretical - and the size of the gap is itself the
 * diagnostic, because the routes fail in independent ways.
 */

import type { TrackSnapshot } from "../lib/api";
import { kmph, metres } from "../lib/format";
import { useStore } from "../lib/store";
import { Stat } from "./primitives";

// Tailwind generates classes by scanning the source, so an interpolated class
// name like `text-${tone}` would simply not exist in the build. The mapping is
// written out.
const SPEED_TONE = {
  stop: "text-stop",
  slow: "text-slow",
  free: "text-free",
} as const;

function tone(speed: number, limit = 60): keyof typeof SPEED_TONE {
  if (speed < 12) return "stop";
  if (speed > limit) return "slow";
  return "free";
}

export function Inspector() {
  const telemetry = useStore((s) => s.telemetry);
  const selected = useStore((s) => s.selectedTrack);
  const selectTrack = useStore((s) => s.selectTrack);

  const tracks = (telemetry?.tracks ?? []).filter((t) => t.state !== "lost");
  const track = tracks.find((t) => t.track_id === selected) ?? null;

  return (
    <div className="flex min-h-0 flex-col">
      <div className="grid grid-cols-2 gap-2 border-b border-subtle p-3">
        <Stat label="Counted" value={String(telemetry?.total_count ?? 0)} tone="accent" big />
        <Stat label="Active tracks" value={String(tracks.length)} big />
        <Stat label="Blobs this frame" value={String(telemetry?.blobs ?? 0)} />
        <Stat
          label="Frame time"
          value={telemetry ? telemetry.frame_ms.toFixed(0) : "—"}
          unit="ms"
          hint="Wall time for the whole nine-stage pipeline on this frame."
        />
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {tracks.length === 0 ? (
          <p className="p-3 text-[12px] text-muted">
            No active tracks on this frame. Play the clip, or step forward past the
            background warm-up.
          </p>
        ) : (
          <ul className="divide-y divide-subtle">
            {tracks.map((t) => (
              <TrackRow
                key={t.track_id}
                track={t}
                selected={t.track_id === selected}
                onSelect={() => selectTrack(t.track_id === selected ? null : t.track_id)}
              />
            ))}
          </ul>
        )}
      </div>

      {track ? <TrackDetail track={track} /> : null}
    </div>
  );
}

function TrackRow({
  track,
  selected,
  onSelect,
}: {
  track: TrackSnapshot;
  selected: boolean;
  onSelect: () => void;
}) {
  const t = tone(track.speed_kmph_track);
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        className={`flex w-full items-center gap-2 px-3 py-1.5 text-left transition-colors duration-fast ${
          selected ? "bg-accent-dim" : "hover:bg-raised"
        }`}
      >
        <span className="num w-8 shrink-0 text-[11px] text-muted">#{track.track_id}</span>
        <span className="min-w-0 flex-1 truncate text-[12px] text-secondary">
          {track.vehicle_class} · {track.colour_name}
          {track.lane_id != null ? ` · lane ${track.lane_id}` : ""}
        </span>
        <span className={`num shrink-0 text-[12.5px] font-semibold ${SPEED_TONE[t]}`}>
          {kmph(track.speed_kmph_track, 0)}
        </span>
        <span className="text-[10px] text-muted">km/h</span>
      </button>
    </li>
  );
}

function TrackDetail({ track }: { track: TrackSnapshot }) {
  const row = (k: string, v: string, hint?: string) => (
    <div className="flex justify-between gap-3" title={hint}>
      <span className="text-muted">{k}</span>
      <span className="num text-primary">{v}</span>
    </div>
  );
  return (
    <div className="border-t border-subtle bg-raised p-3 text-[11.5px]">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
        Track #{track.track_id}
      </div>
      <div className="space-y-1">
        {row("state", track.state)}
        {row("direction", track.direction)}
        {row("hits / age", `${track.hits} / ${track.age_frames}`)}
        {row(
          "world (m)",
          track.world_pos_m ? track.world_pos_m.map((v) => v.toFixed(2)).join(", ") : "—",
          "Position of the ground point on the road plane.",
        )}
        {row(
          "footprint (m)",
          track.footprint_m ? track.footprint_m.map((v) => metres(v, 2)).join(" × ") : "—",
          "Length × width × height in metres, from the homography. This, not appearance, decides the class.",
        )}
      </div>

      {/* The two routes, always together. */}
      <div className="mt-2.5 grid grid-cols-2 gap-2">
        <div className="rounded-ctl border border-subtle bg-surface px-2 py-1.5">
          <div className="text-[9.5px] uppercase tracking-[0.08em] text-muted">Route A ground pt</div>
          <div className="num text-[14px] font-semibold text-primary">
            {kmph(track.speed_kmph_track)}
            <span className="ml-1 text-[10px] font-normal text-muted">km/h</span>
          </div>
        </div>
        <div className="rounded-ctl border border-accent bg-accent-dim px-2 py-1.5">
          <div className="text-[9.5px] uppercase tracking-[0.08em] text-accent">
            Route B block match
          </div>
          <div className="num text-[14px] font-semibold text-accent">
            {kmph(track.speed_kmph_blockmatch)}
            <span className="ml-1 text-[10px] font-normal text-muted">km/h</span>
          </div>
        </div>
      </div>
      <p className="mt-2 text-[10.5px] leading-snug text-muted">
        Measured, not certified. Over-limit rows are flagged for review only and carry
        no evidentiary standing.
      </p>
    </div>
  );
}
