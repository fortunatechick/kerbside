/**
 * Calibration: four clicks, then metres.
 *
 * The important design decision here is that the verdict is not a number. A
 * reprojection RMSE of 0.3 px looks excellent and says nothing about whether the
 * rectangle you clicked was really 3.5 m wide — four points always fit a
 * homography exactly, so the residual is structurally zero. The view therefore
 * leads with the *implied camera* (height, tilt, focal length) and a plain
 * statement of whether those numbers are physically plausible, because that is
 * the check that actually catches a wrong calibration.
 */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, frameUrl, type CalibrationInfo } from "../lib/api";
import { metres } from "../lib/format";
import { Button, Empty, ErrorNote, Panel, Stat } from "../components/primitives";

type Pt = [number, number];

export function CalibrationView({ loaded }: { loaded: boolean }) {
  const client = useQueryClient();
  const info = useQuery({ queryKey: ["calibration"], queryFn: api.calibration });

  const [points, setPoints] = useState<Pt[]>([]);
  const [widthM, setWidthM] = useState(3.5);
  const [lengthM, setLengthM] = useState(12.0);
  const [lanes, setLanes] = useState(4);
  const [natural, setNatural] = useState<[number, number]>([1280, 720]);
  const imgRef = useRef<HTMLImageElement>(null);

  const calibrate = useMutation({
    mutationFn: () =>
      api.calibrate({
        points_px: points,
        rect_width_m: widthM,
        rect_length_m: lengthM,
        lane_count: lanes,
        lane_width_m: 3.5,
      }),
    onSuccess: () => void client.invalidateQueries(),
  });

  const grid = useQuery({
    queryKey: ["grid", info.data?.calibrated],
    queryFn: api.grid,
    enabled: Boolean(info.data?.calibrated),
  });

  if (!loaded) {
    return <Empty message="Load a source first — calibration needs a frame to click on." />;
  }

  const addPoint = (event: React.MouseEvent<HTMLImageElement>) => {
    const img = imgRef.current;
    if (!img?.naturalWidth) return;
    const rect = img.getBoundingClientRect();
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
    const x = (event.clientX - rect.left - (rect.width - img.naturalWidth * scale) / 2) / scale;
    const y = (event.clientY - rect.top - (rect.height - img.naturalHeight * scale) / 2) / scale;
    setPoints((p) => (p.length >= 4 ? [[x, y]] : [...p, [x, y]]));
  };

  return (
    <div className="grid h-full min-h-0 grid-cols-1 gap-3 overflow-auto lg:grid-cols-[1.5fr_1fr]">
      <Panel
        title="Reference rectangle"
        badge={
          <span className="num text-[11px] text-muted">{points.length} / 4 points</span>
        }
        actions={
          <>
            <Button size="sm" variant="ghost" onClick={() => setPoints([])}>
              Clear
            </Button>
            <Button
              size="sm"
              variant="primary"
              disabled={points.length !== 4 || calibrate.isPending}
              onClick={() => calibrate.mutate()}
            >
              Calibrate
            </Button>
          </>
        }
        className="min-h-[320px]"
      >
        <div className="relative flex h-full min-h-[280px] items-center justify-center overflow-hidden rounded-panel bg-black">
          <img
            ref={imgRef}
            src={frameUrl(0, { boxes: false, lanes: false, counting_lines: false })}
            alt="Frame for calibration"
            onClick={addPoint}
            onLoad={(e) =>
              setNatural([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])
            }
            className="max-h-full max-w-full cursor-crosshair object-contain"
          />
          <Markers points={points} grid={grid.data?.polylines} natural={natural} img={imgRef} />
        </div>
        <p className="mt-2 text-[11.5px] leading-snug text-muted">
          Click the four corners of a rectangle you know the dimensions of, on the
          road surface, in order: near-left, near-right, far-right, far-left. Lane
          width across, dashed-line pitch along. Nothing above the horizon can be
          used — a point above it maps to a road position behind the camera.
        </p>
      </Panel>

      <div className="flex min-h-0 flex-col gap-3">
        <Panel title="Known dimensions">
          <div className="space-y-3">
            <NumberField label="Rectangle width (across road)" unit="m" value={widthM} onChange={setWidthM} />
            <NumberField label="Rectangle length (along road)" unit="m" value={lengthM} onChange={setLengthM} />
            <NumberField label="Lane count" unit="" value={lanes} onChange={(v) => setLanes(Math.round(v))} step={1} />
          </div>
          <Suggestions suggestions={info.data?.suggestions ?? []} />
        </Panel>

        <Verdict info={info.data} error={calibrate.error} />
      </div>
    </div>
  );
}

function Markers({
  points,
  grid,
  natural,
  img,
}: {
  points: Pt[];
  grid?: [number, number][][];
  natural: [number, number];
  img: React.RefObject<HTMLImageElement>;
}) {
  // Redraw on resize: the overlay is in the image's pixel space and scaled by
  // the SVG viewBox, so it tracks object-contain letterboxing for free.
  const [, force] = useState(0);
  useEffect(() => {
    const onResize = () => force((n) => n + 1);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const element = img.current;
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  const scale = Math.min(rect.width / natural[0], rect.height / natural[1]);
  const w = natural[0] * scale;
  const h = natural[1] * scale;

  return (
    <svg
      viewBox={`0 0 ${natural[0]} ${natural[1]}`}
      style={{ width: w, height: h }}
      className="pointer-events-none absolute"
    >
      {grid?.map((line, i) => (
        <polyline
          key={i}
          points={line.map((p) => p.join(",")).join(" ")}
          fill="none"
          stroke="var(--accent)"
          strokeOpacity={0.45}
          strokeWidth={1.5}
        />
      ))}
      {points.length === 4 ? (
        <polygon
          points={points.map((p) => p.join(",")).join(" ")}
          fill="var(--accent)"
          fillOpacity={0.12}
          stroke="var(--accent)"
          strokeWidth={2}
        />
      ) : null}
      {points.map((p, i) => (
        <g key={i}>
          <circle cx={p[0]} cy={p[1]} r={6} fill="var(--accent)" />
          <text x={p[0] + 10} y={p[1] - 8} fill="var(--accent)" fontSize={16} fontWeight={700}>
            {i + 1}
          </text>
        </g>
      ))}
    </svg>
  );
}

function NumberField({
  label,
  unit,
  value,
  onChange,
  step = 0.1,
}: {
  label: string;
  unit: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[12px] font-medium text-secondary">{label}</span>
      <div className="flex items-center gap-2">
        <input
          type="number"
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="num h-8 w-full rounded-ctl border border-line bg-raised px-2 text-[13px] text-primary"
        />
        {unit ? <span className="text-[12px] text-muted">{unit}</span> : null}
      </div>
    </label>
  );
}

function Suggestions({
  suggestions,
}: {
  suggestions: { name: string; nominal_m: number; range_m: number[]; caution: string }[];
}) {
  if (!suggestions.length) return null;
  return (
    <div className="mt-4 border-t border-subtle pt-3">
      <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
        Reference distances
      </h4>
      <ul className="space-y-2">
        {suggestions.map((s) => (
          <li key={s.name} className="text-[11.5px] leading-snug">
            <span className="font-medium text-primary">{s.name}</span>{" "}
            <span className="num text-accent">{metres(s.nominal_m, 2)} m</span>{" "}
            <span className="num text-muted">
              ({metres(s.range_m[0] ?? 0, 1)}–{metres(s.range_m[1] ?? 0, 1)})
            </span>
            {/* The caution is the whole value of this list. A nominal figure used
                without checking the local standard is a confident wrong answer. */}
            <div className="text-muted">{s.caution}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Verdict({ info, error }: { info?: CalibrationInfo; error: unknown }) {
  if (error) return <Panel title="Calibration"><ErrorNote message={String(error)} /></Panel>;
  if (!info?.calibrated) {
    return (
      <Panel title="Calibration">
        <p className="text-[12.5px] text-secondary">
          Not calibrated. Until it is, every distance, speed and size in this
          console is in pixels and frames, and none of it is physical.
        </p>
      </Panel>
    );
  }

  const camera = info.camera;
  return (
    <Panel title="Calibration" badge={<span className="num text-[11px] text-free">valid</span>} scroll>
      <div className="mb-3 grid grid-cols-3 gap-2">
        <Stat label="Camera height" value={metres(camera?.height_m ?? null, 2)} unit="m" />
        <Stat label="Tilt below horizontal" value={metres(camera?.tilt_deg ?? null, 2)} unit="°" />
        <Stat label="Focal length" value={metres(camera?.focal_px ?? null, 0)} unit="px" />
      </div>

      <p className="mb-2 text-[12.5px] leading-relaxed text-secondary">{info.verdict}</p>

      {info.exactly_determined ? (
        <div className="mb-2 rounded-panel border border-slow/40 bg-slow/10 px-3 py-2 text-[11.5px] leading-snug text-slow">
          Four points determine a homography exactly, so the reprojection residual
          is structurally zero and proves nothing. The plausibility of the implied
          camera above, and a tape measure, are the real checks.
        </div>
      ) : (
        <p className="num mb-2 text-[12px] text-muted">
          reprojection RMSE {metres(info.rmse_px ?? null, 3)} px
        </p>
      )}

      {camera && !camera.plausible ? (
        <ErrorNote message={camera.reasons.join(" ")} />
      ) : null}

      {info.warnings?.length ? (
        <ul className="mt-2 space-y-1">
          {info.warnings.map((w, i) => (
            <li key={i} className="text-[11.5px] leading-snug text-slow">
              {w}
            </li>
          ))}
        </ul>
      ) : null}

      {info.covered_y_range_m ? (
        <p className="num mt-2 text-[11.5px] text-muted">
          usable road span {metres(info.covered_y_range_m[0], 1)}–
          {metres(info.covered_y_range_m[1], 1)} m
        </p>
      ) : null}
    </Panel>
  );
}
