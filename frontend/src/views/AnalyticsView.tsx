/**
 * Analytics: what a traffic engineer would actually ask for.
 *
 * Flow, mean speed, the 85th percentile, headway distribution and the
 * fundamental diagram. The 85th percentile is given prominence over the mean
 * because it is the figure road design uses; the mean is the one that looks
 * reassuring and decides nothing.
 */

import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../lib/api";
import { kmph } from "../lib/format";
import { Empty, ErrorNote, Panel, Stat, Table } from "../components/primitives";

const AXIS = { stroke: "var(--text-muted)", fontSize: 10 };
const GRID = "var(--border-subtle)";

export function AnalyticsView({ loaded }: { loaded: boolean }) {
  const analytics = useQuery({ queryKey: ["analytics"], queryFn: api.analytics, enabled: loaded });

  if (!loaded) return <Empty message="Load and analyse a clip to see analytics." />;
  if (analytics.isError) return <ErrorNote message={String(analytics.error)} />;
  if (!analytics.data) return <p className="text-[13px] text-muted">Computing analytics…</p>;

  const a = analytics.data;
  const histogram = a.speed_histogram.counts.map((count, i) => ({
    speed: a.speed_histogram.edges[i]?.toFixed(0) ?? "",
    count,
  }));

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto">
      <div className="grid shrink-0 grid-cols-2 gap-2 md:grid-cols-5">
        <Stat label="Vehicles" value={String(a.total_count)} tone="accent" big />
        <Stat label="Duration" value={a.duration_s.toFixed(1)} unit="s" />
        <Stat
          label="85th percentile"
          value={kmph(a.speed.p85_kmph, 1)}
          unit="km/h"
          tone="accent"
          hint="The figure road design uses. Prominent because it decides something; the mean does not."
        />
        <Stat label="Mean speed" value={kmph(a.speed.mean_kmph, 1)} unit="km/h" />
        <Stat label="Median speed" value={kmph(a.speed.median_kmph, 1)} unit="km/h" />
      </div>

      <div className="rounded-panel border border-slow/40 bg-slow/10 px-3 py-2 text-[12px] leading-snug text-slow">
        {a.disclaimer}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="Speed distribution">
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogram} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="speed" tick={AXIS} unit=" km/h" />
                <YAxis tick={AXIS} allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: "var(--bg-raised)",
                    border: "1px solid var(--border-default)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="count" fill="var(--accent)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Flow over time">
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={a.intervals} margin={{ top: 4, right: 8, bottom: 4, left: -8 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="t_s" tick={AXIS} unit=" s" />
                <YAxis tick={AXIS} />
                <Tooltip
                  contentStyle={{
                    background: "var(--bg-raised)",
                    border: "1px solid var(--border-default)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="flow_veh_per_hour"
                  stroke="var(--accent)"
                  dot={false}
                  strokeWidth={1.8}
                />
                <Line
                  type="monotone"
                  dataKey="mean_speed_kmph"
                  stroke="var(--flow-free)"
                  dot={false}
                  strokeWidth={1.8}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-[11px] text-muted">
            Blue: flow in vehicles/hour. Green: mean speed in km/h.
          </p>
        </Panel>

        <Panel title="Fundamental diagram">
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 4, right: 8, bottom: 4, left: -8 }}>
                <CartesianGrid stroke={GRID} />
                <XAxis
                  type="number"
                  dataKey="density_veh_per_km"
                  name="density"
                  tick={AXIS}
                  unit=" v/km"
                />
                <YAxis
                  type="number"
                  dataKey="flow_veh_per_hour"
                  name="flow"
                  tick={AXIS}
                  unit=" v/h"
                />
                <Tooltip
                  cursor={{ stroke: GRID }}
                  contentStyle={{
                    background: "var(--bg-raised)",
                    border: "1px solid var(--border-default)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Scatter data={a.fundamental_diagram} fill="var(--accent)" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-[11px] leading-snug text-muted">
            Flow against density. Free flow rises along the left limb; congestion
            falls along the right. A clip this short will only populate part of it.
          </p>
        </Panel>

        <Panel title="Counts by class and lane">
          <div className="grid grid-cols-2 gap-4">
            <BreakdownList title="Class" data={a.counts_by_class} />
            <BreakdownList title="Lane" data={a.counts_by_lane} />
          </div>
        </Panel>
      </div>

      <Panel title="Vehicles">
        <Table
          columns={[
            { key: "track_id", label: "Track" },
            { key: "lane_id", label: "Lane" },
            { key: "direction", label: "Dir" },
            { key: "time_s", label: "t (s)" },
            { key: "speed_kmph", label: "km/h" },
            { key: "vehicle_class", label: "Class" },
            { key: "colour", label: "Colour" },
            { key: "length_m", label: "L (m)" },
          ]}
          numeric={["track_id", "lane_id", "time_s", "speed_kmph", "length_m"]}
          rows={a.vehicles}
        />
      </Panel>

      {a.flagged.length ? (
        <Panel
          title="Flagged for review"
          badge={<span className="num text-[11px] text-slow">{a.flagged.length}</span>}
        >
          <p className="mb-2 text-[12px] text-muted">
            Above the configured limit of {kmph(a.speed.limit_kmph, 0)} km/h. Flagged for
            review only — never a violation, and carrying no evidentiary standing.
          </p>
          <Table
            columns={[
              { key: "track_id", label: "Track" },
              { key: "lane_id", label: "Lane" },
              { key: "time_s", label: "t (s)" },
              { key: "speed_kmph", label: "km/h" },
            ]}
            numeric={["track_id", "lane_id", "time_s", "speed_kmph"]}
            rows={a.flagged}
          />
        </Panel>
      ) : null}

      <div className="pb-2">
        <a
          href="/api/analytics/csv"
          className="text-[12.5px] text-accent hover:underline"
          download
        >
          Download the per-vehicle table as CSV →
        </a>
      </div>
    </div>
  );
}

function BreakdownList({ title, data }: { title: string; data: Record<string, number> }) {
  const entries = Object.entries(data);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  return (
    <div>
      <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">
        {title}
      </h4>
      <ul className="space-y-1.5">
        {entries.map(([k, v]) => (
          <li key={k}>
            <div className="flex justify-between text-[12px]">
              <span className="text-secondary">{k}</span>
              <span className="num text-primary">{v}</span>
            </div>
            <div className="mt-0.5 h-[3px] overflow-hidden rounded bg-raised">
              <div className="h-full bg-accent" style={{ width: `${(v / max) * 100}%` }} />
            </div>
          </li>
        ))}
        {entries.length === 0 ? <li className="text-[12px] text-muted">none</li> : null}
      </ul>
    </div>
  );
}
