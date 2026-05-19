"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

export type KpiRow = {
  year_month: string;
  ppm_external: number;
  otd_pct: number;
  audit_score: number;
  scar_count: number;
};

type ChartConfig = {
  key: keyof KpiRow;
  label: string;
  color: string;
  unit?: string;
  reference?: number;
  referenceLabel?: string;
  invertRisk?: boolean;
};

const CHARTS: ChartConfig[] = [
  { key: "ppm_external", label: "PPM (External)", color: "#f87171", unit: "", reference: 200, referenceLabel: "200 threshold" },
  { key: "otd_pct",      label: "On-Time Delivery %", color: "#60a5fa", unit: "%", reference: 95, referenceLabel: "95% target", invertRisk: true },
  { key: "audit_score",  label: "Audit Score", color: "#34d399", unit: "", reference: 75, referenceLabel: "75 pass", invertRisk: true },
  { key: "scar_count",   label: "SCARs / month", color: "#fb923c", unit: "" },
];

const fmt = (v: number, unit: string) => `${v.toFixed(1)}${unit}`;

function MiniChart({ data, cfg }: { data: KpiRow[]; cfg: ChartConfig }) {
  const last12 = data.slice(-12);
  const vals = last12.map((r) => Number(r[cfg.key]));
  const minY = Math.min(...vals);
  const maxY = Math.max(...vals);
  const pad = (maxY - minY) * 0.15 || 5;

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
      <p className="text-xs font-medium text-zinc-400 mb-3 uppercase tracking-wide">
        {cfg.label}
      </p>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={last12} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <XAxis
            dataKey="year_month"
            tick={{ fontSize: 9, fill: "#52525b" }}
            tickFormatter={(v) => String(v).slice(2, 7)}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[minY - pad, maxY + pad]}
            tick={{ fontSize: 9, fill: "#52525b" }}
            tickLine={false}
            axisLine={false}
            width={32}
            tickFormatter={(v) => fmt(v, cfg.unit ?? "")}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 11 }}
            labelStyle={{ color: "#a1a1aa" }}
            formatter={(v) => [fmt(Number(v ?? 0), cfg.unit ?? ""), cfg.label]}
          />
          {cfg.reference !== undefined && (
            <ReferenceLine
              y={cfg.reference}
              stroke="#475569"
              strokeDasharray="3 3"
              label={{ value: cfg.referenceLabel, position: "insideTopRight", fontSize: 9, fill: "#475569" }}
            />
          )}
          <Line
            type="monotone"
            dataKey={cfg.key as string}
            stroke={cfg.color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3, fill: cfg.color }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function KpiCharts({ kpis }: { kpis: KpiRow[] }) {
  if (!kpis.length) {
    return <p className="text-zinc-500 text-sm">No KPI history available.</p>;
  }
  return (
    <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
      {CHARTS.map((cfg) => (
        <MiniChart key={cfg.key} data={kpis} cfg={cfg} />
      ))}
    </div>
  );
}
