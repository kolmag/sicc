"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

export type ShapFeature = {
  feature: string;
  value: number;
};

const FMT: Record<string, string> = {
  ppm_external: "PPM",
  otd_pct: "OTD%",
  audit_score: "Audit",
  scar_count: "SCARs",
  cost_of_poor_quality_eur: "COPQ",
  ppap_first_time_pass_pct: "PPAP FTP",
  ca_closure_rate_pct: "CA Closure",
  oqd_pct: "OQD%",
  spend_tier_enc: "Spend Tier",
  strat_imp_enc: "Strategic Imp",
  qual_status_enc: "Qual Status",
  region_risk_enc: "Region Risk",
  single_source_int: "Single Source",
  years_active: "Yrs Active",
  annual_spend_eur: "Annual Spend",
};

function fmtFeature(name: string): string {
  let s = name;
  for (const [k, v] of Object.entries(FMT)) s = s.replace(k, v);
  s = s.replace(/_3m$/, " 3m").replace(/_6m$/, " 6m").replace(/_12m$/, " 12m")
       .replace(/_trend$/, " trend").replace(/_std_12m$/, " σ12m")
       .replace(/_worst_3m$/, " worst").replace(/_/g, " ");
  return s.length > 28 ? s.slice(0, 28) + "…" : s;
}

export function ShapChart({ features }: { features: ShapFeature[] }) {
  if (!features.length) {
    return <p className="text-zinc-500 text-sm">No SHAP data available.</p>;
  }

  const data = [...features]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .map((f) => ({ ...f, label: fmtFeature(f.feature) }));

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
      <p className="text-xs font-medium text-zinc-400 mb-1 uppercase tracking-wide">
        SHAP — top feature contributions → RED risk
      </p>
      <p className="text-[10px] text-zinc-600 mb-3">
        Positive = pushes toward RED · Negative = reduces RED risk
      </p>
      <ResponsiveContainer width="100%" height={data.length * 28 + 20}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 16, bottom: 0, left: 8 }}
        >
          <XAxis
            type="number"
            tick={{ fontSize: 9, fill: "#52525b" }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={110}
            tick={{ fontSize: 10, fill: "#a1a1aa" }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 11 }}
            formatter={(v) => [Number(v ?? 0).toFixed(4), "SHAP value"]}
          />
          <ReferenceLine x={0} stroke="#3f3f46" />
          <Bar dataKey="value" radius={[0, 3, 3, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.value > 0 ? "#f87171" : "#34d399"} fillOpacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
