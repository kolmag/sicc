"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export type FeatureRow = { feature: string; importance: number };

const FMT: Record<string, string> = {
  ppm_external: "PPM ext",
  ppm_internal: "PPM int",
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

function fmt(name: string) {
  let s = name;
  for (const [k, v] of Object.entries(FMT)) s = s.replace(k, v);
  return s
    .replace(/_3m$/, " 3m").replace(/_6m$/, " 6m").replace(/_12m$/, " 12m")
    .replace(/_trend$/, " trend").replace(/_std_12m$/, " σ12m")
    .replace(/_worst_3m$/, " worst").replace(/_/g, " ")
    .slice(0, 26);
}

export function FeatureImportanceChart({ features }: { features: FeatureRow[] }) {
  const data = features.map((f) => ({ label: fmt(f.feature), importance: f.importance }));

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
      <p className="text-xs font-medium text-zinc-400 uppercase tracking-wide mb-0.5">
        Global feature importance
      </p>
      <p className="text-[10px] text-zinc-600 mb-3">Mean |SHAP| across all suppliers → RED risk driver</p>
      <ResponsiveContainer width="100%" height={data.length * 24 + 16}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 12, bottom: 0, left: 0 }}>
          <XAxis
            type="number"
            tick={{ fontSize: 9, fill: "#52525b" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v.toFixed(3)}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={100}
            tick={{ fontSize: 10, fill: "#a1a1aa" }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 11 }}
            formatter={(v: unknown) => [Number(v).toFixed(5), "Mean |SHAP|"]}
          />
          <Bar dataKey="importance" fill="#3b82f6" fillOpacity={0.8} radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
