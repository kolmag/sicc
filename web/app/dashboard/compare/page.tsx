import Link from "next/link";
import { notFound } from "next/navigation";
import { RiskBadge } from "@/components/RiskBadge";
import { KpiCharts, type KpiRow } from "@/components/KpiCharts";
import { ShapChart, type ShapFeature } from "@/components/ShapChart";

const API = process.env.SICC_API_URL ?? "http://localhost:8000";

async function fetchComparison(ids: string[]) {
  try {
    const res = await fetch(`${API}/suppliers/compare?ids=${ids.join(",")}`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

type SupplierData = {
  supplier: Record<string, unknown>;
  kpis: KpiRow[];
  ml_prediction: string | null;
  ml_confidence: number | null;
  shap_top_features: ShapFeature[];
};

const META_ROWS: { label: string; key: string; format?: (v: unknown) => string }[] = [
  { label: "Country", key: "country" },
  { label: "Product family", key: "product_family" },
  { label: "Spend tier", key: "spend_tier" },
  { label: "Annual spend", key: "annual_spend_eur", format: (v) => `€${Number(v).toLocaleString()}` },
  { label: "Qualification", key: "qualification_status" },
  { label: "Single source", key: "single_source", format: (v) => String(v) === "True" ? "Yes" : "No" },
  { label: "Strategic importance", key: "strategic_importance" },
  { label: "Years active", key: "years_active" },
  { label: "Archetype", key: "archetype" },
  { label: "Composite score", key: "composite_risk_score", format: (v) => Number(v).toFixed(1) },
  { label: "PPM (3m avg)", key: "avg_ppm_3m", format: (v) => Number(v).toFixed(0) },
  { label: "OTD% (3m avg)", key: "avg_otd_3m", format: (v) => `${Number(v).toFixed(1)}%` },
  { label: "Audit score (3m avg)", key: "avg_audit_score_3m", format: (v) => Number(v).toFixed(1) },
  { label: "SCARs (3m avg)", key: "avg_scar_count_3m", format: (v) => Number(v).toFixed(1) },
  { label: "Recommended action", key: "recommended_action" },
];

function cellVal(s: Record<string, unknown>, key: string, format?: (v: unknown) => string): string {
  const v = s[key];
  if (v == null || v === "") return "—";
  return format ? format(v) : String(v);
}

function isWorst(values: string[], idx: number, key: string): boolean {
  const higherIsBad = ["avg_ppm_3m", "avg_scar_count_3m", "composite_risk_score"];
  const lowerIsBad = ["avg_otd_3m", "avg_audit_score_3m"];
  const nums = values.map(Number).filter((n) => !isNaN(n));
  if (nums.length < 2) return false;
  const n = Number(values[idx]);
  if (higherIsBad.includes(key)) return n === Math.max(...nums);
  if (lowerIsBad.includes(key)) return n === Math.min(...nums);
  return false;
}

function isBest(values: string[], idx: number, key: string): boolean {
  const higherIsBad = ["avg_ppm_3m", "avg_scar_count_3m", "composite_risk_score"];
  const lowerIsBad = ["avg_otd_3m", "avg_audit_score_3m"];
  const nums = values.map(Number).filter((n) => !isNaN(n));
  if (nums.length < 2) return false;
  const n = Number(values[idx]);
  if (higherIsBad.includes(key)) return n === Math.min(...nums);
  if (lowerIsBad.includes(key)) return n === Math.max(...nums);
  return false;
}

export default async function ComparePage({
  searchParams,
}: {
  searchParams: Promise<{ ids?: string }>;
}) {
  const { ids: idsParam } = await searchParams;
  const ids = (idsParam ?? "").split(",").map((s) => s.trim()).filter(Boolean).slice(0, 3);
  if (ids.length < 2) notFound();

  const data = await fetchComparison(ids);
  if (!data || !data.suppliers?.length) {
    return (
      <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100">
        <header className="border-b border-zinc-800 px-6 py-4">
          <Link href="/dashboard" className="text-zinc-500 hover:text-zinc-200 text-sm">← Dashboard</Link>
        </header>
        <main className="flex-1 flex items-center justify-center text-zinc-500 text-sm">
          Could not load comparison data. Make sure the API is running.
        </main>
      </div>
    );
  }

  const suppliers: SupplierData[] = data.suppliers;
  const cols = suppliers.length;

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center gap-4">
        <Link href="/dashboard" className="text-zinc-500 hover:text-zinc-200 text-sm transition-colors">
          ← Dashboard
        </Link>
        <div className="h-4 w-px bg-zinc-700" />
        <h1 className="text-base font-semibold">Supplier Comparison</h1>
        <span className="text-xs text-zinc-500">{cols} suppliers</span>
      </header>

      <main className="flex-1 px-6 py-6 space-y-10 max-w-7xl mx-auto w-full">

        {/* Header cards */}
        <div className={`grid gap-4 grid-cols-${cols}`} style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
          {suppliers.map(({ supplier, ml_prediction, ml_confidence }) => {
            const mismatch = ml_prediction && ml_prediction.toLowerCase() !== String(supplier.risk_label).toLowerCase();
            return (
              <div key={String(supplier.supplier_id)} className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-2">
                <Link
                  href={`/dashboard/${supplier.supplier_id}`}
                  className="text-sm font-semibold text-zinc-100 hover:text-white block truncate"
                  title={String(supplier.name)}
                >
                  {String(supplier.name)}
                </Link>
                <p className="text-xs text-zinc-500 font-mono">{String(supplier.supplier_id)}</p>
                <div className="flex items-center gap-2 flex-wrap">
                  <RiskBadge label={String(supplier.risk_label)} />
                  <span className="text-[10px] text-zinc-500">rule-based</span>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <RiskBadge label={ml_prediction} />
                  {ml_confidence !== null && (
                    <span className="text-[10px] text-zinc-500">{((ml_confidence ?? 0) * 100).toFixed(0)}% conf</span>
                  )}
                  {mismatch && <span className="text-[10px] text-amber-400">⚠ mismatch</span>}
                </div>
              </div>
            );
          })}
        </div>

        {/* Metric comparison table */}
        <section>
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">Side-by-side metrics</h2>
          <div className="rounded-xl border border-zinc-800 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-zinc-900 border-b border-zinc-800">
                  <th className="px-4 py-2.5 text-left text-xs font-medium text-zinc-400 w-44">Metric</th>
                  {suppliers.map(({ supplier }) => (
                    <th key={String(supplier.supplier_id)} className="px-4 py-2.5 text-left text-xs font-medium text-zinc-300 truncate max-w-48">
                      {String(supplier.name).split(" ").slice(0, 3).join(" ")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {META_ROWS.map(({ label, key, format }) => {
                  const rawVals = suppliers.map(({ supplier }) => String(supplier[key] ?? ""));
                  const fmtVals = suppliers.map(({ supplier }) => cellVal(supplier, key, format));
                  return (
                    <tr key={key} className="border-t border-zinc-800 hover:bg-zinc-800/30">
                      <td className="px-4 py-2 text-xs text-zinc-500">{label}</td>
                      {fmtVals.map((val, i) => {
                        const worst = isWorst(rawVals, i, key);
                        const best = isBest(rawVals, i, key);
                        return (
                          <td key={i} className={`px-4 py-2 text-sm tabular-nums ${worst ? "text-red-400" : best ? "text-emerald-400" : "text-zinc-200"}`}>
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-zinc-600 mt-1.5">
            <span className="text-emerald-500">Green</span> = best · <span className="text-red-500">Red</span> = worst (for KPI metrics)
          </p>
        </section>

        {/* KPI trend charts — one row per supplier */}
        <section>
          <h2 className="text-sm font-semibold text-zinc-300 mb-4">KPI trends (last 12 months)</h2>
          <div className="space-y-6">
            {suppliers.map(({ supplier, kpis }) => (
              <div key={String(supplier.supplier_id)}>
                <p className="text-xs text-zinc-400 font-medium mb-2">
                  {String(supplier.name)}
                </p>
                <KpiCharts kpis={[...kpis].reverse()} />
              </div>
            ))}
          </div>
        </section>

        {/* SHAP comparison */}
        <section>
          <h2 className="text-sm font-semibold text-zinc-300 mb-4">ML explainability — top SHAP features</h2>
          <div className={`grid gap-4`} style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
            {suppliers.map(({ supplier, shap_top_features }) => (
              <div key={String(supplier.supplier_id)}>
                <p className="text-xs text-zinc-500 mb-2">{String(supplier.name).split(" ").slice(0, 3).join(" ")}</p>
                <ShapChart features={shap_top_features} />
              </div>
            ))}
          </div>
        </section>

      </main>
    </div>
  );
}
