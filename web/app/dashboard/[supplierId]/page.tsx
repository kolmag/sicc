import Link from "next/link";
import { notFound } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RiskBadge } from "@/components/RiskBadge";
import { KpiCharts, type KpiRow } from "@/components/KpiCharts";
import { ShapChart, type ShapFeature } from "@/components/ShapChart";
import { ApqpGates } from "@/components/ApqpGates";
import { ExternalEvents } from "@/components/ExternalEvents";

const API = process.env.SICC_API_URL ?? "http://localhost:8000";

async function fetchSupplier(id: string) {
  try {
    const res = await fetch(`${API}/suppliers/${id}`, { cache: "no-store" });
    if (res.status === 404) return null;
    if (!res.ok) return { error: `API error ${res.status}` };
    return await res.json();
  } catch {
    return { error: "Cannot reach the SICC API. Start it with: uv run sicc-api" };
  }
}

function MetaItem({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-0.5">{label}</p>
      <p className="text-sm text-zinc-200">{value ?? "—"}</p>
    </div>
  );
}

function ClaimsTable({ claims }: { claims: Record<string, unknown>[] }) {
  if (!claims.length) return <p className="text-zinc-500 text-sm">No claims on record.</p>;
  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-xs text-zinc-300">
        <thead>
          <tr className="bg-zinc-900 text-zinc-400">
            {["Incident", "Date", "Category", "Status", "Bad Parts", "Chargeback €"].map((h) => (
              <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {claims.map((c, i) => (
            <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/40">
              <td className="px-3 py-2 font-mono text-zinc-400">{String(c.incident_number)}</td>
              <td className="px-3 py-2">{String(c.creation_date ?? "").slice(0, 10)}</td>
              <td className="px-3 py-2">{String(c.category ?? "—")}</td>
              <td className="px-3 py-2">{String(c.status ?? "—")}</td>
              <td className="px-3 py-2 text-right tabular-nums">{String(c.number_of_bad_parts ?? "—")}</td>
              <td className="px-3 py-2 text-right tabular-nums">
                {c.chargeback_value_eur != null
                  ? `€${Number(c.chargeback_value_eur).toFixed(0)}`
                  : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AuditsTable({ audits }: { audits: Record<string, unknown>[] }) {
  if (!audits.length) return <p className="text-zinc-500 text-sm">No audits on record.</p>;
  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-xs text-zinc-300">
        <thead>
          <tr className="bg-zinc-900 text-zinc-400">
            {["Date", "Type", "Remote", "Score", "Findings", "Highest Finding", "Status"].map((h) => (
              <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {audits.map((a, i) => (
            <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/40">
              <td className="px-3 py-2">{String(a.audit_date ?? "").slice(0, 10)}</td>
              <td className="px-3 py-2">{String(a.audit_type ?? "—")}</td>
              <td className="px-3 py-2">{String(a.is_remote) === "1" ? "Yes" : "No"}</td>
              <td className="px-3 py-2 text-right tabular-nums">{Number(a.audit_score).toFixed(1)}</td>
              <td className="px-3 py-2 text-right tabular-nums">{String(a.n_findings ?? "—")}</td>
              <td className="px-3 py-2">{String(a.highest_finding_type ?? "—")}</td>
              <td className="px-3 py-2">{String(a.status ?? "—")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default async function SupplierDetailPage({
  params,
}: {
  params: Promise<{ supplierId: string }>;
}) {
  const { supplierId } = await params;
  const data = await fetchSupplier(supplierId);
  if (!data) notFound();
  if ("error" in data) {
    return (
      <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100">
        <header className="border-b border-zinc-800 px-6 py-4">
          <Link href="/dashboard" className="text-zinc-500 hover:text-zinc-200 text-sm">
            ← Dashboard
          </Link>
        </header>
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-3">
            <p className="text-zinc-400">{data.error}</p>
            <code className="text-sm text-zinc-300 bg-zinc-800 px-2 py-1 rounded">
              uv run sicc-api
            </code>
          </div>
        </main>
      </div>
    );
  }

  const s = data.supplier as Record<string, unknown>;
  const kpis = data.kpis as KpiRow[];
  const claims = data.claims as Record<string, unknown>[];
  const audits = data.audits as Record<string, unknown>[];
  const shapFeatures = data.shap_top_features as ShapFeature[];
  const apqpProjects = data.apqp_projects as Record<string, unknown>[];
  const externalEvents = data.external_events as Record<string, unknown>[];
  const mlPrediction = data.ml_prediction as string | null;
  const mlConfidence = data.ml_confidence as number | null;

  const mismatch =
    mlPrediction !== null &&
    mlPrediction?.toLowerCase() !== String(s.risk_label).toLowerCase();

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center gap-4">
        <Link
          href="/dashboard"
          className="text-zinc-500 hover:text-zinc-200 text-sm transition-colors"
        >
          ← Dashboard
        </Link>
        <div className="h-4 w-px bg-zinc-700" />
        <h1 className="text-base font-semibold truncate">{String(s.name)}</h1>
        <span className="text-zinc-500 text-sm font-mono">{String(s.supplier_id)}</span>
      </header>

      <main className="flex-1 px-6 py-6 space-y-8 max-w-7xl mx-auto w-full">

        {/* Supplier meta card */}
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="pt-5 pb-5">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-x-6 gap-y-4">
              <div>
                <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-0.5">Rule-based risk</p>
                <RiskBadge label={String(s.risk_label)} />
              </div>
              <div>
                <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-0.5">ML prediction</p>
                <div className="flex items-center gap-1.5">
                  <RiskBadge label={mlPrediction} />
                  {mlConfidence !== null && (
                    <span className="text-[10px] text-zinc-500 tabular-nums">
                      {(mlConfidence * 100).toFixed(0)}%
                    </span>
                  )}
                  {mismatch && (
                    <span className="text-[10px] text-amber-400 font-medium">⚠ mismatch</span>
                  )}
                </div>
              </div>
              <MetaItem label="Country" value={String(s.country)} />
              <MetaItem label="Product family" value={String(s.product_family)} />
              <MetaItem label="Spend tier" value={String(s.spend_tier)} />
              <MetaItem
                label="Annual spend"
                value={`€${Number(s.annual_spend_eur).toLocaleString()}`}
              />
              <MetaItem label="Qualification" value={String(s.qualification_status)} />
              <MetaItem label="Strategic importance" value={String(s.strategic_importance)} />
              <MetaItem label="Single source" value={String(s.single_source) === "True" ? "Yes" : "No"} />
              <MetaItem label="Years active" value={String(s.years_active)} />
              <MetaItem label="Archetype" value={String(s.archetype ?? "—")} />
              <MetaItem label="Recommended action" value={String(s.recommended_action ?? "—")} />
            </div>
          </CardContent>
        </Card>

        {/* KPI trends */}
        <section>
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">
            KPI Trends — last 12 months
          </h2>
          <KpiCharts kpis={kpis} />
        </section>

        {/* SHAP */}
        <section>
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">
            ML Explainability — SHAP top features
          </h2>
          <ShapChart features={shapFeatures} />
        </section>

        {/* Claims + Audits */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <section>
            <h2 className="text-sm font-semibold text-zinc-300 mb-3">Recent Claims</h2>
            <ClaimsTable claims={claims} />
          </section>
          <section>
            <h2 className="text-sm font-semibold text-zinc-300 mb-3">Recent Audits</h2>
            <AuditsTable audits={audits} />
          </section>
        </div>

        {/* APQP gate matrix */}
        <section>
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">APQP Gate Matrix</h2>
          <ApqpGates projects={apqpProjects} />
        </section>

        {/* External events */}
        <section>
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">External Events</h2>
          <ExternalEvents events={externalEvents} />
        </section>

      </main>
    </div>
  );
}
