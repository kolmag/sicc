import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SupplierTable, type Supplier } from "@/components/SupplierTable";
import { FeatureImportanceChart, type FeatureRow } from "@/components/FeatureImportanceChart";
import { LogoutButton } from "@/components/LogoutButton";

type Sparklines = Record<string, number[]>;

const API = process.env.SICC_API_URL ?? "http://localhost:8000";

async function fetchSuppliers(): Promise<Supplier[]> {
  try {
    const res = await fetch(`${API}/suppliers`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return data.suppliers ?? [];
  } catch {
    return [];
  }
}

type ModelMetrics = {
  winner_name: string;
  n_features: number;
  n_suppliers: number;
  accuracy: number;
  f1_macro: number;
  auc_ovr: number;
  f1_red: number;
};

async function fetchMetrics(): Promise<ModelMetrics | null> {
  try {
    const res = await fetch(`${API}/model/metrics`, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function fetchSparklines(): Promise<Sparklines> {
  try {
    const res = await fetch(`${API}/suppliers/sparklines`, { cache: "no-store" });
    if (!res.ok) return {};
    const data = await res.json();
    return data.sparklines ?? {};
  } catch {
    return {};
  }
}

async function fetchFeatureImportance(): Promise<FeatureRow[]> {
  try {
    const res = await fetch(`${API}/model/feature-importance?top_n=20`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return data.features ?? [];
  } catch {
    return [];
  }
}

function kpiCards(suppliers: Supplier[]) {
  const red = suppliers.filter((s) => s.risk_label?.toLowerCase() === "red").length;
  const amber = suppliers.filter((s) => s.risk_label?.toLowerCase() === "amber").length;
  const green = suppliers.filter((s) => s.risk_label?.toLowerCase() === "green").length;
  const totalSpend = suppliers.reduce((acc, s) => acc + (s.annual_spend_eur ?? 0), 0);
  const singleSource = suppliers.filter(
    (s) => String(s.single_source).toLowerCase() === "true"
  ).length;
  const mlMismatch = suppliers.filter(
    (s) =>
      s.ml_prediction !== null &&
      s.ml_prediction?.toLowerCase() !== s.risk_label?.toLowerCase()
  ).length;
  return { red, amber, green, totalSpend, singleSource, mlMismatch };
}

const fmtBn = (n: number) =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);

export default async function DashboardPage() {
  const [suppliers, metrics, featureImportance, sparklines] = await Promise.all([
    fetchSuppliers(),
    fetchMetrics(),
    fetchFeatureImportance(),
    fetchSparklines(),
  ]);
  const kpi = kpiCards(suppliers);

  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">
            SICC — Supplier Intelligence Command Center
          </h1>
          <p className="text-xs text-zinc-500 mt-0.5">Risk Scoring Engine</p>
        </div>
        <div className="flex items-center gap-3">
          {metrics && (
            <div className="text-xs text-emerald-400 border border-emerald-800 bg-emerald-950/40 rounded-md px-3 py-1.5">
              ✓ {metrics.winner_name} · {metrics.n_features} features ·{" "}
              AUC {metrics.auc_ovr.toFixed(3)} · F1-Red {metrics.f1_red.toFixed(3)}
            </div>
          )}
          <Link
            href="/chat"
            className="text-xs border border-zinc-700 rounded-md px-3 py-1.5 text-zinc-300 hover:bg-zinc-800 transition-colors"
          >
            Supplier Q&A →
          </Link>
          <LogoutButton />
        </div>
      </header>

      <main className="flex-1 px-6 py-6 space-y-6">
        {/* KPI cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-1 pt-4 px-4">
              <CardTitle className="text-xs font-medium text-zinc-400 uppercase tracking-wide">
                RED suppliers
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <p className="text-3xl font-bold text-red-400">{kpi.red}</p>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-1 pt-4 px-4">
              <CardTitle className="text-xs font-medium text-zinc-400 uppercase tracking-wide">
                AMBER suppliers
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <p className="text-3xl font-bold text-amber-400">{kpi.amber}</p>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-1 pt-4 px-4">
              <CardTitle className="text-xs font-medium text-zinc-400 uppercase tracking-wide">
                GREEN suppliers
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <p className="text-3xl font-bold text-emerald-400">{kpi.green}</p>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-1 pt-4 px-4">
              <CardTitle className="text-xs font-medium text-zinc-400 uppercase tracking-wide">
                Total portfolio
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <p className="text-3xl font-bold text-zinc-100">{suppliers.length}</p>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-1 pt-4 px-4">
              <CardTitle className="text-xs font-medium text-zinc-400 uppercase tracking-wide">
                Annual spend
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <p className="text-3xl font-bold text-zinc-100">€{fmtBn(kpi.totalSpend)}</p>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-1 pt-4 px-4">
              <CardTitle className="text-xs font-medium text-zinc-400 uppercase tracking-wide">
                ML mismatches
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <p className="text-3xl font-bold text-amber-300">{kpi.mlMismatch}</p>
              <p className="text-[10px] text-zinc-500 mt-0.5">ML ≠ rule-based</p>
            </CardContent>
          </Card>
        </div>

        {/* Feature importance */}
        {featureImportance.length > 0 && (
          <FeatureImportanceChart features={featureImportance} />
        )}

        {/* Supplier table */}
        <div>
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">
            Supplier Portfolio — sorted by risk score ↓
          </h2>
          {suppliers.length === 0 ? (
            <p className="text-zinc-500 text-sm">
              Cannot reach the SICC API. Start it with{" "}
              <code className="text-zinc-300 bg-zinc-800 px-1 rounded">uv run sicc-api</code>{" "}
              in the project root.
            </p>
          ) : (
            <SupplierTable suppliers={suppliers} sparklines={sparklines} />
          )}
        </div>
      </main>
    </div>
  );
}
