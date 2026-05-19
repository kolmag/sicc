"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Sparkline } from "@/components/Sparkline";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RiskBadge } from "@/components/RiskBadge";

export type Supplier = {
  supplier_id: string;
  name: string;
  country: string;
  product_family: string;
  spend_tier: string;
  annual_spend_eur: number;
  composite_risk_score: number;
  risk_label: string;
  avg_ppm_3m: number;
  avg_otd_3m: number;
  avg_audit_score_3m: number;
  recommended_action: string;
  ml_prediction: string | null;
  ml_confidence: number | null;
  single_source: string;
  strategic_importance: string;
};

export function SupplierTable({
  suppliers,
  sparklines = {},
}: {
  suppliers: Supplier[];
  sparklines?: Record<string, number[]>;
}) {
  const router = useRouter();
  const [riskFilter, setRiskFilter] = useState("all");
  const [familyFilter, setFamilyFilter] = useState("all");
  const [mlMismatchOnly, setMlMismatchOnly] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  function toggleSelect(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); }
      else if (next.size < 3) { next.add(id); }
      return next;
    });
  }

  const families = useMemo(
    () => Array.from(new Set(suppliers.map((s) => s.product_family))).sort(),
    [suppliers]
  );

  const filtered = useMemo(() => {
    return suppliers.filter((s) => {
      if (riskFilter !== "all" && s.risk_label?.toLowerCase() !== riskFilter)
        return false;
      if (familyFilter !== "all" && s.product_family !== familyFilter)
        return false;
      if (
        mlMismatchOnly &&
        s.ml_prediction !== null &&
        s.ml_prediction?.toLowerCase() === s.risk_label?.toLowerCase()
      )
        return false;
      return true;
    });
  }, [suppliers, riskFilter, familyFilter, mlMismatchOnly]);

  const fmt = (n: number | null, decimals = 1) =>
    n == null ? "—" : n.toFixed(decimals);
  const fmtSpend = (n: number) =>
    new Intl.NumberFormat("en-US", {
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(n);

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="flex flex-wrap gap-3 items-center">
        <Select value={riskFilter} onValueChange={(v) => setRiskFilter(v ?? "all")}>
          <SelectTrigger className="w-36 bg-zinc-900 border-zinc-700 text-zinc-200">
            <SelectValue placeholder="Risk tier" />
          </SelectTrigger>
          <SelectContent className="bg-zinc-900 border-zinc-700 text-zinc-200">
            <SelectItem value="all">All tiers</SelectItem>
            <SelectItem value="red">RED</SelectItem>
            <SelectItem value="amber">AMBER</SelectItem>
            <SelectItem value="green">GREEN</SelectItem>
          </SelectContent>
        </Select>

        <Select value={familyFilter} onValueChange={(v) => setFamilyFilter(v ?? "all")}>
          <SelectTrigger className="w-44 bg-zinc-900 border-zinc-700 text-zinc-200">
            <SelectValue placeholder="Product family" />
          </SelectTrigger>
          <SelectContent className="bg-zinc-900 border-zinc-700 text-zinc-200">
            <SelectItem value="all">All families</SelectItem>
            {families.map((f) => (
              <SelectItem key={f} value={f}>
                {f}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <label className="flex items-center gap-2 text-sm text-zinc-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={mlMismatchOnly}
            onChange={(e) => setMlMismatchOnly(e.target.checked)}
            className="accent-amber-500 w-4 h-4"
          />
          ML/rule mismatch only
        </label>

        <span className="ml-auto text-xs text-zinc-500">
          {filtered.length} of {suppliers.length} suppliers
        </span>
      </div>

      {/* Floating compare bar */}
      {selected.size >= 2 && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-zinc-800 border border-zinc-600 rounded-2xl px-5 py-3 shadow-xl">
          <span className="text-sm text-zinc-300">{selected.size} suppliers selected</span>
          <button
            onClick={() => router.push(`/dashboard/compare?ids=${Array.from(selected).join(",")}`)}
            className="text-sm bg-zinc-100 text-zinc-900 font-medium rounded-xl px-4 py-1.5 hover:bg-white transition-colors"
          >
            Compare →
          </button>
          <button
            onClick={() => setSelected(new Set())}
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            Clear
          </button>
        </div>
      )}

      {/* Table */}
      <div className="rounded-xl border border-zinc-800 overflow-x-auto">
        <Table className="min-w-[1100px]">
          <TableHeader>
            <TableRow className="bg-zinc-900 border-zinc-800 hover:bg-zinc-900">
              <TableHead className="w-8" />
              <TableHead className="text-zinc-400 font-medium">Supplier</TableHead>
              <TableHead className="text-zinc-400 font-medium">Country</TableHead>
              <TableHead className="text-zinc-400 font-medium">Family</TableHead>
              <TableHead className="text-zinc-400 font-medium text-right">Score</TableHead>
              <TableHead className="text-zinc-400 font-medium">Rule</TableHead>
              <TableHead className="text-zinc-400 font-medium">ML</TableHead>
              <TableHead className="text-zinc-400 font-medium">PPM trend</TableHead>
              <TableHead className="text-zinc-400 font-medium text-right">PPM</TableHead>
              <TableHead className="text-zinc-400 font-medium text-right">OTD%</TableHead>
              <TableHead className="text-zinc-400 font-medium text-right">Audit</TableHead>
              <TableHead className="text-zinc-400 font-medium text-right">Spend</TableHead>
              <TableHead className="text-zinc-400 font-medium">Action</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.length === 0 && (
              <TableRow>
                <TableCell colSpan={11} className="text-center text-zinc-500 py-10">
                  No suppliers match the current filters.
                </TableCell>
              </TableRow>
            )}
            {filtered.map((s) => {
              const mismatch =
                s.ml_prediction !== null &&
                s.ml_prediction?.toLowerCase() !== s.risk_label?.toLowerCase();
              return (
                <TableRow
                  key={s.supplier_id}
                  className={`border-zinc-800 hover:bg-zinc-800/50 cursor-pointer ${mismatch ? "bg-amber-950/10" : ""} ${selected.has(s.supplier_id) ? "bg-blue-950/20" : ""}`}
                  onClick={() => router.push(`/dashboard/${s.supplier_id}`)}
                >
                  <TableCell className="w-8 pr-0" onClick={(e) => toggleSelect(s.supplier_id, e)}>
                    <input
                      type="checkbox"
                      checked={selected.has(s.supplier_id)}
                      onChange={() => {}}
                      disabled={!selected.has(s.supplier_id) && selected.size >= 3}
                      className="accent-blue-500 w-3.5 h-3.5 cursor-pointer disabled:opacity-30"
                    />
                  </TableCell>
                  <TableCell className="font-medium text-zinc-100 max-w-48 truncate" title={s.name}>
                    {s.name}
                  </TableCell>
                  <TableCell className="text-zinc-400 text-sm">{s.country}</TableCell>
                  <TableCell className="text-zinc-400 text-sm">{s.product_family}</TableCell>
                  <TableCell className="text-right tabular-nums text-zinc-200">
                    {fmt(s.composite_risk_score)}
                  </TableCell>
                  <TableCell>
                    <RiskBadge label={s.risk_label} />
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <RiskBadge label={s.ml_prediction} />
                      {s.ml_confidence !== null && (
                        <span className="text-[10px] text-zinc-500 tabular-nums">
                          {(s.ml_confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="py-1">
                    <Sparkline values={sparklines[s.supplier_id] ?? []} />
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-zinc-300">
                    {fmt(s.avg_ppm_3m, 0)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-zinc-300">
                    {fmt(s.avg_otd_3m)}%
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-zinc-300">
                    {fmt(s.avg_audit_score_3m)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-zinc-300">
                    €{fmtSpend(s.annual_spend_eur)}
                  </TableCell>
                  <TableCell className="text-xs text-zinc-400 min-w-44">
                    {s.recommended_action}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
