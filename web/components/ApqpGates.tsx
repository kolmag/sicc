const GATES = [
  { key: "supplier_selection",           label: "Supplier Selection" },
  { key: "supplier_nomination",          label: "Supplier Nomination" },
  { key: "design_validation_of_process", label: "Design Validation" },
  { key: "process_validation",           label: "Process Validation" },
  { key: "initial_sample_validation",    label: "Initial Sample" },
  { key: "start_of_production",          label: "SOP" },
  { key: "pqa_management",               label: "PQA Management" },
  { key: "yearly_is_submission",         label: "Yearly IS" },
  { key: "ppap_update",                  label: "PPAP Update" },
];

const STATUS_COLORS: Record<string, string> = {
  Validated:    "bg-emerald-900/60 text-emerald-300 border-emerald-700",
  Submitted:    "bg-blue-900/60 text-blue-300 border-blue-700",
  "In Progress":"bg-amber-900/60 text-amber-300 border-amber-700",
  "Not Started":"bg-zinc-800 text-zinc-500 border-zinc-700",
  "On Hold":    "bg-red-900/60 text-red-300 border-red-700",
};

type ApqpProject = Record<string, unknown>;

function GateCell({ status }: { status: string }) {
  const cls = STATUS_COLORS[status] ?? "bg-zinc-800 text-zinc-500 border-zinc-700";
  return (
    <span className={`inline-block text-[10px] font-medium px-1.5 py-0.5 rounded border ${cls}`}>
      {status || "—"}
    </span>
  );
}

export function ApqpGates({ projects }: { projects: ApqpProject[] }) {
  if (!projects.length) return <p className="text-zinc-500 text-sm">No APQP projects.</p>;

  return (
    <div className="space-y-4">
      {projects.map((p, i) => (
        <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
            <div>
              <span className="text-sm font-medium text-zinc-100">{String(p.project_type)}</span>
              <span className="ml-2 text-xs text-zinc-500">{String(p.project_id)}</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-zinc-400">
              <span>Status: <span className="text-zinc-200">{String(p.status)}</span></span>
              <span>SOP: <span className="text-zinc-200">{String(p.customer_sop_date ?? "—").slice(0, 10)}</span></span>
              <span>Complete: <span className="text-zinc-200">{String(p.completion_pct)}%</span></span>
              {String(p.is_delayed) === "True" && (
                <span className="text-red-400 font-medium">⚠ Delayed</span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-3 sm:grid-cols-5 xl:grid-cols-9 gap-2">
            {GATES.map((g) => {
              const status = String(p[`${g.key}_status`] ?? "—");
              return (
                <div key={g.key} className="text-center">
                  <p className="text-[9px] text-zinc-600 mb-1 leading-tight">{g.label}</p>
                  <GateCell status={status} />
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
