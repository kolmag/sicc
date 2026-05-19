const SEVERITY_COLORS: Record<string, string> = {
  High:   "text-red-400",
  Medium: "text-amber-400",
  Low:    "text-emerald-400",
};

const STATUS_COLORS: Record<string, string> = {
  Open:      "text-red-400",
  Mitigated: "text-amber-400",
  Closed:    "text-emerald-400",
};

type Event = Record<string, unknown>;

export function ExternalEvents({ events }: { events: Event[] }) {
  if (!events.length) return <p className="text-zinc-500 text-sm">No external events.</p>;

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800">
      <table className="w-full text-xs text-zinc-300 min-w-[700px]">
        <thead>
          <tr className="bg-zinc-900 text-zinc-400">
            {["Date", "Type", "Severity", "Description", "Status", "CAPA"].map((h) => (
              <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {events.map((e, i) => (
            <tr key={i} className="border-t border-zinc-800 hover:bg-zinc-800/40">
              <td className="px-3 py-2 whitespace-nowrap">{String(e.event_date ?? "").slice(0, 10)}</td>
              <td className="px-3 py-2">{String(e.event_type ?? "—")}</td>
              <td className={`px-3 py-2 font-medium ${SEVERITY_COLORS[String(e.severity)] ?? "text-zinc-400"}`}>
                {String(e.severity ?? "—")}
              </td>
              <td className="px-3 py-2 max-w-xs text-zinc-400">{String(e.description ?? "—")}</td>
              <td className={`px-3 py-2 font-medium ${STATUS_COLORS[String(e.status)] ?? "text-zinc-400"}`}>
                {String(e.status ?? "—")}
              </td>
              <td className="px-3 py-2">
                {String(e.requires_capa) === "True"
                  ? <span className={String(e.capa_linked) === "True" ? "text-emerald-400" : "text-amber-400"}>
                      {String(e.capa_linked) === "True" ? "Linked" : "Required"}
                    </span>
                  : <span className="text-zinc-600">—</span>
                }
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
