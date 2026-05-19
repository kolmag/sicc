import { Badge } from "@/components/ui/badge";

const COLORS: Record<string, string> = {
  red: "bg-red-900/60 text-red-300 border-red-700",
  amber: "bg-amber-900/60 text-amber-300 border-amber-700",
  green: "bg-emerald-900/60 text-emerald-300 border-emerald-700",
};

export function RiskBadge({ label }: { label: string | null }) {
  if (!label) return <span className="text-zinc-500 text-xs">—</span>;
  const key = label.toLowerCase();
  return (
    <Badge
      variant="outline"
      className={`uppercase text-[10px] font-semibold tracking-wide ${COLORS[key] ?? "bg-zinc-800 text-zinc-400 border-zinc-600"}`}
    >
      {label}
    </Badge>
  );
}
