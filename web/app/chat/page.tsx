"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { RiskBadge } from "@/components/RiskBadge";

type Mode = "kb" | "portfolio";
type MessageRole = "user" | "assistant" | "status";

type PortfolioRow = Record<string, unknown>;

type Message = {
  id: string;
  role: MessageRole;
  text: string;
  mode: Mode;
  // KB fields
  sources?: string[];
  confidence?: string;
  risk_level?: string | null;
  action_required?: boolean;
  // Portfolio fields
  rows?: PortfolioRow[];
  columns?: string[];
  scope_count?: number;
  intent?: string;
  elapsed_ms?: number;
};

const RISK_DOMAINS = ["ppap", "apqp", "scar", "capa", "8d", "audit", "ncr", "ppm", "otd"];
const FAMILIES = ["Raw Materials", "Electronics", "Plastics", "Machined Parts", "Assemblies", "Packaging"];

const KB_PROMPTS = [
  "What does PPAP Level 3 require?",
  "When is a for-cause audit mandatory?",
  "What are the RED tier KPI thresholds?",
  "What is the SCAR escalation process?",
  "What are the APQP Phase 4 pass criteria?",
  "What buffer stock is required for single-source suppliers?",
];

const PORTFOLIO_PROMPTS = [
  "Which RED-risk suppliers have open major audit findings?",
  "Show sole-source suppliers with PPM > 300",
  "Which APQP programmes are delayed and linked to RED suppliers?",
  "Top 10 worst suppliers by OTD",
  "Which suppliers have critical external events and no linked CAPA?",
  "Highest spend suppliers in Electronics",
];

function ConfidencePip({ level }: { level: string }) {
  const colors: Record<string, string> = { high: "bg-emerald-400", medium: "bg-amber-400", low: "bg-red-400" };
  return <span className={`inline-block w-2 h-2 rounded-full mr-1 ${colors[level] ?? "bg-zinc-500"}`} />;
}

function fmtCell(val: unknown): string {
  if (val === null || val === undefined || val === "") return "—";
  if (typeof val === "number") return val.toFixed(1);
  return String(val);
}

function PortfolioTable({ columns, rows }: { columns: string[]; rows: PortfolioRow[] }) {
  const router = useRouter();
  const RISK_COL = "risk_label";

  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-700 mt-3">
      <table className="w-full text-xs text-zinc-300 min-w-[500px]">
        <thead>
          <tr className="bg-zinc-800 text-zinc-400">
            {columns.map((c) => (
              <th key={c} className="px-2 py-1.5 text-left font-medium whitespace-nowrap">
                {c.replace(/_/g, " ").replace("avg ", "").replace(" 3m", "")}
              </th>
            ))}
            <th className="px-2 py-1.5 w-6" />
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const supplierId = row["supplier_id"] as string | undefined;
            const clickable = !!supplierId;
            return (
              <tr
                key={i}
                onClick={() => supplierId && router.push(`/dashboard/${supplierId}`)}
                className={`border-t border-zinc-800 transition-colors ${clickable ? "cursor-pointer hover:bg-zinc-700/50" : "hover:bg-zinc-800/40"}`}
              >
                {columns.map((c) => (
                  <td key={c} className="px-2 py-1.5 whitespace-nowrap">
                    {c === RISK_COL
                      ? <RiskBadge label={String(row[c] ?? "")} />
                      : fmtCell(row[c])}
                  </td>
                ))}
                <td className="px-2 py-1.5 text-zinc-600">
                  {clickable && <span className="text-[10px]">→</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.some((r) => r["supplier_id"]) && (
        <p className="text-[10px] text-zinc-600 px-2 py-1.5 border-t border-zinc-800">
          Click a row to open supplier detail
        </p>
      )}
    </div>
  );
}

function KbMessage({ msg }: { msg: Message }) {
  return (
    <div className="space-y-2">
      <p className="text-sm text-zinc-100 leading-relaxed whitespace-pre-wrap">{msg.text}</p>
      {msg.sources && msg.sources.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {msg.sources.map((s, i) => (
            <span key={i} className="text-[10px] bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-zinc-400">{s}</span>
          ))}
        </div>
      )}
      <div className="flex items-center gap-3 pt-0.5 flex-wrap">
        {msg.confidence && (
          <span className="text-[10px] text-zinc-500">
            <ConfidencePip level={msg.confidence} />{msg.confidence} confidence
          </span>
        )}
        {msg.risk_level && msg.risk_level !== "not_applicable" && (
          <span className="text-[10px] text-zinc-500">risk: <RiskBadge label={msg.risk_level} /></span>
        )}
        {msg.action_required && <span className="text-[10px] text-amber-400 font-medium">⚠ action required</span>}
        {msg.elapsed_ms != null && <span className="text-[10px] text-zinc-600 ml-auto">{msg.elapsed_ms}ms</span>}
      </div>
    </div>
  );
}

function downloadCsv(columns: string[], rows: PortfolioRow[], filename: string) {
  const escape = (v: unknown) => {
    const s = String(v ?? "");
    return s.includes(",") || s.includes('"') || s.includes("\n") ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [
    columns.map(escape).join(","),
    ...rows.map((r) => columns.map((c) => escape(r[c])).join(",")),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function PortfolioMessage({ msg }: { msg: Message }) {
  return (
    <div className="space-y-1">
      <p className="text-sm text-zinc-100 font-medium">{msg.text}</p>
      {msg.rows && msg.rows.length > 0 && msg.columns && (
        <PortfolioTable columns={msg.columns} rows={msg.rows} />
      )}
      {msg.rows && msg.rows.length === 0 && (
        <p className="text-xs text-zinc-500 italic">No suppliers matched.</p>
      )}
      <div className="flex items-center gap-3 pt-1 flex-wrap">
        {msg.intent && <span className="text-[10px] text-zinc-600">intent: {msg.intent}</span>}
        {msg.scope_count != null && <span className="text-[10px] text-zinc-600">{msg.scope_count} suppliers in scope</span>}
        {msg.rows && msg.rows.length > 0 && msg.columns && (
          <button
            onClick={() => downloadCsv(msg.columns!, msg.rows!, `sicc-${msg.intent ?? "query"}.csv`)}
            className="text-[10px] text-zinc-400 hover:text-zinc-200 border border-zinc-700 rounded px-2 py-0.5 transition-colors"
          >
            ↓ Export CSV
          </button>
        )}
        {msg.elapsed_ms != null && <span className="text-[10px] text-zinc-600 ml-auto">{msg.elapsed_ms}ms</span>}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const [mode, setMode] = useState<Mode>("kb");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [risk, setRisk] = useState("");
  const [family, setFamily] = useState("");
  const [region, setRegion] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function applyPrompt(p: string) {
    setInput(p);
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", text: q, mode };
    const statusMsg: Message = { id: crypto.randomUUID(), role: "status", text: "Thinking…", mode };
    setMessages((prev) => [...prev, userMsg, statusMsg]);
    setInput("");
    setLoading(true);

    try {
      if (mode === "kb") {
        await streamKb(q);
      } else {
        await fetchPortfolio(q);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  async function streamKb(q: string) {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, risk: risk || null, family: family || null, session_id: "chat-ui" }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(`API error ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const lines = part.split("\n");
          const eventLine = lines.find((l) => l.startsWith("event:"));
          const dataLine = lines.find((l) => l.startsWith("data:"));
          if (!eventLine || !dataLine) continue;
          const event = eventLine.replace("event:", "").trim();
          const payload = JSON.parse(dataLine.replace("data:", "").trim());

          if (event === "status") {
            setMessages((prev) => prev.map((m) =>
              m.role === "status"
                ? { ...m, text: payload.message === "running_sicc_brain" ? "Running RAG pipeline…" : "Thinking…" }
                : m
            ));
          }
          if (event === "result") {
            const r = payload.result;
            setMessages((prev) => [
              ...prev.filter((m) => m.role !== "status"),
              {
                id: crypto.randomUUID(), role: "assistant", mode: "kb",
                text: r.answer, sources: r.sources, confidence: r.confidence,
                risk_level: r.risk_level, action_required: r.action_required,
                elapsed_ms: payload.elapsed_ms,
              },
            ]);
          }
          if (event === "error") {
            setMessages((prev) => [
              ...prev.filter((m) => m.role !== "status"),
              { id: crypto.randomUUID(), role: "assistant", mode: "kb", text: `Error: ${payload.message}` },
            ]);
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        setMessages((prev) => [
          ...prev.filter((m) => m.role !== "status"),
          { id: crypto.randomUUID(), role: "assistant", mode: "kb", text: "Cannot reach the SICC API. Make sure `uv run sicc-api` is running." },
        ]);
      }
    }
  }

  async function fetchPortfolio(q: string) {
    try {
      const res = await fetch("/api/chat/portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, family: family || null, region: region || null, risk: risk || null }),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      setMessages((prev) => [
        ...prev.filter((m) => m.role !== "status"),
        {
          id: crypto.randomUUID(), role: "assistant", mode: "portfolio",
          text: data.answer, rows: data.rows, columns: data.columns,
          scope_count: data.scope_count, intent: data.intent, elapsed_ms: data.elapsed_ms,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev.filter((m) => m.role !== "status"),
        { id: crypto.randomUUID(), role: "assistant", mode: "portfolio", text: "Cannot reach the SICC API. Make sure `uv run sicc-api` is running." },
      ]);
    }
  }

  const prompts = mode === "kb" ? KB_PROMPTS : PORTFOLIO_PROMPTS;

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-3 flex items-center gap-4 shrink-0 flex-wrap">
        <Link href="/dashboard" className="text-zinc-500 hover:text-zinc-200 text-sm transition-colors">← Dashboard</Link>
        <div className="h-4 w-px bg-zinc-700" />

        {/* Mode toggle */}
        <div className="flex rounded-lg border border-zinc-700 overflow-hidden text-xs">
          <button
            onClick={() => setMode("kb")}
            className={`px-3 py-1.5 transition-colors ${mode === "kb" ? "bg-zinc-700 text-zinc-100" : "text-zinc-400 hover:text-zinc-200"}`}
          >
            📚 Knowledge Base
          </button>
          <button
            onClick={() => setMode("portfolio")}
            className={`px-3 py-1.5 transition-colors border-l border-zinc-700 ${mode === "portfolio" ? "bg-zinc-700 text-zinc-100" : "text-zinc-400 hover:text-zinc-200"}`}
          >
            📊 Portfolio Data
          </button>
        </div>

        {/* Filters */}
        <div className="ml-auto flex gap-2 items-center flex-wrap">
          {mode === "kb" && (
            <select value={risk} onChange={(e) => setRisk(e.target.value)}
              className="text-xs bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-zinc-300 focus:outline-none">
              <option value="">All risk domains</option>
              {RISK_DOMAINS.map((d) => <option key={d} value={d}>{d.toUpperCase()}</option>)}
            </select>
          )}
          <select value={family} onChange={(e) => setFamily(e.target.value)}
            className="text-xs bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-zinc-300 focus:outline-none">
            <option value="">All families</option>
            {FAMILIES.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          {mode === "portfolio" && (
            <input value={region} onChange={(e) => setRegion(e.target.value)}
              placeholder="Region (e.g. Europe)"
              className="text-xs bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 text-zinc-300 focus:outline-none w-36" />
          )}
        </div>
      </header>

      {/* Suggested prompts */}
      <div className="border-b border-zinc-800/60 px-6 py-2 flex gap-2 overflow-x-auto shrink-0">
        {prompts.map((p) => (
          <button key={p} onClick={() => applyPrompt(p)}
            className="text-[10px] whitespace-nowrap border border-zinc-700 rounded-full px-2.5 py-1 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 transition-colors shrink-0">
            {p}
          </button>
        ))}
      </div>

      {/* Message thread */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-2 text-zinc-600">
            {mode === "kb"
              ? <><p className="text-sm">Ask anything about supplier quality procedures.</p>
                  <p className="text-xs">PPAP · APQP · SCAR · audit standards · risk thresholds</p></>
              : <><p className="text-sm">Query your live supplier portfolio data.</p>
                  <p className="text-xs">Risk tiers · OTD · PPM · SCARs · APQP delays · external events</p></>
            }
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "status" && (
              <div className="flex items-center gap-2 text-xs text-zinc-500 italic">
                <span className="inline-flex gap-0.5">
                  {[0, 1, 2].map((i) => (
                    <span key={i} className="w-1 h-1 bg-zinc-500 rounded-full animate-bounce"
                      style={{ animationDelay: `${i * 0.15}s` }} />
                  ))}
                </span>
                {msg.text}
              </div>
            )}
            {msg.role === "user" && (
              <div className="max-w-xl bg-zinc-800 rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm text-zinc-100">
                {msg.text}
              </div>
            )}
            {msg.role === "assistant" && (
              <div className="w-full max-w-4xl bg-zinc-900 border border-zinc-800 rounded-2xl rounded-tl-sm px-4 py-3">
                <div className="flex items-center gap-1.5 mb-2">
                  <span className="text-[9px] uppercase tracking-wider font-medium text-zinc-600">
                    {msg.mode === "kb" ? "Knowledge Base" : "Portfolio Data"}
                  </span>
                </div>
                {msg.mode === "kb"
                  ? <KbMessage msg={msg} />
                  : <PortfolioMessage msg={msg} />
                }
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <form onSubmit={submit} className="shrink-0 border-t border-zinc-800 px-6 py-4 flex gap-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={mode === "kb"
            ? "Ask about PPAP, audits, SCARs, risk thresholds…"
            : "Which suppliers have PPM > 500? Show delayed APQP programmes…"}
          disabled={loading}
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 disabled:opacity-50"
        />
        {loading && mode === "kb" ? (
          <button type="button" onClick={() => abortRef.current?.abort()}
            className="px-4 py-2.5 text-sm rounded-xl border border-zinc-700 text-zinc-400 hover:bg-zinc-800 transition-colors">
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!input.trim() || loading}
            className="px-4 py-2.5 text-sm rounded-xl bg-zinc-100 text-zinc-900 font-medium hover:bg-white transition-colors disabled:opacity-30">
            Send
          </button>
        )}
      </form>
    </div>
  );
}
