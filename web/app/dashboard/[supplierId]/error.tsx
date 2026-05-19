"use client";

import Link from "next/link";

export default function SupplierError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex flex-col min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 px-6 py-4">
        <Link href="/dashboard" className="text-zinc-500 hover:text-zinc-200 text-sm">
          ← Dashboard
        </Link>
      </header>
      <main className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-4">
          <p className="text-zinc-400 text-sm">{error.message}</p>
          <div className="flex gap-3 justify-center">
            <button
              onClick={reset}
              className="text-xs text-zinc-300 border border-zinc-700 rounded px-3 py-1.5 hover:bg-zinc-800 transition-colors"
            >
              Retry
            </button>
            <Link
              href="/dashboard"
              className="text-xs text-zinc-300 border border-zinc-700 rounded px-3 py-1.5 hover:bg-zinc-800 transition-colors"
            >
              Back to dashboard
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
