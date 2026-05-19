"use client";

import { useRouter } from "next/navigation";

export function LogoutButton() {
  const router = useRouter();

  async function handleLogout() {
    await fetch("/api/auth", { method: "DELETE" });
    router.push("/login");
    router.refresh();
  }

  return (
    <button
      onClick={handleLogout}
      className="text-xs border border-zinc-700 rounded-md px-3 py-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
    >
      Sign out
    </button>
  );
}
