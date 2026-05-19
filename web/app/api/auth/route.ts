import { NextRequest, NextResponse } from "next/server";

const SESSION_COOKIE = "sicc_session";
const PASSWORD = process.env.SICC_PASSWORD ?? "sicc2025";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}));

  if (body.password !== PASSWORD) {
    return NextResponse.json({ error: "Invalid password" }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(SESSION_COOKIE, "1", {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 8, // 8 hours
  });
  return response;
}

export async function DELETE() {
  const response = NextResponse.json({ ok: true });
  response.cookies.delete(SESSION_COOKIE);
  return response;
}
