import { NextRequest, NextResponse } from "next/server";
import {
  SESSION_COOKIE,
  SESSION_TTL_SECONDS,
  createSessionToken,
  getSessionSecret,
  safeEqual,
} from "@/lib/session";

export async function POST(request: NextRequest) {
  const password = process.env.SICC_PASSWORD;
  const secret = getSessionSecret();

  // Fail closed: refuse to authenticate unless the server is configured with
  // both a password and a session-signing secret. No insecure defaults.
  if (!password || !secret) {
    return NextResponse.json(
      { error: "Server auth is not configured (set SICC_PASSWORD and SICC_SESSION_SECRET)." },
      { status: 500 },
    );
  }

  const body = await request.json().catch(() => ({}));

  if (typeof body.password !== "string" || !(await safeEqual(body.password, password))) {
    return NextResponse.json({ error: "Invalid password" }, { status: 401 });
  }

  const token = await createSessionToken();
  if (!token) {
    return NextResponse.json({ error: "Server auth is not configured." }, { status: 500 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(SESSION_COOKIE, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_TTL_SECONDS,
  });
  return response;
}

export async function DELETE() {
  const response = NextResponse.json({ ok: true });
  response.cookies.delete(SESSION_COOKIE);
  return response;
}
