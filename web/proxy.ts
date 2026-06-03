import { NextRequest, NextResponse } from "next/server";
import { SESSION_COOKIE, verifySessionToken } from "@/lib/session";

const LOGIN_PATH = "/login";
const AUTH_API_PATH = "/api/auth";

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Login page and the auth endpoint itself must be reachable unauthenticated.
  if (pathname === LOGIN_PATH || pathname === AUTH_API_PATH) {
    return NextResponse.next();
  }

  const token = request.cookies.get(SESSION_COOKIE)?.value;
  if (await verifySessionToken(token)) {
    return NextResponse.next();
  }

  // API requests (including the FastAPI rewrites under /api/*) get a 401 rather
  // than an HTML redirect, so the backend is never reachable without a session.
  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const loginUrl = new URL(LOGIN_PATH, request.url);
  loginUrl.searchParams.set("from", pathname);
  return NextResponse.redirect(loginUrl);
}

export const config = {
  // Run on all routes except static assets and the favicon. /api/* is now
  // included so the FastAPI rewrites are gated behind a valid session.
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
