import { NextRequest, NextResponse } from "next/server";

const SESSION_COOKIE = "sicc_session";
const LOGIN_PATH = "/login";

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (pathname === LOGIN_PATH) {
    return NextResponse.next();
  }

  const session = request.cookies.get(SESSION_COOKIE);
  if (!session?.value) {
    const loginUrl = new URL(LOGIN_PATH, request.url);
    loginUrl.searchParams.set("from", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  // Exclude /api/* (Route Handlers + FastAPI rewrite), static assets, and login page
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
