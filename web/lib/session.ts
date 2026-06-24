// Signed session tokens for the SICC web gate.
//
// The session cookie holds an HMAC-SHA256–signed, expiring token rather than a
// constant value, so it cannot be forged by simply setting `sicc_session=1`.
// Uses the Web Crypto API so it runs in both the Edge (proxy) and Node (route
// handler) runtimes without any extra dependency.

export const SESSION_COOKIE = "sicc_session";
export const SESSION_TTL_SECONDS = 60 * 60 * 8; // 8 hours

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function bytesToBase64Url(bytes: Uint8Array): string {
  let str = "";
  for (const b of bytes) str += String.fromCharCode(b);
  return btoa(str).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlToBytes(value: string): Uint8Array {
  let s = value.replace(/-/g, "+").replace(/_/g, "/");
  const pad = s.length % 4;
  if (pad) s += "=".repeat(4 - pad);
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/** The signing secret, or null if the server is not configured. */
export function getSessionSecret(): string | null {
  return process.env.SICC_SESSION_SECRET || null;
}

async function importKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

/** Create a signed, expiring session token. Returns null if unconfigured. */
export async function createSessionToken(): Promise<string | null> {
  const secret = getSessionSecret();
  if (!secret) return null;

  const exp = Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS;
  const payload = bytesToBase64Url(encoder.encode(JSON.stringify({ exp })));
  const key = await importKey(secret);
  const sig = await crypto.subtle.sign("HMAC", key, encoder.encode(payload));
  return `${payload}.${bytesToBase64Url(new Uint8Array(sig))}`;
}

/** Verify a session token's signature and expiry. */
export async function verifySessionToken(token: string | undefined): Promise<boolean> {
  if (!token) return false;
  const secret = getSessionSecret();
  if (!secret) return false;

  const parts = token.split(".");
  if (parts.length !== 2) return false;
  const [payload, sig] = parts;

  try {
    const key = await importKey(secret);
    const valid = await crypto.subtle.verify(
      "HMAC",
      key,
      base64UrlToBytes(sig),
      encoder.encode(payload),
    );
    if (!valid) return false;

    const data = JSON.parse(decoder.decode(base64UrlToBytes(payload)));
    return typeof data.exp === "number" && data.exp >= Math.floor(Date.now() / 1000);
  } catch {
    return false;
  }
}

/** Constant-time string comparison via fixed-length digests. */
export async function safeEqual(a: string, b: string): Promise<boolean> {
  const [ha, hb] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(a)),
    crypto.subtle.digest("SHA-256", encoder.encode(b)),
  ]);
  const va = new Uint8Array(ha);
  const vb = new Uint8Array(hb);
  let diff = 0;
  for (let i = 0; i < va.length; i++) diff |= va[i] ^ vb[i];
  return diff === 0;
}
