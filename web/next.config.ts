import type { NextConfig } from "next";

const API_URL = process.env.SICC_API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return {
      // afterFiles: checked AFTER Route Handlers, so /api/auth route.ts takes priority
      afterFiles: [
        {
          source: "/api/:path*",
          destination: `${API_URL}/:path*`,
        },
      ],
    };
  },
};

export default nextConfig;
