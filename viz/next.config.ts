import type { NextConfig } from "next";

const basePath = "/game-app";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
  devIndicators: false,
};

export default nextConfig;
