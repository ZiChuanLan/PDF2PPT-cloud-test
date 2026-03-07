const internalApiOrigin =
  process.env.INTERNAL_API_ORIGIN?.trim().replace(/\/+$/, "") || "http://api:8000"

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/health",
        destination: `${internalApiOrigin}/health`,
      },
      {
        source: "/api/:path*",
        destination: `${internalApiOrigin}/api/:path*`,
      },
    ]
  },
}

export default nextConfig
