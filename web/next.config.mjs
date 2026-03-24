function resolveInternalApiOrigin() {
  const rawValue =
    process.env.INTERNAL_API_ORIGIN?.trim() ||
    process.env.INTERNAL_API_HOSTPORT?.trim() ||
    "http://api:8000"

  const normalized = rawValue.replace(/\/+$/, "")
  if (/^https?:\/\//i.test(normalized)) {
    return normalized
  }

  return `http://${normalized}`
}

const internalApiOrigin = resolveInternalApiOrigin()

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
