const API_ORIGIN_STORAGE_KEY = "ppt_opencode_api_origin"
const DEFAULT_FALLBACK_ORIGIN = "http://localhost:8000"

let resolvedApiOriginCache: string | null = null
let resolveInFlight: Promise<string> | null = null

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "")
}

function uniq(values: string[]): string[] {
  const out: string[] = []
  for (const value of values) {
    const v = value.trim()
    if (!v || out.includes(v)) continue
    out.push(v)
  }
  return out
}

function normalizeApiOrigin(raw: string): string | null {
  const input = String(raw || "").trim()
  if (!input) return null

  let candidate = input
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(candidate)) {
    candidate = `http://${candidate}`
  }

  try {
    const parsed = new URL(candidate)
    if (!parsed.hostname) return null
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null
    return trimTrailingSlash(`${parsed.protocol}//${parsed.host}`)
  } catch {
    return null
  }
}

function getConfiguredApiOrigin(): string | null {
  return normalizeApiOrigin(String(process.env.NEXT_PUBLIC_API_URL || ""))
}

export function getStoredApiOrigin(): string | null {
  if (typeof window === "undefined") return null
  try {
    const raw = localStorage.getItem(API_ORIGIN_STORAGE_KEY) || ""
    return normalizeApiOrigin(raw)
  } catch {
    return null
  }
}

export function setStoredApiOrigin(raw: string): string {
  const normalized = normalizeApiOrigin(raw)
  if (!normalized) {
    throw new Error("API 地址格式错误，请输入类似 http://127.0.0.1:8000")
  }
  if (typeof window !== "undefined") {
    localStorage.setItem(API_ORIGIN_STORAGE_KEY, normalized)
  }
  resolvedApiOriginCache = normalized
  return normalized
}

export function clearStoredApiOrigin(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem(API_ORIGIN_STORAGE_KEY)
  }
  resolvedApiOriginCache = null
}

function inferAutoCandidates(): string[] {
  const candidates: string[] = []
  const envPort = String(process.env.NEXT_PUBLIC_API_PORT || "").trim()
  const ports = uniq([envPort, "8000", "8001"]).filter(Boolean)

  if (typeof window !== "undefined") {
    const protocol = window.location.protocol || "http:"
    const host = window.location.hostname || "localhost"
    const hosts = uniq([
      host,
      host === "localhost" ? "127.0.0.1" : "",
      host === "127.0.0.1" ? "localhost" : "",
    ]).filter(Boolean)

    for (const h of hosts) {
      for (const p of ports) {
        candidates.push(`${protocol}//${h}:${p}`)
      }
    }
  }

  for (const h of ["localhost", "127.0.0.1"]) {
    for (const p of ports) {
      candidates.push(`http://${h}:${p}`)
    }
  }

  return uniq(candidates)
}

export function listApiOriginCandidates(): string[] {
  return uniq(
    [
      getStoredApiOrigin() || "",
      getConfiguredApiOrigin() || "",
      resolvedApiOriginCache || "",
      ...inferAutoCandidates(),
      DEFAULT_FALLBACK_ORIGIN,
    ].filter(Boolean)
  )
}

async function probeApiOrigin(origin: string, timeoutMs = 1200): Promise<boolean> {
  if (typeof window === "undefined") return false

  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${origin}/health`, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    })
    if (!response.ok) return false
    const body = await response.json().catch(() => null)
    if (body && typeof body === "object" && "status" in body) {
      return String((body as { status?: unknown }).status || "").toLowerCase() === "ok"
    }
    return true
  } catch {
    return false
  } finally {
    window.clearTimeout(timer)
  }
}

export function getApiOrigin(): string {
  return listApiOriginCandidates()[0] || DEFAULT_FALLBACK_ORIGIN
}

export async function resolveApiOrigin(options?: { force?: boolean }): Promise<string> {
  const force = Boolean(options?.force)

  if (!force && resolvedApiOriginCache) return resolvedApiOriginCache
  if (!force && resolveInFlight) return resolveInFlight

  const configured = getConfiguredApiOrigin()
  const manual = getStoredApiOrigin()
  const candidates = listApiOriginCandidates()

  // Server-side render fallback: skip probing.
  if (typeof window === "undefined") {
    resolvedApiOriginCache = manual || configured || candidates[0] || DEFAULT_FALLBACK_ORIGIN
    return resolvedApiOriginCache
  }

  resolveInFlight = (async () => {
    for (const origin of candidates) {
      // If user manually configured address, prefer quickly but still verify.
      if (await probeApiOrigin(origin)) {
        resolvedApiOriginCache = origin
        return origin
      }
    }
    resolvedApiOriginCache = manual || configured || candidates[0] || DEFAULT_FALLBACK_ORIGIN
    return resolvedApiOriginCache
  })()

  try {
    return await resolveInFlight
  } finally {
    resolveInFlight = null
  }
}

export function getApiBaseUrl(origin = getApiOrigin()): string {
  return `${trimTrailingSlash(origin)}/api/v1`
}

export async function getResolvedApiBaseUrl(): Promise<string> {
  return getApiBaseUrl(await resolveApiOrigin())
}

function normalizeApiPath(path: string): string {
  const raw = String(path || "").trim()
  if (!raw) return "/"
  return raw.startsWith("/") ? raw : `/${raw}`
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const base = await getResolvedApiBaseUrl()
  return fetch(`${base}${normalizeApiPath(path)}`, init)
}

export function normalizeFetchError(error: unknown, fallback: string): string {
  if (error instanceof DOMException && error.name === "AbortError") {
    return "请求已取消"
  }

  if (error instanceof TypeError) {
    const raw = String(error.message || "").toLowerCase()
    if (raw.includes("network") || raw.includes("fetch") || raw.includes("failed")) {
      return `${fallback}（网络连接失败，请检查 API 地址与后端 CORS 设置）`
    }
  }

  if (error instanceof Error) {
    const message = String(error.message || "").trim()
    if (message) return message
  }
  return fallback
}
