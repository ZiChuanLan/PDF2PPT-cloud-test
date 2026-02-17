export type Provider = "openai" | "claude" | "siliconflow" | "mineru"
export type OcrProvider =
  | "auto"
  | "aiocr"
  | "baidu"
  | "tesseract"
  | "paddle_local"
export type OcrAiProvider = "auto" | "openai" | "siliconflow" | "deepseek" | "ppio" | "novita"
export type OcrAiLinebreakAssistMode = "auto" | "on" | "off"
export type ScannedPageMode = "segmented" | "fullpage"
export type MineruModelVersion = "pipeline" | "vlm" | "MinerU-HTML"
export type TextEraseMode = "smart" | "fill"

export type Settings = {
  provider: Provider
  openaiApiKey: string
  openaiBaseUrl: string
  openaiModel: string
  siliconflowApiKey: string
  siliconflowBaseUrl: string
  siliconflowModel: string
  claudeApiKey: string
  mineruApiToken: string
  mineruBaseUrl: string
  mineruModelVersion: MineruModelVersion
  mineruEnableFormula: boolean
  mineruEnableTable: boolean
  mineruLanguage: string
  mineruIsOcr: boolean
  mineruHybridOcr: boolean
  enableLayoutAssist: boolean
  layoutAssistApplyImageRegions: boolean
  enableOcr: boolean
  textEraseMode: TextEraseMode
  scannedPageMode: ScannedPageMode
  ocrStrictMode: boolean
  ocrProvider: OcrProvider
  ocrBaiduAppId: string
  ocrBaiduApiKey: string
  ocrBaiduSecretKey: string
  ocrTesseractMinConfidence: string
  ocrTesseractLanguage: string
  ocrAiApiKey: string
  ocrAiProvider: OcrAiProvider
  ocrAiBaseUrl: string
  ocrAiModel: string
  ocrAiLinebreakAssistMode: OcrAiLinebreakAssistMode
}

export const SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

export const SETTINGS_STORAGE_KEY = "pdf-to-ppt.settings.v1"

export const defaultSettings: Settings = {
  provider: "openai",
  openaiApiKey: "",
  openaiBaseUrl: "",
  openaiModel: "",
  siliconflowApiKey: "",
  siliconflowBaseUrl: SILICONFLOW_BASE_URL,
  siliconflowModel: "",
  claudeApiKey: "",
  mineruApiToken: "",
  mineruBaseUrl: "",
  mineruModelVersion: "vlm",
  mineruEnableFormula: true,
  mineruEnableTable: true,
  mineruLanguage: "",
  mineruIsOcr: false,
  mineruHybridOcr: false,
  // Experimental and can timeout on some endpoints; keep opt-in by default.
  enableLayoutAssist: false,
  // Keep off by default: some pages may over-match and hide decorative images.
  layoutAssistApplyImageRegions: false,
  // Most real-world PDFs are scans. Default to OCR-on so output is editable.
  enableOcr: true,
  // smart: current adaptive erase; fill: fast rectangle background fill.
  textEraseMode: "fill",
  // segmented: keep some images as editable blocks; fullpage: keep a single page background.
  scannedPageMode: "segmented",
  // Non-strict is more open-source friendly: it enables fallbacks/downgrades
  // and keeps conversion running even if OCR fails on some pages.
  ocrStrictMode: false,
  // Local mode now defaults to pure local OCR (no auto fallback).
  ocrProvider: "tesseract",
  ocrBaiduAppId: "",
  ocrBaiduApiKey: "",
  ocrBaiduSecretKey: "",
  // Lower confidence improves recall on scan-heavy slide decks; text refinement (AI) can clean up later.
  ocrTesseractMinConfidence: "35",
  ocrTesseractLanguage: "chi_sim+eng",
  ocrAiApiKey: "",
  ocrAiProvider: "auto",
  ocrAiBaseUrl: "",
  ocrAiModel: "",
  // Optional OCR visual line-break split for coarse block boxes.
  // auto: backend decides based on OCR provider capabilities.
  ocrAiLinebreakAssistMode: "auto",
}

export function safeParseSettings(value: string | null): Partial<Settings> | null {
  if (!value) return null

  try {
    const parsed = JSON.parse(value) as unknown
    if (!parsed || typeof parsed !== "object") return null
    return parsed as Partial<Settings>
  } catch {
    return null
  }
}

export function loadStoredSettings(): Settings {
  if (typeof window === "undefined") return defaultSettings

  const parsed = safeParseSettings(localStorage.getItem(SETTINGS_STORAGE_KEY))
  const merged = { ...defaultSettings, ...(parsed ?? {}) } as Settings
  const parsedProvider = (parsed as { provider?: string } | null)?.provider
  const parsedParseProvider = (parsed as { parseProvider?: string } | null)?.parseProvider
  if (parsedProvider === "domestic" || parsedParseProvider === "mineru") {
    merged.provider = "mineru"
  }
  if (parsedProvider === "v2" || parsedParseProvider === "v2") {
    // Backward compatibility: legacy "v2 full-page OCR" maps to the normal
    // pipeline with `scannedPageMode=fullpage` + AI OCR settings.
    merged.provider = "siliconflow"
    merged.enableOcr = true
    merged.scannedPageMode = "fullpage"
    merged.ocrProvider = "aiocr"
    if (merged.ocrAiProvider === "auto") {
      merged.ocrAiProvider = "siliconflow"
    }
  }
  if (
    parsedProvider === "openai" &&
    typeof (parsed as { openaiBaseUrl?: unknown } | null)?.openaiBaseUrl === "string"
  ) {
    const legacyOpenaiBase = String(
      (parsed as { openaiBaseUrl?: string } | null)?.openaiBaseUrl || ""
    )
      .trim()
      .toLowerCase()
    if (legacyOpenaiBase.includes("api.siliconflow.cn")) {
      merged.provider = "siliconflow"
      merged.siliconflowApiKey = String(
        (parsed as { openaiApiKey?: string } | null)?.openaiApiKey || merged.siliconflowApiKey
      )
      merged.siliconflowBaseUrl = String(
        (parsed as { openaiBaseUrl?: string } | null)?.openaiBaseUrl || merged.siliconflowBaseUrl
      )
      merged.siliconflowModel = String(
        (parsed as { openaiModel?: string } | null)?.openaiModel || merged.siliconflowModel
      )
    }
  }

  const validProviders: Provider[] = ["openai", "claude", "siliconflow", "mineru"]
  if (!validProviders.includes(merged.provider)) {
    merged.provider = "openai"
  }
  if (!merged.siliconflowBaseUrl.trim()) {
    merged.siliconflowBaseUrl = SILICONFLOW_BASE_URL
  }

  const mineruModels: MineruModelVersion[] = ["pipeline", "vlm", "MinerU-HTML"]
  if (!mineruModels.includes(merged.mineruModelVersion)) {
    merged.mineruModelVersion = "vlm"
  }

  // Backward compatibility: migrate legacy values to the new canonical id.
  const legacyProvider = (parsed as { ocrProvider?: string } | null)?.ocrProvider
  if (legacyProvider === "ai" || legacyProvider === "remote" || legacyProvider === "paddle") {
    merged.ocrProvider = "aiocr"
  }
  if (legacyProvider === "paddle-local" || legacyProvider === "local_paddle") {
    merged.ocrProvider = "paddle_local"
  }
  const validOcrProviders: OcrProvider[] = [
    "auto",
    "aiocr",
    "baidu",
    "tesseract",
    "paddle_local",
  ]
  if (!validOcrProviders.includes(merged.ocrProvider)) {
    merged.ocrProvider = merged.provider === "mineru" ? "auto" : "tesseract"
  }
  if (merged.provider !== "mineru" && merged.ocrProvider === "auto") {
    merged.ocrProvider = "tesseract"
  }
  const validOcrAiProviders: OcrAiProvider[] = [
    "auto",
    "openai",
    "siliconflow",
    "deepseek",
    "ppio",
    "novita",
  ]
  if (!validOcrAiProviders.includes(merged.ocrAiProvider)) {
    merged.ocrAiProvider = "auto"
  }
  const validTextEraseModes: TextEraseMode[] = ["smart", "fill"]
  if (!validTextEraseModes.includes(merged.textEraseMode)) {
    merged.textEraseMode = "fill"
  }
  const validScannedPageModes: ScannedPageMode[] = ["segmented", "fullpage"]
  if (!validScannedPageModes.includes(merged.scannedPageMode)) {
    merged.scannedPageMode = "segmented"
  }
  if (typeof merged.layoutAssistApplyImageRegions !== "boolean") {
    merged.layoutAssistApplyImageRegions = false
  }
  if (typeof merged.ocrStrictMode !== "boolean") {
    merged.ocrStrictMode = false
  }
  const validLinebreakAssistModes: OcrAiLinebreakAssistMode[] = ["auto", "on", "off"]
  const legacyLinebreakAssist = (parsed as { ocrAiLinebreakAssist?: unknown } | null)
    ?.ocrAiLinebreakAssist
  const legacyLinebreakMode = (parsed as { ocrAiLinebreakAssistMode?: unknown } | null)
    ?.ocrAiLinebreakAssistMode
  if (
    typeof legacyLinebreakMode === "string" &&
    validLinebreakAssistModes.includes(legacyLinebreakMode as OcrAiLinebreakAssistMode)
  ) {
    merged.ocrAiLinebreakAssistMode = legacyLinebreakMode as OcrAiLinebreakAssistMode
  } else if (typeof legacyLinebreakAssist === "boolean") {
    merged.ocrAiLinebreakAssistMode = legacyLinebreakAssist ? "on" : "off"
  }
  if (!validLinebreakAssistModes.includes(merged.ocrAiLinebreakAssistMode)) {
    merged.ocrAiLinebreakAssistMode = "auto"
  }
  return merged
}
