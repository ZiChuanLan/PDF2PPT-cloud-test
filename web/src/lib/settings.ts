export type MainProvider = "openai" | "claude" | "siliconflow"
export type Provider = MainProvider | "mineru"
export type ParseEngineMode = "local_ocr" | "remote_ocr" | "baidu_doc" | "mineru_cloud"
export type BaiduDocParseType = "general" | "paddle_vl"
export type OcrProvider =
  | "auto"
  | "aiocr"
  | "baidu"
  | "tesseract"
  | "paddle_local"
export type OcrAiProvider = "auto" | "openai" | "siliconflow" | "deepseek" | "ppio" | "novita"
export type OcrAiLinebreakAssistMode = "auto" | "on" | "off"
export type LayoutAssistMode = "off" | "on" | "auto"
export type VisionAssistMode = LayoutAssistMode
export type OcrGeometryMode = "auto" | "local_tesseract" | "direct_ai"
export type ScannedPageMode = "segmented" | "fullpage"
export type MineruModelVersion = "pipeline" | "vlm" | "MinerU-HTML"
export type TextEraseMode = "smart" | "fill"

export type Settings = {
  provider: Provider
  preferredMainProvider: MainProvider
  parseEngineMode: ParseEngineMode
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
  visualAssistModeLocal: LayoutAssistMode
  visualAssistModeRemote: LayoutAssistMode
  visualAssistModeBaiduDoc: LayoutAssistMode
  visualAssistModeMineru: LayoutAssistMode
  enableOcr: boolean
  textEraseMode: TextEraseMode
  scannedPageMode: ScannedPageMode
  imageBgClearExpandMinPt: string
  imageBgClearExpandMaxPt: string
  imageBgClearExpandRatio: string
  scannedImageRegionMinAreaRatio: string
  scannedImageRegionMaxAreaRatio: string
  scannedImageRegionMaxAspectRatio: string
  ocrStrictMode: boolean
  ocrProvider: OcrProvider
  baiduDocParseType: BaiduDocParseType
  ocrBaiduAppId: string
  ocrBaiduApiKey: string
  ocrBaiduSecretKey: string
  ocrTesseractMinConfidence: string
  ocrTesseractLanguage: string
  ocrAiApiKey: string
  ocrAiProvider: OcrAiProvider
  ocrAiBaseUrl: string
  ocrAiModel: string
  ocrPaddleVlDocparserMaxSidePx: string
  ocrAiLinebreakAssistMode: OcrAiLinebreakAssistMode
  ocrGeometryMode: OcrGeometryMode
}

export const SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

export const SETTINGS_STORAGE_KEY = "pdf-to-ppt.settings.v1"
export const BAIDU_DOC_PARSE_TYPE_LABELS: Record<BaiduDocParseType, string> = {
  general: "普通文档解析",
  paddle_vl: "PaddleOCR-VL",
}

export const PARSE_ENGINE_MODE_LABELS: Record<ParseEngineMode, string> = {
  local_ocr: "传统 OCR",
  remote_ocr: "AIOCR",
  baidu_doc: "百度解析",
  mineru_cloud: "云端 MinerU",
}

export const defaultSettings: Settings = {
  provider: "openai",
  preferredMainProvider: "openai",
  parseEngineMode: "local_ocr",
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
  // AI layout-assist policy for the four parse chains. Keep all off by default.
  visualAssistModeLocal: "off",
  visualAssistModeRemote: "off",
  visualAssistModeBaiduDoc: "off",
  visualAssistModeMineru: "off",
  // Most real-world PDFs are scans. Default to OCR-on so output is editable.
  enableOcr: true,
  // smart: current adaptive erase; fill: fast rectangle background fill.
  textEraseMode: "fill",
  // segmented: keep some images as editable blocks; fullpage: keep a single page background.
  scannedPageMode: "segmented",
  // Tunables for image-underlay cleanup and scanned image-region filtering.
  imageBgClearExpandMinPt: "0.35",
  imageBgClearExpandMaxPt: "1.5",
  imageBgClearExpandRatio: "0.012",
  scannedImageRegionMinAreaRatio: "0.0025",
  scannedImageRegionMaxAreaRatio: "0.72",
  scannedImageRegionMaxAspectRatio: "4.8",
  // Default on. Strict mode keeps OCR benchmarking and production runs honest
  // by surfacing provider/setup failures instead of silently downgrading.
  ocrStrictMode: true,
  // Local mode now defaults to pure local OCR (no auto fallback).
  ocrProvider: "tesseract",
  baiduDocParseType: "paddle_vl",
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
  ocrPaddleVlDocparserMaxSidePx: "2200",
  // AI OCR line-break post-process policy. `auto` lets the backend decide
  // based on the selected OCR model instead of piggybacking on layout assist.
  ocrAiLinebreakAssistMode: "auto",
  // OCR geometry strategy for AI OCR models.
  // auto: use stable local geometry for generic VL, keep direct AI geometry for OCR-specialized models.
  ocrGeometryMode: "auto",
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
  const parsedPreferredMainProvider = (
    parsed as { preferredMainProvider?: string } | null
  )?.preferredMainProvider
  const parsedParseProvider = (parsed as { parseProvider?: string } | null)?.parseProvider
  const parsedParseEngineMode = (
    parsed as { parseEngineMode?: string } | null
  )?.parseEngineMode
  const parsedBaiduDocParseType = (
    parsed as { baiduDocParseType?: string } | null
  )?.baiduDocParseType
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
  const validMainProviders: MainProvider[] = ["openai", "claude", "siliconflow"]
  if (!validMainProviders.includes(merged.preferredMainProvider)) {
    if (parsedPreferredMainProvider && validMainProviders.includes(parsedPreferredMainProvider as MainProvider)) {
      merged.preferredMainProvider = parsedPreferredMainProvider as MainProvider
    } else if (merged.provider !== "mineru") {
      merged.preferredMainProvider = merged.provider
    } else if (parsedProvider === "claude") {
      merged.preferredMainProvider = "claude"
    } else if (parsedProvider === "siliconflow") {
      merged.preferredMainProvider = "siliconflow"
    } else {
      merged.preferredMainProvider = "openai"
    }
  }
  if (merged.provider !== "mineru") {
    merged.preferredMainProvider = merged.provider
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
  const validParseEngineModes: ParseEngineMode[] = [
    "local_ocr",
    "remote_ocr",
    "baidu_doc",
    "mineru_cloud",
  ]
  if (
    typeof parsedParseEngineMode === "string" &&
    validParseEngineModes.includes(parsedParseEngineMode as ParseEngineMode)
  ) {
    merged.parseEngineMode = parsedParseEngineMode as ParseEngineMode
  } else if (parsedParseProvider === "baidu_doc") {
    merged.parseEngineMode = "baidu_doc"
  } else if (merged.ocrProvider === "baidu") {
    merged.parseEngineMode = "baidu_doc"
  } else if (merged.provider === "mineru") {
    merged.parseEngineMode = "mineru_cloud"
  } else if (parsedProvider === "v2" || parsedParseProvider === "v2" || merged.ocrProvider === "aiocr") {
    merged.parseEngineMode = "remote_ocr"
  } else {
    merged.parseEngineMode = "local_ocr"
  }
  if (merged.parseEngineMode === "mineru_cloud") {
    merged.provider = "mineru"
  } else if (merged.provider === "mineru") {
    merged.provider = merged.preferredMainProvider
  }
  if (merged.parseEngineMode === "local_ocr" && merged.ocrProvider === "baidu") {
    merged.ocrProvider = "tesseract"
  }
  const validBaiduDocParseTypes: BaiduDocParseType[] = ["general", "paddle_vl"]
  if (typeof parsedBaiduDocParseType === "string") {
    const normalizedBaiduDocParseType = parsedBaiduDocParseType.trim().toLowerCase()
    if (
      normalizedBaiduDocParseType === "paddle_vl" ||
      normalizedBaiduDocParseType === "paddle-vl" ||
      normalizedBaiduDocParseType === "paddleocr-vl" ||
      normalizedBaiduDocParseType === "paddleocr_vl" ||
      normalizedBaiduDocParseType === "vl"
    ) {
      merged.baiduDocParseType = "paddle_vl"
    } else if (
      normalizedBaiduDocParseType === "general" ||
      normalizedBaiduDocParseType === "normal" ||
      normalizedBaiduDocParseType === "default" ||
      normalizedBaiduDocParseType === "ordinary"
    ) {
      merged.baiduDocParseType = "general"
    }
  }
  if (!validBaiduDocParseTypes.includes(merged.baiduDocParseType)) {
    merged.baiduDocParseType = "paddle_vl"
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
  const toNumberLikeString = (value: unknown, fallback: string): string => {
    if (typeof value === "string") {
      return value
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      return String(value)
    }
    return fallback
  }
  merged.imageBgClearExpandMinPt = toNumberLikeString(
    merged.imageBgClearExpandMinPt,
    defaultSettings.imageBgClearExpandMinPt
  )
  merged.imageBgClearExpandMaxPt = toNumberLikeString(
    merged.imageBgClearExpandMaxPt,
    defaultSettings.imageBgClearExpandMaxPt
  )
  merged.imageBgClearExpandRatio = toNumberLikeString(
    merged.imageBgClearExpandRatio,
    defaultSettings.imageBgClearExpandRatio
  )
  merged.scannedImageRegionMinAreaRatio = toNumberLikeString(
    merged.scannedImageRegionMinAreaRatio,
    defaultSettings.scannedImageRegionMinAreaRatio
  )
  merged.scannedImageRegionMaxAreaRatio = toNumberLikeString(
    merged.scannedImageRegionMaxAreaRatio,
    defaultSettings.scannedImageRegionMaxAreaRatio
  )
  merged.scannedImageRegionMaxAspectRatio = toNumberLikeString(
    merged.scannedImageRegionMaxAspectRatio,
    defaultSettings.scannedImageRegionMaxAspectRatio
  )
  merged.ocrPaddleVlDocparserMaxSidePx = toNumberLikeString(
    merged.ocrPaddleVlDocparserMaxSidePx,
    defaultSettings.ocrPaddleVlDocparserMaxSidePx
  )
  const paddleDocMaxSidePx = Number(merged.ocrPaddleVlDocparserMaxSidePx)
  if (!Number.isFinite(paddleDocMaxSidePx) || paddleDocMaxSidePx < 0) {
    merged.ocrPaddleVlDocparserMaxSidePx = defaultSettings.ocrPaddleVlDocparserMaxSidePx
  } else {
    merged.ocrPaddleVlDocparserMaxSidePx = String(Math.round(paddleDocMaxSidePx))
  }
  if (typeof merged.layoutAssistApplyImageRegions !== "boolean") {
    merged.layoutAssistApplyImageRegions = false
  }
  if (typeof merged.ocrStrictMode !== "boolean") {
    merged.ocrStrictMode = true
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
    merged.ocrAiLinebreakAssistMode = "off"
  }

  const validLayoutAssistModes: LayoutAssistMode[] = ["off", "on", "auto"]
  const legacyLayoutAssistMode: LayoutAssistMode = merged.enableLayoutAssist ? "auto" : "off"
  if (!validLayoutAssistModes.includes(merged.visualAssistModeLocal)) {
    merged.visualAssistModeLocal = legacyLayoutAssistMode
  }
  if (!validLayoutAssistModes.includes(merged.visualAssistModeRemote)) {
    merged.visualAssistModeRemote = legacyLayoutAssistMode
  }
  if (!validLayoutAssistModes.includes(merged.visualAssistModeBaiduDoc)) {
    merged.visualAssistModeBaiduDoc = "off"
  }
  if (!validLayoutAssistModes.includes(merged.visualAssistModeMineru)) {
    merged.visualAssistModeMineru = merged.enableLayoutAssist ? "auto" : "off"
  }

  const validOcrGeometryModes: OcrGeometryMode[] = ["auto", "local_tesseract", "direct_ai"]
  if (!validOcrGeometryModes.includes(merged.ocrGeometryMode)) {
    merged.ocrGeometryMode = "auto"
  }
  return merged
}
