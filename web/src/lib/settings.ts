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
export type OcrAiChainMode = "direct" | "doc_parser" | "layout_block"
export type OcrAiLayoutModel = "pp_doclayout_v3"
export type OcrAiPromptPreset =
  | "auto"
  | "generic_vision"
  | "openai_vision"
  | "qwen_vl"
  | "glm_v"
  | "deepseek_ocr"
export type LayoutAssistMode = "off" | "on" | "auto"
export type VisionAssistMode = LayoutAssistMode
export type ScannedPageMode = "segmented" | "fullpage"
export type PptGenerationMode = "standard" | "fast" | "turbo"
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
  removeFooterNotebooklm: boolean
  textEraseMode: TextEraseMode
  scannedPageMode: ScannedPageMode
  pptGenerationMode: PptGenerationMode
  imageBgClearExpandMinPt: string
  imageBgClearExpandMaxPt: string
  imageBgClearExpandRatio: string
  scannedImageRegionMinAreaRatio: string
  scannedImageRegionMaxAreaRatio: string
  scannedImageRegionMaxAspectRatio: string
  ocrRenderDpi: string
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
  ocrAiChainMode: OcrAiChainMode
  ocrAiLayoutModel: OcrAiLayoutModel
  ocrAiPromptPreset: OcrAiPromptPreset
  ocrAiDirectPromptOverride: string
  ocrAiLayoutBlockPromptOverride: string
  ocrAiImageRegionPromptOverride: string
  ocrPaddleVlDocparserMaxSidePx: string
  ocrAiPageConcurrencyAuto: boolean
  ocrAiPageConcurrency: string
  ocrAiBlockConcurrency: string
  ocrAiRequestsPerMinute: string
  ocrAiTokensPerMinute: string
  ocrAiMaxRetries: string
}

export const SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
export const DEFAULT_AIOCR_PROVIDER: OcrAiProvider = "siliconflow"
export const DEFAULT_AIOCR_MODEL = ""
export const DEFAULT_AIOCR_CHAIN_MODE: OcrAiChainMode = "layout_block"

export const SETTINGS_STORAGE_KEY = "pdf-to-ppt.settings.v1"
export const BAIDU_DOC_PARSE_TYPE_LABELS: Record<BaiduDocParseType, string> = {
  general: "普通文档解析",
  paddle_vl: "PaddleOCR-VL",
}

export const PPT_GENERATION_MODE_LABELS: Record<PptGenerationMode, string> = {
  standard: "精准",
  fast: "快速",
  turbo: "极速",
}

export function isPaddleOcrVlModelName(value: string | null | undefined): boolean {
  return String(value || "").trim().toLowerCase().includes("paddleocr-vl")
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
  // Keep off by default: only enable when exported decks contain NotebookLM footer branding.
  removeFooterNotebooklm: false,
  // smart: current adaptive erase; fill: fast rectangle background fill.
  textEraseMode: "fill",
  // segmented: keep images as editable blocks; fullpage: leave them in the page background.
  scannedPageMode: "segmented",
  // standard: current fidelity-first generator. fast: experiment for speed-first runs.
  pptGenerationMode: "fast",
  // Tunables for image-underlay cleanup and scanned image-region filtering.
  imageBgClearExpandMinPt: "0.35",
  imageBgClearExpandMaxPt: "1.5",
  imageBgClearExpandRatio: "0.012",
  scannedImageRegionMinAreaRatio: "0.0025",
  scannedImageRegionMaxAreaRatio: "0.72",
  scannedImageRegionMaxAspectRatio: "4.8",
  ocrRenderDpi: "200",
  // Default on. Strict mode keeps OCR benchmarking and production runs honest
  // by surfacing provider/setup failures instead of silently downgrading.
  ocrStrictMode: true,
  // Local mode now defaults to pure local OCR (no auto fallback).
  ocrProvider: "tesseract",
  baiduDocParseType: "paddle_vl",
  ocrBaiduAppId: "",
  ocrBaiduApiKey: "",
  ocrBaiduSecretKey: "",
  // Lower confidence improves recall on scan-heavy slide decks.
  ocrTesseractMinConfidence: "35",
  ocrTesseractLanguage: "chi_sim+eng",
  ocrAiApiKey: "",
  ocrAiProvider: DEFAULT_AIOCR_PROVIDER,
  ocrAiBaseUrl: SILICONFLOW_BASE_URL,
  ocrAiModel: DEFAULT_AIOCR_MODEL,
  ocrAiChainMode: DEFAULT_AIOCR_CHAIN_MODE,
  ocrAiLayoutModel: "pp_doclayout_v3",
  ocrAiPromptPreset: "auto",
  ocrAiDirectPromptOverride: "",
  ocrAiLayoutBlockPromptOverride: "",
  ocrAiImageRegionPromptOverride: "",
  ocrPaddleVlDocparserMaxSidePx: "2200",
  ocrAiPageConcurrencyAuto: true,
  ocrAiPageConcurrency: "1",
  ocrAiBlockConcurrency: "",
  ocrAiRequestsPerMinute: "",
  ocrAiTokensPerMinute: "",
  ocrAiMaxRetries: "0",
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
  const merged = {
    ...defaultSettings,
    ...(parsed ?? {}),
  } as Settings & {
    ocrGeometryMode?: unknown
    ocrAiLinebreakAssist?: unknown
    ocrAiLinebreakAssistMode?: unknown
  }
  const parsedOcrAiPageConcurrencyAuto = (
    parsed as { ocrAiPageConcurrencyAuto?: unknown } | null
  )?.ocrAiPageConcurrencyAuto
  const parsedOcrAiPageConcurrency = (
    parsed as { ocrAiPageConcurrency?: unknown } | null
  )?.ocrAiPageConcurrency
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
    // pipeline with `scannedPageMode=fullpage` + AIOCR settings.
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
    merged.ocrAiProvider = DEFAULT_AIOCR_PROVIDER
  }
  const validOcrAiChainModes: OcrAiChainMode[] = ["direct", "doc_parser", "layout_block"]
  if (!validOcrAiChainModes.includes(merged.ocrAiChainMode)) {
    merged.ocrAiChainMode = DEFAULT_AIOCR_CHAIN_MODE
  }
  const validOcrAiLayoutModels: OcrAiLayoutModel[] = ["pp_doclayout_v3"]
  if (!validOcrAiLayoutModels.includes(merged.ocrAiLayoutModel)) {
    merged.ocrAiLayoutModel = "pp_doclayout_v3"
  }
  const validOcrAiPromptPresets: OcrAiPromptPreset[] = [
    "auto",
    "generic_vision",
    "openai_vision",
    "qwen_vl",
    "glm_v",
    "deepseek_ocr",
  ]
  if (!validOcrAiPromptPresets.includes(merged.ocrAiPromptPreset)) {
    merged.ocrAiPromptPreset = "auto"
  }
  const validTextEraseModes: TextEraseMode[] = ["smart", "fill"]
  if (!validTextEraseModes.includes(merged.textEraseMode)) {
    merged.textEraseMode = "fill"
  }
  const validScannedPageModes: ScannedPageMode[] = ["segmented", "fullpage"]
  if (!validScannedPageModes.includes(merged.scannedPageMode)) {
    merged.scannedPageMode = "fullpage"
  }
  const validPptGenerationModes: PptGenerationMode[] = ["standard", "fast", "turbo"]
  if (!validPptGenerationModes.includes(merged.pptGenerationMode)) {
    merged.pptGenerationMode = "fast"
  }
  if (merged.parseEngineMode === "remote_ocr") {
    if (merged.ocrAiProvider === "auto") {
      merged.ocrAiProvider = DEFAULT_AIOCR_PROVIDER
    }
    if (!merged.ocrAiBaseUrl.trim() && merged.ocrAiProvider === "siliconflow") {
      merged.ocrAiBaseUrl = SILICONFLOW_BASE_URL
    }
    if (!merged.ocrAiModel.trim()) {
      merged.ocrAiModel = DEFAULT_AIOCR_MODEL
    }
    if (
      merged.ocrAiChainMode === "direct" &&
      isPaddleOcrVlModelName(merged.ocrAiModel)
    ) {
      merged.ocrAiModel = DEFAULT_AIOCR_MODEL
    }
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
  merged.ocrRenderDpi = toNumberLikeString(
    merged.ocrRenderDpi,
    defaultSettings.ocrRenderDpi
  )
  merged.ocrPaddleVlDocparserMaxSidePx = toNumberLikeString(
    merged.ocrPaddleVlDocparserMaxSidePx,
    defaultSettings.ocrPaddleVlDocparserMaxSidePx
  )
  merged.ocrAiPageConcurrency = toNumberLikeString(
    merged.ocrAiPageConcurrency,
    defaultSettings.ocrAiPageConcurrency
  )
  merged.ocrAiBlockConcurrency = toNumberLikeString(
    merged.ocrAiBlockConcurrency,
    defaultSettings.ocrAiBlockConcurrency
  )
  merged.ocrAiRequestsPerMinute = toNumberLikeString(
    merged.ocrAiRequestsPerMinute,
    defaultSettings.ocrAiRequestsPerMinute
  )
  merged.ocrAiTokensPerMinute = toNumberLikeString(
    merged.ocrAiTokensPerMinute,
    defaultSettings.ocrAiTokensPerMinute
  )
  merged.ocrAiMaxRetries = toNumberLikeString(
    merged.ocrAiMaxRetries,
    defaultSettings.ocrAiMaxRetries
  )
  const normalizePromptOverride = (value: unknown): string => {
    if (typeof value !== "string") return ""
    return value.replace(/\r\n?/g, "\n").slice(0, 6000).trim()
  }
  merged.ocrAiDirectPromptOverride = normalizePromptOverride(
    merged.ocrAiDirectPromptOverride
  )
  merged.ocrAiLayoutBlockPromptOverride = normalizePromptOverride(
    merged.ocrAiLayoutBlockPromptOverride
  )
  merged.ocrAiImageRegionPromptOverride = normalizePromptOverride(
    merged.ocrAiImageRegionPromptOverride
  )
  const ocrRenderDpi = Number(merged.ocrRenderDpi)
  if (!Number.isFinite(ocrRenderDpi) || ocrRenderDpi < 72) {
    merged.ocrRenderDpi = defaultSettings.ocrRenderDpi
  } else {
    merged.ocrRenderDpi = String(Math.min(400, Math.round(ocrRenderDpi)))
  }
  const paddleDocMaxSidePx = Number(merged.ocrPaddleVlDocparserMaxSidePx)
  if (!Number.isFinite(paddleDocMaxSidePx) || paddleDocMaxSidePx < 0) {
    merged.ocrPaddleVlDocparserMaxSidePx = defaultSettings.ocrPaddleVlDocparserMaxSidePx
  } else {
    merged.ocrPaddleVlDocparserMaxSidePx = String(Math.round(paddleDocMaxSidePx))
  }
  const pageConcurrency = Number(merged.ocrAiPageConcurrency)
  if (!Number.isFinite(pageConcurrency) || pageConcurrency < 1) {
    merged.ocrAiPageConcurrency = defaultSettings.ocrAiPageConcurrency
  } else {
    merged.ocrAiPageConcurrency = String(Math.min(8, Math.round(pageConcurrency)))
  }
  if (typeof parsedOcrAiPageConcurrencyAuto === "boolean") {
    merged.ocrAiPageConcurrencyAuto = parsedOcrAiPageConcurrencyAuto
  } else {
    const normalizedParsedPageConcurrency =
      typeof parsedOcrAiPageConcurrency === "number" &&
      Number.isFinite(parsedOcrAiPageConcurrency)
        ? String(parsedOcrAiPageConcurrency)
        : typeof parsedOcrAiPageConcurrency === "string"
          ? parsedOcrAiPageConcurrency.trim()
          : ""
    merged.ocrAiPageConcurrencyAuto =
      !normalizedParsedPageConcurrency ||
      normalizedParsedPageConcurrency === defaultSettings.ocrAiPageConcurrency
  }
  const normalizeOptionalPositiveIntString = (value: string): string => {
    const trimmed = value.trim()
    if (!trimmed) return ""
    const parsed = Number(trimmed)
    if (!Number.isFinite(parsed) || parsed <= 0) return ""
    return String(Math.round(parsed))
  }
  merged.ocrAiBlockConcurrency = normalizeOptionalPositiveIntString(
    merged.ocrAiBlockConcurrency
  )
  merged.ocrAiRequestsPerMinute = normalizeOptionalPositiveIntString(
    merged.ocrAiRequestsPerMinute
  )
  merged.ocrAiTokensPerMinute = normalizeOptionalPositiveIntString(
    merged.ocrAiTokensPerMinute
  )
  const maxRetries = Number(merged.ocrAiMaxRetries)
  if (!Number.isFinite(maxRetries) || maxRetries < 0) {
    merged.ocrAiMaxRetries = defaultSettings.ocrAiMaxRetries
  } else {
    merged.ocrAiMaxRetries = String(Math.min(8, Math.round(maxRetries)))
  }
  merged.enableLayoutAssist = false
  merged.layoutAssistApplyImageRegions = false
  if (typeof merged.ocrAiPageConcurrencyAuto !== "boolean") {
    merged.ocrAiPageConcurrencyAuto = true
  }
  if (typeof merged.ocrStrictMode !== "boolean") {
    merged.ocrStrictMode = true
  }
  merged.visualAssistModeLocal = "off"
  merged.visualAssistModeRemote = "off"
  merged.visualAssistModeBaiduDoc = "off"
  merged.visualAssistModeMineru = "off"
  delete merged.ocrGeometryMode
  delete merged.ocrAiLinebreakAssist
  delete merged.ocrAiLinebreakAssistMode
  return merged
}
