import {
  SILICONFLOW_BASE_URL,
  type LayoutAssistMode,
  type MainProvider,
  type OcrAiLinebreakAssistMode,
  type OcrProvider,
  type Settings,
} from "./settings.ts"

export type ParseEngineMode = "local_ocr" | "remote_ocr" | "mineru_cloud"
export type OcrConfigSource = "dedicated" | "main" | "none"

export type RunConfig = {
  parseProvider: "local" | "mineru"
  llmProvider: "openai" | "claude"
  mainApiKey: string
  mainBaseUrl: string
  mainModel: string
  selectedOcrProvider: OcrProvider
  effectiveOcrProvider: OcrProvider
  effectiveOcrAiKey: string
  effectiveOcrAiBaseUrl: string
  effectiveOcrAiModel: string
  effectiveOcrAiProvider: string
  ocrAiConfigSource: OcrConfigSource
  layoutAssistChain: ParseEngineMode
  layoutAssistMode: LayoutAssistMode
  layoutAssistEnabled: boolean
  ocrLinebreakAssistMode: OcrAiLinebreakAssistMode
  visionGeometryMode: Settings["ocrGeometryMode"]
  predictedGeometryMode: Settings["ocrGeometryMode"] | "n/a"
  predictedGeometryReason: string | null
  shouldAttachOcrAiParams: boolean
}

export type ValidationResult = {
  ok: boolean
  message?: string
}

export type OcrSettingsState = {
  isMineruProvider: boolean
  isOcrEnabledForCurrentEngine: boolean
  hasBaiduCredentials: boolean
  canUseAiOcr: boolean
  selectedOcrProvider: OcrProvider
  visibleOcrProvider: OcrProvider
  parseEngineMode: ParseEngineMode
  currentLayoutAssistMode: LayoutAssistMode
  isLayoutAssistEnabledForCurrentEngine: boolean
  currentOcrLinebreakAssistMode: OcrAiLinebreakAssistMode
  canConfigureOcrLinebreakAssist: boolean
  isRemoteOcrMode: boolean
  isOcrProviderAuto: boolean
  isOcrProviderAi: boolean
  isOcrProviderPaddleLocal: boolean
  isOcrProviderBaidu: boolean
  isOcrProviderTesseract: boolean
  isAiOcrProviderSelected: boolean
  needsRequiredOcrAiConfig: boolean
  supportsOptionalOcrAiConfig: boolean
  hasAnyOcrAiConfigValue: boolean
  shouldExpandOptionalOcrAiConfig: boolean
  shouldShowAiVendorAdapter: boolean
  shouldShowBaiduConfig: boolean
  shouldShowTesseractConfig: boolean
  shouldShowLocalOcrCheck: boolean
  availableOcrProviders: OcrProvider[]
  ocrModelsConfigSource: OcrConfigSource
  ocrModelsApiKey: string
  ocrModelsBaseUrl: string
  predictedGeometryMode: Settings["ocrGeometryMode"] | "n/a"
  predictedGeometryReason: string | null
  runConfig: RunConfig
}

export const OCR_PROVIDER_LABELS: Record<OcrProvider, string> = {
  auto: "自动（混合）",
  aiocr: "AI OCR（OpenAI 兼容）",
  paddle_local: "PaddleOCR（本地）",
  baidu: "百度 OCR",
  tesseract: "Tesseract（本地）",
}

export const OCR_GEOMETRY_MODE_LABELS: Record<Settings["ocrGeometryMode"], string> = {
  auto: "自动",
  local_tesseract: "本地定位",
  direct_ai: "AI bbox",
}

const OCR_CONFIG_SOURCE_LABELS: Record<OcrConfigSource, string> = {
  dedicated: "OCR 独立配置",
  main: "复用主 AI 配置",
  none: "未配置",
}

export const LOCAL_PARSE_OCR_PROVIDERS: OcrProvider[] = [
  "tesseract",
  "paddle_local",
]

export const REMOTE_PARSE_OCR_PROVIDERS: OcrProvider[] = ["aiocr", "baidu"]
export const MINERU_OCR_PROVIDERS: OcrProvider[] = []

export function getOcrConfigSourceLabel(source: OcrConfigSource): string {
  return OCR_CONFIG_SOURCE_LABELS[source]
}

function getResolvedMainProvider(settings: Settings): MainProvider {
  return settings.provider === "mineru" ? settings.preferredMainProvider : settings.provider
}

export function getMainProviderConfig(settings: Settings) {
  const provider = getResolvedMainProvider(settings)
  if (provider === "siliconflow") {
    return {
      provider: "openai" as const,
      ocrAdapter: "siliconflow" as const,
      apiKey: settings.siliconflowApiKey.trim(),
      baseUrl: settings.siliconflowBaseUrl.trim() || SILICONFLOW_BASE_URL,
      model: settings.siliconflowModel.trim(),
    }
  }
  if (provider === "claude") {
    return {
      provider: "claude" as const,
      ocrAdapter: null,
      apiKey: settings.claudeApiKey.trim(),
      baseUrl: "",
      model: "",
    }
  }
  return {
    provider: "openai" as const,
    ocrAdapter: "openai" as const,
    apiKey: settings.openaiApiKey.trim(),
    baseUrl: settings.openaiBaseUrl.trim(),
    model: settings.openaiModel.trim(),
  }
}

export function normalizeVisibleOcrProvider(settings: Settings): OcrProvider {
  const rawProvider = (settings.ocrProvider || "").trim().toLowerCase()

  if (settings.provider === "mineru") {
    if (rawProvider === "baidu" || rawProvider === "tesseract" || rawProvider === "auto") {
      return rawProvider
    }
    return "auto"
  }

  if (
    rawProvider === "aiocr" ||
    rawProvider === "baidu" ||
    rawProvider === "tesseract" ||
    rawProvider === "paddle_local"
  ) {
    return rawProvider
  }

  return "tesseract"
}

function predictGeometryMode(
  provider: OcrProvider,
  requestedMode: Settings["ocrGeometryMode"]
): {
  mode: Settings["ocrGeometryMode"] | "n/a"
  reason: string | null
} {
  if (provider !== "aiocr") {
    return { mode: "n/a", reason: null }
  }
  if (requestedMode === "local_tesseract") {
    return {
      mode: "direct_ai",
      reason: "显式 AI OCR 已固定为纯 AI OCR，不再混用本地 Tesseract 定位。",
    }
  }
  return {
    mode: "direct_ai",
    reason: "显式 AI OCR 固定使用模型自身 bbox。",
  }
}

function resolveOcrAiConfigSource({
  explicitAiOcrSelected,
  dedicatedApiKey,
}: {
  explicitAiOcrSelected: boolean
  dedicatedApiKey: string
}): OcrConfigSource {
  if (!explicitAiOcrSelected) return "none"
  return dedicatedApiKey ? "dedicated" : "none"
}

function toFiniteFloatStringOrUndefined(value: string): string | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const n = Number(trimmed)
  if (!Number.isFinite(n)) return undefined
  return String(n)
}

function appendBaiduFields(form: FormData, settings: Settings) {
  form.append("ocr_baidu_app_id", settings.ocrBaiduAppId.trim())
  form.append("ocr_baidu_api_key", settings.ocrBaiduApiKey.trim())
  form.append("ocr_baidu_secret_key", settings.ocrBaiduSecretKey.trim())
}

function appendTesseractFields(form: FormData, settings: Settings) {
  if (settings.ocrTesseractLanguage.trim()) {
    form.append("ocr_tesseract_language", settings.ocrTesseractLanguage.trim())
  }
  const minConf = Number(settings.ocrTesseractMinConfidence)
  if (Number.isFinite(minConf)) {
    form.append("ocr_tesseract_min_confidence", String(minConf))
  }
}

function getDedicatedOcrAiBaseUrl(settings: Settings): string {
  const baseUrl = settings.ocrAiBaseUrl.trim()
  if (baseUrl) return baseUrl
  return settings.ocrAiProvider === "siliconflow" ? SILICONFLOW_BASE_URL : ""
}

export function resolveRunConfig(settings: Settings): RunConfig {
  const parseProvider: RunConfig["parseProvider"] =
    settings.provider === "mineru" ? "mineru" : "local"
  const main = getMainProviderConfig(settings)
  const selectedOcrProvider = normalizeVisibleOcrProvider(settings)
  const effectiveOcrProvider = selectedOcrProvider
  const explicitAiOcrSelected =
    parseProvider === "local" && effectiveOcrProvider === "aiocr"
  const dedicatedOcrAiKey = settings.ocrAiApiKey.trim()
  const ocrAiConfigSource = resolveOcrAiConfigSource({
    explicitAiOcrSelected,
    dedicatedApiKey: dedicatedOcrAiKey,
  })
  const effectiveOcrAiKey =
    ocrAiConfigSource === "dedicated"
      ? dedicatedOcrAiKey
      : ""
  const effectiveOcrAiBaseUrl =
    ocrAiConfigSource === "dedicated"
      ? getDedicatedOcrAiBaseUrl(settings)
      : ""
  const effectiveOcrAiModel = explicitAiOcrSelected ? settings.ocrAiModel.trim() : ""
  const effectiveOcrAiProvider = explicitAiOcrSelected
    ? (settings.ocrAiProvider || "auto").trim() || "auto"
    : "auto"

  const layoutAssistChain: ParseEngineMode =
    parseProvider === "mineru"
      ? "mineru_cloud"
      : effectiveOcrProvider === "aiocr" || effectiveOcrProvider === "baidu"
        ? "remote_ocr"
        : "local_ocr"
  const layoutAssistMode: LayoutAssistMode =
    layoutAssistChain === "local_ocr"
      ? settings.visualAssistModeLocal
      : layoutAssistChain === "remote_ocr"
        ? settings.visualAssistModeRemote
        : settings.visualAssistModeMineru
  const layoutAssistEnabled = layoutAssistMode !== "off"
  const ocrLinebreakAssistMode: OcrAiLinebreakAssistMode =
    parseProvider === "local" ? settings.ocrAiLinebreakAssistMode : "off"
  const visionGeometryMode: Settings["ocrGeometryMode"] =
    effectiveOcrProvider === "aiocr" ? "direct_ai" : "auto"
  const geometryPrediction = predictGeometryMode(
    effectiveOcrProvider,
    visionGeometryMode
  )
  const shouldAttachOcrAiParams =
    parseProvider === "local" &&
    effectiveOcrProvider === "aiocr" &&
    Boolean(effectiveOcrAiKey)

  return {
    parseProvider,
    llmProvider: main.provider,
    mainApiKey: main.apiKey,
    mainBaseUrl: main.baseUrl,
    mainModel: main.model,
    selectedOcrProvider,
    effectiveOcrProvider,
    effectiveOcrAiKey,
    effectiveOcrAiBaseUrl,
    effectiveOcrAiModel,
    effectiveOcrAiProvider,
    ocrAiConfigSource,
    layoutAssistChain,
    layoutAssistMode,
    layoutAssistEnabled,
    ocrLinebreakAssistMode,
    visionGeometryMode,
    predictedGeometryMode: geometryPrediction.mode,
    predictedGeometryReason: geometryPrediction.reason,
    shouldAttachOcrAiParams,
  }
}

export function resolveOcrSettingsState(settings: Settings): OcrSettingsState {
  const runConfig = resolveRunConfig(settings)
  const isMineruProvider = settings.provider === "mineru"
  const isOcrEnabledForCurrentEngine = !isMineruProvider
  const hasBaiduCredentials =
    Boolean(settings.ocrBaiduAppId.trim()) &&
    Boolean(settings.ocrBaiduApiKey.trim()) &&
    Boolean(settings.ocrBaiduSecretKey.trim())
  const canUseAiOcr = !isMineruProvider
  const selectedOcrProvider = runConfig.selectedOcrProvider
  const parseEngineMode = runConfig.layoutAssistChain
  const currentLayoutAssistMode = runConfig.layoutAssistMode
  const isLayoutAssistEnabledForCurrentEngine = currentLayoutAssistMode !== "off"
  const currentOcrLinebreakAssistMode = runConfig.ocrLinebreakAssistMode
  const canConfigureOcrLinebreakAssist = !isMineruProvider && isOcrEnabledForCurrentEngine
  const isRemoteOcrMode = parseEngineMode === "remote_ocr"
  const isOcrProviderAuto = selectedOcrProvider === "auto"
  const isOcrProviderAi = selectedOcrProvider === "aiocr"
  const isOcrProviderPaddleLocal = selectedOcrProvider === "paddle_local"
  const isOcrProviderBaidu = selectedOcrProvider === "baidu"
  const isOcrProviderTesseract = selectedOcrProvider === "tesseract"
  const isAiOcrProviderSelected = isOcrProviderAi || isOcrProviderAuto
  const needsRequiredOcrAiConfig = !isMineruProvider && isOcrProviderAi
  const supportsOptionalOcrAiConfig = false
  const hasAnyOcrAiConfigValue =
    Boolean(settings.ocrAiApiKey.trim()) ||
    Boolean(settings.ocrAiBaseUrl.trim()) ||
    Boolean(settings.ocrAiModel.trim())
  const shouldExpandOptionalOcrAiConfig = false
  const shouldShowAiVendorAdapter = canUseAiOcr && needsRequiredOcrAiConfig
  const shouldShowBaiduConfig = isOcrProviderBaidu
  const shouldShowTesseractConfig = isOcrProviderTesseract
  const shouldShowLocalOcrCheck =
    !isMineruProvider && (isOcrProviderTesseract || isOcrProviderPaddleLocal)
  const availableOcrProviders = isMineruProvider
    ? MINERU_OCR_PROVIDERS
    : isRemoteOcrMode
      ? REMOTE_PARSE_OCR_PROVIDERS
      : LOCAL_PARSE_OCR_PROVIDERS
  const ocrModelsConfigSource = needsRequiredOcrAiConfig
    ? settings.ocrAiApiKey.trim()
      ? "dedicated"
      : "none"
    : "none"
  const ocrModelsApiKey =
    ocrModelsConfigSource === "dedicated"
      ? settings.ocrAiApiKey.trim()
      : ""
  const ocrModelsBaseUrl =
    ocrModelsConfigSource === "dedicated"
      ? getDedicatedOcrAiBaseUrl(settings)
      : ""

  return {
    isMineruProvider,
    isOcrEnabledForCurrentEngine,
    hasBaiduCredentials,
    canUseAiOcr,
    selectedOcrProvider,
    visibleOcrProvider: selectedOcrProvider,
    parseEngineMode,
    currentLayoutAssistMode,
    isLayoutAssistEnabledForCurrentEngine,
    currentOcrLinebreakAssistMode,
    canConfigureOcrLinebreakAssist,
    isRemoteOcrMode,
    isOcrProviderAuto,
    isOcrProviderAi,
    isOcrProviderPaddleLocal,
    isOcrProviderBaidu,
    isOcrProviderTesseract,
    isAiOcrProviderSelected,
    needsRequiredOcrAiConfig,
    supportsOptionalOcrAiConfig,
    hasAnyOcrAiConfigValue,
    shouldExpandOptionalOcrAiConfig,
    shouldShowAiVendorAdapter,
    shouldShowBaiduConfig,
    shouldShowTesseractConfig,
    shouldShowLocalOcrCheck,
    availableOcrProviders,
    ocrModelsConfigSource,
    ocrModelsApiKey,
    ocrModelsBaseUrl,
    predictedGeometryMode: runConfig.predictedGeometryMode,
    predictedGeometryReason: runConfig.predictedGeometryReason,
    runConfig,
  }
}

export const deriveSettingsUiState = resolveOcrSettingsState

export function getRunModelLabel(runConfig: RunConfig): string {
  if (runConfig.parseProvider === "mineru") {
    return "MinerU 云端解析"
  }
  if (runConfig.effectiveOcrProvider === "tesseract" || runConfig.effectiveOcrProvider === "paddle_local") {
    return "本地 OCR（无需远程模型）"
  }
  if (runConfig.effectiveOcrProvider === "baidu") {
    return "百度 OCR"
  }
  if (runConfig.effectiveOcrProvider === "aiocr" && !runConfig.effectiveOcrAiModel) {
    return "未设置 OCR 模型"
  }
  return runConfig.effectiveOcrAiModel || "未设置"
}

export function validateRunConfig(settings: Settings): ValidationResult {
  const ui = resolveOcrSettingsState(settings)
  const run = ui.runConfig

  if (run.parseProvider === "mineru" && !settings.mineruApiToken.trim()) {
    return { ok: false, message: "当前为 MinerU 解析，请先在设置页填写 MinerU API Token。" }
  }

  if (run.parseProvider === "mineru") return { ok: true }

  if (run.effectiveOcrProvider === "baidu") {
    const ok =
      Boolean(settings.ocrBaiduAppId.trim()) &&
      Boolean(settings.ocrBaiduApiKey.trim()) &&
      Boolean(settings.ocrBaiduSecretKey.trim())
    if (!ok) {
      return {
        ok: false,
        message: "当前 OCR 提供方为百度，请在设置页补全 app_id / api_key / secret_key。",
      }
    }
  }

  if (run.effectiveOcrProvider === "aiocr") {
    if (!settings.ocrAiApiKey.trim()) {
      return {
        ok: false,
        message: "显式 AI OCR 不再复用主 AI 配置，请在设置页单独填写 OCR API Key。",
      }
    }
    if (!settings.ocrAiModel.trim()) {
      return { ok: false, message: "当前 OCR 提供方为 AI OCR，请先在设置页选择 OCR 模型。" }
    }
  }

  return { ok: true }
}

export function createJobFormData(
  file: File,
  settings: Settings,
  pageStart?: number,
  pageEnd?: number
): FormData {
  const ui = resolveOcrSettingsState(settings)
  const run = ui.runConfig
  const form = new FormData()

  form.append("file", file)
  form.append("parse_provider", run.parseProvider)
  form.append("provider", run.llmProvider)

  if (run.mainApiKey) form.append("api_key", run.mainApiKey)
  if (run.mainBaseUrl) form.append("base_url", run.mainBaseUrl)
  if (run.mainModel) form.append("model", run.mainModel)

  form.append("enable_layout_assist", String(run.layoutAssistEnabled))
  form.append(
    "layout_assist_apply_image_regions",
    String(Boolean(run.layoutAssistEnabled && settings.layoutAssistApplyImageRegions))
  )
  form.append("enable_ocr", String(run.parseProvider === "mineru" ? false : Boolean(settings.enableOcr)))
  form.append("text_erase_mode", settings.textEraseMode)
  form.append("scanned_page_mode", settings.scannedPageMode)
  const imageBgClearExpandMinPt = toFiniteFloatStringOrUndefined(settings.imageBgClearExpandMinPt)
  const imageBgClearExpandMaxPt = toFiniteFloatStringOrUndefined(settings.imageBgClearExpandMaxPt)
  const imageBgClearExpandRatio = toFiniteFloatStringOrUndefined(settings.imageBgClearExpandRatio)
  const scannedImageRegionMinAreaRatio = toFiniteFloatStringOrUndefined(
    settings.scannedImageRegionMinAreaRatio
  )
  const scannedImageRegionMaxAreaRatio = toFiniteFloatStringOrUndefined(
    settings.scannedImageRegionMaxAreaRatio
  )
  const scannedImageRegionMaxAspectRatio = toFiniteFloatStringOrUndefined(
    settings.scannedImageRegionMaxAspectRatio
  )
  if (imageBgClearExpandMinPt) {
    form.append("image_bg_clear_expand_min_pt", imageBgClearExpandMinPt)
  }
  if (imageBgClearExpandMaxPt) {
    form.append("image_bg_clear_expand_max_pt", imageBgClearExpandMaxPt)
  }
  if (imageBgClearExpandRatio) {
    form.append("image_bg_clear_expand_ratio", imageBgClearExpandRatio)
  }
  if (scannedImageRegionMinAreaRatio) {
    form.append("scanned_image_region_min_area_ratio", scannedImageRegionMinAreaRatio)
  }
  if (scannedImageRegionMaxAreaRatio) {
    form.append("scanned_image_region_max_area_ratio", scannedImageRegionMaxAreaRatio)
  }
  if (scannedImageRegionMaxAspectRatio) {
    form.append("scanned_image_region_max_aspect_ratio", scannedImageRegionMaxAspectRatio)
  }
  form.append("ocr_strict_mode", String(Boolean(settings.ocrStrictMode)))

  if (run.parseProvider === "mineru") {
    form.append("mineru_api_token", settings.mineruApiToken.trim())
    form.append("mineru_model_version", settings.mineruModelVersion)
    form.append("mineru_enable_formula", String(Boolean(settings.mineruEnableFormula)))
    form.append("mineru_enable_table", String(Boolean(settings.mineruEnableTable)))
    form.append("mineru_is_ocr", String(Boolean(settings.mineruIsOcr)))
    if (settings.mineruBaseUrl.trim()) form.append("mineru_base_url", settings.mineruBaseUrl.trim())
    if (settings.mineruLanguage.trim()) form.append("mineru_language", settings.mineruLanguage.trim())
  }

  if (run.parseProvider === "local") {
    form.append("ocr_provider", run.effectiveOcrProvider)

    if (run.shouldAttachOcrAiParams) {
      if (run.effectiveOcrAiKey) form.append("ocr_ai_api_key", run.effectiveOcrAiKey)
      if (run.effectiveOcrAiBaseUrl) form.append("ocr_ai_base_url", run.effectiveOcrAiBaseUrl)
      if (run.effectiveOcrAiModel) form.append("ocr_ai_model", run.effectiveOcrAiModel)
      form.append("ocr_ai_provider", run.effectiveOcrAiProvider)
    }
    if (run.ocrLinebreakAssistMode === "on") {
      form.append("ocr_ai_linebreak_assist", "true")
    } else if (run.ocrLinebreakAssistMode === "off") {
      form.append("ocr_ai_linebreak_assist", "false")
    }

    if (run.effectiveOcrProvider === "baidu") {
      appendBaiduFields(form, settings)
    }

    if (run.effectiveOcrProvider === "tesseract" || run.effectiveOcrProvider === "auto") {
      appendTesseractFields(form, settings)
    }
  }

  if (pageStart && pageEnd) {
    form.append("page_start", String(pageStart))
    form.append("page_end", String(pageEnd))
  }

  return form
}

export function applyParseEngineMode(
  settings: Settings,
  nextMode: ParseEngineMode
): Settings {
  const mainProvider = getResolvedMainProvider(settings)

  if (nextMode === "mineru_cloud") {
    return {
      ...settings,
      provider: "mineru",
      preferredMainProvider: mainProvider,
    }
  }

  if (nextMode === "remote_ocr") {
    return {
      ...settings,
      provider: mainProvider,
      preferredMainProvider: mainProvider,
      ocrProvider:
        settings.ocrProvider === "aiocr" || settings.ocrProvider === "baidu"
          ? settings.ocrProvider
          : "aiocr",
    }
  }

  return {
    ...settings,
    provider: mainProvider,
    preferredMainProvider: mainProvider,
    ocrProvider: settings.ocrProvider === "paddle_local" ? "paddle_local" : "tesseract",
  }
}
