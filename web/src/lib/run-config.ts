import {
  BAIDU_DOC_PARSE_TYPE_LABELS,
  PARSE_ENGINE_MODE_LABELS as SETTINGS_PARSE_ENGINE_MODE_LABELS,
  SILICONFLOW_BASE_URL,
  type OcrAiChainMode,
  type OcrAiLayoutModel,
  type BaiduDocParseType,
  type LayoutAssistMode,
  type MainProvider,
  type OcrProvider,
  type ParseEngineMode,
  type Settings,
} from "./settings.ts"

export type OcrConfigSource = "dedicated" | "none"
export type { ParseEngineMode } from "./settings.ts"

export type RunConfig = {
  parseProvider: "local" | "baidu_doc" | "mineru"
  baiduDocParseType: BaiduDocParseType
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
  ocrAiChainMode: OcrAiChainMode
  ocrAiLayoutModel: OcrAiLayoutModel
  ocrAiPageConcurrency: number
  ocrAiBlockConcurrency: number | null
  ocrAiRequestsPerMinute: number | null
  ocrAiTokensPerMinute: number | null
  ocrAiMaxRetries: number
  ocrAiConfigSource: OcrConfigSource
  layoutAssistChain: ParseEngineMode
  layoutAssistMode: LayoutAssistMode
  layoutAssistEnabled: boolean
  shouldAttachOcrAiParams: boolean
}

export type ValidationResult = {
  ok: boolean
  message?: string
}

export type CreateJobOptions = {
  retainProcessArtifacts?: boolean
}

export type OcrSettingsState = {
  isMineruProvider: boolean
  isBaiduDocParseMode: boolean
  isOcrEnabledForCurrentEngine: boolean
  hasBaiduCredentials: boolean
  canUseAiOcr: boolean
  selectedOcrProvider: OcrProvider
  parseEngineMode: ParseEngineMode
  isOcrProviderPaddleLocal: boolean
  isOcrProviderBaidu: boolean
  isOcrProviderTesseract: boolean
  needsRequiredOcrAiConfig: boolean
  shouldShowAiVendorAdapter: boolean
  shouldShowOcrProviderSelector: boolean
  shouldShowBaiduConfig: boolean
  shouldShowTesseractConfig: boolean
  shouldShowLocalOcrCheck: boolean
  availableOcrProviders: OcrProvider[]
  ocrModelsConfigSource: OcrConfigSource
  ocrModelsApiKey: string
  ocrModelsBaseUrl: string
  isOcrAiChainDirect: boolean
  isOcrAiChainDocParser: boolean
  isOcrAiChainLayoutBlock: boolean
  runConfig: RunConfig
}

export const OCR_PROVIDER_LABELS: Record<OcrProvider, string> = {
  auto: "自动（混合）",
  aiocr: "AIOCR",
  paddle_local: "本地 OCR（PaddleOCR）",
  baidu: "百度 OCR",
  tesseract: "本地 OCR（Tesseract）",
}

const OCR_CONFIG_SOURCE_LABELS: Record<OcrConfigSource, string> = {
  dedicated: "OCR 独立配置",
  none: "未配置",
}

export const PARSE_ENGINE_MODE_LABELS = SETTINGS_PARSE_ENGINE_MODE_LABELS

export const PARSE_ENGINE_OPTIONS: Array<{ id: ParseEngineMode; label: string }> = [
  { id: "baidu_doc", label: PARSE_ENGINE_MODE_LABELS.baidu_doc },
  { id: "remote_ocr", label: PARSE_ENGINE_MODE_LABELS.remote_ocr },
  { id: "local_ocr", label: PARSE_ENGINE_MODE_LABELS.local_ocr },
  { id: "mineru_cloud", label: PARSE_ENGINE_MODE_LABELS.mineru_cloud },
]

export const LOCAL_PARSE_OCR_PROVIDERS: OcrProvider[] = ["tesseract", "paddle_local"]

export const REMOTE_PARSE_OCR_PROVIDERS: OcrProvider[] = ["aiocr"]
export const BAIDU_DOC_PARSE_OCR_PROVIDERS: OcrProvider[] = []
export const MINERU_OCR_PROVIDERS: OcrProvider[] = []

export function getOcrConfigSourceLabel(source: OcrConfigSource): string {
  return OCR_CONFIG_SOURCE_LABELS[source]
}

function getResolvedMainProvider(settings: Settings): MainProvider {
  return settings.parseEngineMode === "mineru_cloud" || settings.provider === "mineru"
    ? settings.preferredMainProvider
    : settings.provider
}

function getPreferredLocalOcrProvider(settings: Settings): OcrProvider {
  const rawProvider = (settings.ocrProvider || "").trim().toLowerCase()
  if (rawProvider === "tesseract" || rawProvider === "paddle_local") {
    return rawProvider
  }
  return "tesseract"
}

function resolveParseEngineMode(settings: Settings): ParseEngineMode {
  const mode = settings.parseEngineMode
  if (
    mode === "local_ocr" ||
    mode === "remote_ocr" ||
    mode === "baidu_doc" ||
    mode === "mineru_cloud"
  ) {
    return mode
  }
  if (settings.provider === "mineru") {
    return "mineru_cloud"
  }
  if (settings.ocrProvider === "baidu") {
    return "baidu_doc"
  }
  if (settings.ocrProvider === "aiocr") {
    return "remote_ocr"
  }
  return "local_ocr"
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
  const parseEngineMode = resolveParseEngineMode(settings)

  if (parseEngineMode === "mineru_cloud" || settings.provider === "mineru") {
    return "auto"
  }

  if (parseEngineMode === "remote_ocr") {
    return "aiocr"
  }

  if (parseEngineMode === "baidu_doc") {
    return "baidu"
  }

  return getPreferredLocalOcrProvider(settings)
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

function toFiniteIntStringOrUndefined(value: string): string | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const n = Number(trimmed)
  if (!Number.isFinite(n) || n < 0) return undefined
  return String(Math.round(n))
}

function toFinitePositiveIntOrNull(value: string): number | null {
  const trimmed = value.trim()
  if (!trimmed) return null
  const n = Number(trimmed)
  if (!Number.isFinite(n) || n <= 0) return null
  return Math.round(n)
}

function appendBaiduFields(form: FormData, settings: Settings) {
  form.append("ocr_baidu_app_id", settings.ocrBaiduAppId.trim())
  form.append("ocr_baidu_api_key", settings.ocrBaiduApiKey.trim())
  form.append("ocr_baidu_secret_key", settings.ocrBaiduSecretKey.trim())
}

function appendBaiduDocFields(form: FormData, settings: Settings) {
  appendBaiduFields(form, settings)
  form.append("baidu_doc_parse_type", settings.baiduDocParseType)
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
  const parseEngineMode = resolveParseEngineMode(settings)
  const parseProvider: RunConfig["parseProvider"] =
    parseEngineMode === "mineru_cloud"
      ? "mineru"
      : parseEngineMode === "baidu_doc"
        ? "baidu_doc"
        : "local"
  const main = getMainProviderConfig(settings)
  const selectedOcrProvider = normalizeVisibleOcrProvider(settings)
  const effectiveOcrProvider = selectedOcrProvider
  const explicitAiOcrSelected = parseEngineMode === "remote_ocr"
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
  const ocrAiChainMode: OcrAiChainMode = explicitAiOcrSelected
    ? settings.ocrAiChainMode
    : "direct"
  const ocrAiLayoutModel: OcrAiLayoutModel = explicitAiOcrSelected
    ? settings.ocrAiLayoutModel
    : "pp_doclayout_v3"
  const ocrAiPageConcurrency = Math.min(
    8,
    Math.max(1, Number(settings.ocrAiPageConcurrency) || 1)
  )
  const ocrAiBlockConcurrency = toFinitePositiveIntOrNull(settings.ocrAiBlockConcurrency)
  const ocrAiRequestsPerMinute = toFinitePositiveIntOrNull(settings.ocrAiRequestsPerMinute)
  const ocrAiTokensPerMinute = toFinitePositiveIntOrNull(settings.ocrAiTokensPerMinute)
  const ocrAiMaxRetries = Math.min(8, Math.max(0, Number(settings.ocrAiMaxRetries) || 0))

  const layoutAssistChain = parseEngineMode
  const layoutAssistMode: LayoutAssistMode = "off"
  const layoutAssistEnabled = false
  const shouldAttachOcrAiParams = explicitAiOcrSelected && Boolean(effectiveOcrAiKey)

  return {
    parseProvider,
    baiduDocParseType: settings.baiduDocParseType,
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
    ocrAiChainMode,
    ocrAiLayoutModel,
    ocrAiPageConcurrency,
    ocrAiBlockConcurrency,
    ocrAiRequestsPerMinute,
    ocrAiTokensPerMinute,
    ocrAiMaxRetries,
    ocrAiConfigSource,
    layoutAssistChain,
    layoutAssistMode,
    layoutAssistEnabled,
    shouldAttachOcrAiParams,
  }
}

export function resolveOcrSettingsState(settings: Settings): OcrSettingsState {
  const runConfig = resolveRunConfig(settings)
  const parseEngineMode = runConfig.layoutAssistChain
  const isMineruProvider = parseEngineMode === "mineru_cloud"
  const isBaiduDocParseMode = parseEngineMode === "baidu_doc"
  const isOcrEnabledForCurrentEngine = !isMineruProvider
  const hasBaiduCredentials =
    Boolean(settings.ocrBaiduApiKey.trim()) &&
    Boolean(settings.ocrBaiduSecretKey.trim())
  const canUseAiOcr = parseEngineMode === "local_ocr" || parseEngineMode === "remote_ocr"
  const selectedOcrProvider = runConfig.selectedOcrProvider
  const isOcrProviderPaddleLocal = selectedOcrProvider === "paddle_local"
  const isOcrProviderBaidu = selectedOcrProvider === "baidu"
  const isOcrProviderTesseract = selectedOcrProvider === "tesseract"
  const needsRequiredOcrAiConfig = parseEngineMode === "remote_ocr"
  const shouldShowAiVendorAdapter = needsRequiredOcrAiConfig
  const shouldShowOcrProviderSelector = parseEngineMode === "local_ocr"
  const shouldShowBaiduConfig = isBaiduDocParseMode || isOcrProviderBaidu
  const shouldShowTesseractConfig = parseEngineMode === "local_ocr" && isOcrProviderTesseract
  const shouldShowLocalOcrCheck =
    !isMineruProvider &&
    !isBaiduDocParseMode &&
    (isOcrProviderTesseract || isOcrProviderPaddleLocal)
  const availableOcrProviders = isMineruProvider
    ? MINERU_OCR_PROVIDERS
    : isBaiduDocParseMode
      ? BAIDU_DOC_PARSE_OCR_PROVIDERS
    : parseEngineMode === "remote_ocr"
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
    isBaiduDocParseMode,
    isOcrEnabledForCurrentEngine,
    hasBaiduCredentials,
    canUseAiOcr,
    selectedOcrProvider,
    parseEngineMode,
    isOcrProviderPaddleLocal,
    isOcrProviderBaidu,
    isOcrProviderTesseract,
    needsRequiredOcrAiConfig,
    shouldShowAiVendorAdapter,
    shouldShowOcrProviderSelector,
    shouldShowBaiduConfig,
    shouldShowTesseractConfig,
    shouldShowLocalOcrCheck,
    availableOcrProviders,
    ocrModelsConfigSource,
    ocrModelsApiKey,
    ocrModelsBaseUrl,
    isOcrAiChainDirect: runConfig.ocrAiChainMode === "direct",
    isOcrAiChainDocParser: runConfig.ocrAiChainMode === "doc_parser",
    isOcrAiChainLayoutBlock: runConfig.ocrAiChainMode === "layout_block",
    runConfig,
  }
}

export const deriveSettingsUiState = resolveOcrSettingsState

export function getRunParseEngineLabel(runConfig: RunConfig): string {
  return PARSE_ENGINE_MODE_LABELS[runConfig.layoutAssistChain]
}

export function getRunModelLabel(runConfig: RunConfig): string {
  if (runConfig.parseProvider === "mineru") {
    return "MinerU 云端解析"
  }
  if (runConfig.parseProvider === "baidu_doc") {
    return BAIDU_DOC_PARSE_TYPE_LABELS[runConfig.baiduDocParseType]
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
  if (runConfig.effectiveOcrProvider === "aiocr" && runConfig.ocrAiChainMode === "layout_block") {
    const model = runConfig.effectiveOcrAiModel || "未设置"
    return `PP-DocLayoutV3 + ${model}`
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

  if (run.parseProvider === "baidu_doc") {
    const ok =
      Boolean(settings.ocrBaiduApiKey.trim()) &&
      Boolean(settings.ocrBaiduSecretKey.trim())
    if (!ok) {
      return {
        ok: false,
        message: "当前为百度解析，请在设置页补全 api_key / secret_key。",
      }
    }
    return { ok: true }
  }

  if (run.effectiveOcrProvider === "baidu") {
    const ok =
      Boolean(settings.ocrBaiduApiKey.trim()) &&
      Boolean(settings.ocrBaiduSecretKey.trim())
    if (!ok) {
      return {
        ok: false,
        message: "当前 OCR 提供方为百度，请在设置页补全 api_key / secret_key。",
      }
    }
  }

  if (run.effectiveOcrProvider === "aiocr") {
    if (!settings.ocrAiApiKey.trim()) {
      return {
        ok: false,
        message: "AIOCR 不再复用主 AI 配置，请在设置页单独填写 OCR API Key。",
      }
    }
    if (!settings.ocrAiModel.trim()) {
      return { ok: false, message: "当前链路为 AIOCR，请先在设置页选择 OCR 模型。" }
    }
    if (
      run.ocrAiChainMode === "doc_parser" &&
      !settings.ocrAiModel.trim().toLowerCase().includes("paddleocr-vl")
    ) {
      return { ok: false, message: "内置文档解析链路仅支持 PaddleOCR-VL 模型。" }
    }
  }

  return { ok: true }
}

export function createJobFormData(
  file: File,
  settings: Settings,
  pageStart?: number,
  pageEnd?: number,
  options?: CreateJobOptions
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

  // Product-side layout assist has been retired for speed-focused runs.
  form.append("enable_layout_assist", "false")
  form.append("layout_assist_apply_image_regions", "false")
  form.append("retain_process_artifacts", String(Boolean(options?.retainProcessArtifacts)))
  form.append("enable_ocr", String(run.parseProvider === "local" ? Boolean(settings.enableOcr) : false))
  form.append("remove_footer_notebooklm", String(Boolean(settings.removeFooterNotebooklm)))
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

  if (run.parseProvider === "baidu_doc") {
    appendBaiduDocFields(form, settings)
  }

  if (run.parseProvider === "local") {
    form.append("ocr_provider", run.effectiveOcrProvider)

    if (run.shouldAttachOcrAiParams) {
      if (run.effectiveOcrAiKey) form.append("ocr_ai_api_key", run.effectiveOcrAiKey)
      if (run.effectiveOcrAiBaseUrl) form.append("ocr_ai_base_url", run.effectiveOcrAiBaseUrl)
      if (run.effectiveOcrAiModel) form.append("ocr_ai_model", run.effectiveOcrAiModel)
      form.append("ocr_ai_provider", run.effectiveOcrAiProvider)
      form.append("ocr_ai_chain_mode", run.ocrAiChainMode)
      form.append("ocr_ai_layout_model", run.ocrAiLayoutModel)
      const paddleDocMaxSidePx = toFiniteIntStringOrUndefined(
        settings.ocrPaddleVlDocparserMaxSidePx
      )
      if (paddleDocMaxSidePx !== undefined) {
        form.append("ocr_paddle_vl_docparser_max_side_px", paddleDocMaxSidePx)
      }
      form.append("ocr_ai_page_concurrency", String(run.ocrAiPageConcurrency))
      const ocrAiBlockConcurrency = toFiniteIntStringOrUndefined(
        settings.ocrAiBlockConcurrency
      )
      if (ocrAiBlockConcurrency !== undefined) {
        form.append("ocr_ai_block_concurrency", ocrAiBlockConcurrency)
      }
      const ocrAiRequestsPerMinute = toFiniteIntStringOrUndefined(
        settings.ocrAiRequestsPerMinute
      )
      if (ocrAiRequestsPerMinute !== undefined) {
        form.append("ocr_ai_requests_per_minute", ocrAiRequestsPerMinute)
      }
      const ocrAiTokensPerMinute = toFiniteIntStringOrUndefined(
        settings.ocrAiTokensPerMinute
      )
      if (ocrAiTokensPerMinute !== undefined) {
        form.append("ocr_ai_tokens_per_minute", ocrAiTokensPerMinute)
      }
      const ocrAiMaxRetries = toFiniteIntStringOrUndefined(settings.ocrAiMaxRetries)
      if (ocrAiMaxRetries !== undefined) {
        form.append("ocr_ai_max_retries", ocrAiMaxRetries)
      }
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
      parseEngineMode: nextMode,
      provider: "mineru",
      preferredMainProvider: mainProvider,
    }
  }

  if (nextMode === "remote_ocr") {
    return {
      ...settings,
      parseEngineMode: nextMode,
      provider: mainProvider,
      preferredMainProvider: mainProvider,
    }
  }

  if (nextMode === "baidu_doc") {
    return {
      ...settings,
      parseEngineMode: nextMode,
      provider: mainProvider,
      preferredMainProvider: mainProvider,
      visualAssistModeBaiduDoc: "off",
    }
  }

  return {
    ...settings,
    parseEngineMode: nextMode,
    provider: mainProvider,
    preferredMainProvider: mainProvider,
    ocrProvider: getPreferredLocalOcrProvider(settings),
  }
}
