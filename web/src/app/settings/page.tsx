"use client"

import * as React from "react"
import {
  CheckIcon,
  KeyRoundIcon,
} from "lucide-react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
} from "@/components/ui/card"
import { HoverHint } from "@/components/ui/hover-hint"
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/select"

import {
  BAIDU_DOC_PARSE_TYPE_LABELS,
  isPaddleOcrVlModelName,
  type BaiduDocParseType,
  SETTINGS_STORAGE_KEY,
  defaultSettings,
  loadStoredSettings,
  type OcrAiProvider,
  type Settings,
} from "@/lib/settings"
import {
  applyParseEngineMode,
  getMainProviderConfig,
  getOcrConfigSourceLabel,
  PARSE_ENGINE_MODE_LABELS,
  PARSE_ENGINE_OPTIONS,
  resolveOcrSettingsState,
} from "@/lib/run-config"
import {
  apiFetch,
  clearStoredApiOrigin,
  getStoredApiOrigin,
  normalizeFetchError,
  resolveApiOrigin,
  setStoredApiOrigin,
} from "@/lib/api"

type SettingsSectionId = "api" | "strategy" | "ocr"

const settingsSectionItems: Array<{
  id: SettingsSectionId
  label: string
  description: string
}> = [
  { id: "api", label: "接口配置", description: "密钥与连接方式" },
  { id: "strategy", label: "处理策略", description: "输出方式与版式处理" },
  { id: "ocr", label: "识别配置", description: "OCR 与文档解析" },
]

function SectionTitle({
  children,
  hint,
}: {
  children: React.ReactNode
  hint?: string
}) {
  return (
    <div className="flex items-center gap-2 font-sans text-sm font-semibold uppercase tracking-[0.14em]">
      <span>{children}</span>
      {hint ? <HoverHint text={hint} /> : null}
    </div>
  )
}

function FieldLabel({
  htmlFor,
  children,
  hint,
}: {
  htmlFor: string
  children: React.ReactNode
  hint?: string
}) {
  return (
    <div className="flex items-center gap-1.5">
      <label className="text-muted-foreground text-xs" htmlFor={htmlFor}>
        {children}
      </label>
      {hint ? <HoverHint text={hint} /> : null}
    </div>
  )
}

function AdvancedReveal({
  show,
  children,
}: {
  show: boolean
  children: React.ReactNode
}) {
  return (
    <div
      aria-hidden={!show}
      className={`grid transition-[grid-template-rows,opacity,transform] duration-300 ${
        show
          ? "grid-rows-[1fr] translate-y-0 opacity-100 ease-out"
          : "pointer-events-none grid-rows-[0fr] -translate-y-1 opacity-0 ease-in"
      }`}
    >
      <div className="min-h-0 overflow-hidden pt-0.5">{children}</div>
    </div>
  )
}

const ocrAiProviderOptions: Array<{ id: OcrAiProvider; label: string }> = [
  { id: "auto", label: "自动识别（推荐）" },
  { id: "openai", label: "OpenAI" },
  { id: "siliconflow", label: "SiliconFlow" },
  { id: "deepseek", label: "DeepSeek" },
  { id: "ppio", label: "PPIO" },
  { id: "novita", label: "Novita" },
]

const ocrAiChainModeOptions: Array<{ id: Settings["ocrAiChainMode"]; label: string }> = [
  { id: "direct", label: "模型直出框和文字" },
  { id: "doc_parser", label: "内置文档解析（PaddleOCR-VL）" },
  { id: "layout_block", label: "本地切块识别" },
]

const ocrAiLayoutModelOptions: Array<{ id: Settings["ocrAiLayoutModel"]; label: string }> = [
  { id: "pp_doclayout_v3", label: "PP-DocLayoutV3" },
]

const baiduDocParseTypeOptions: Array<{ id: BaiduDocParseType; label: string }> = [
  { id: "general", label: BAIDU_DOC_PARSE_TYPE_LABELS.general },
  { id: "paddle_vl", label: BAIDU_DOC_PARSE_TYPE_LABELS.paddle_vl },
]

const ocrProviderLabels: Record<Settings["ocrProvider"], string> = {
  auto: "自动（混合）",
  aiocr: "AIOCR",
  paddle_local: "本地 OCR（PaddleOCR）",
  baidu: "百度 OCR",
  tesseract: "本地 OCR（Tesseract）",
}

type LocalOcrCheckResult = {
  provider: string
  requested_language: string
  requested_languages: string[]
  python_package_available: boolean
  binary_available: boolean
  version: string | null
  available_languages: string[]
  missing_languages: string[]
  model_root_dir?: string | null
  required_models?: string[]
  found_models?: string[]
  missing_models?: string[]
  model_files?: string[]
  issues: string[]
  ready: boolean
  message: string
}

type LocalOcrCheckResponse = {
  ok: boolean
  check: LocalOcrCheckResult
}

type LocalOcrCheckSuiteEntry = {
  runtime: LocalOcrCheckResponse | null
  runtimeError: string | null
  models: LocalOcrCheckResponse | null
  modelsError: string | null
}

type LocalOcrCheckSuiteResult = {
  tesseract: LocalOcrCheckSuiteEntry
  paddle: LocalOcrCheckSuiteEntry
}

type AiOcrCheckSampleItem = {
  text: string
  bbox: number[]
  confidence: number | null
}

type AiOcrCheckResult = {
  provider: string
  model: string
  base_url: string | null
  elapsed_ms: number
  items_count: number
  valid_bbox_items: number
  ready: boolean
  message: string
  error: string | null
  sample_items: AiOcrCheckSampleItem[]
}

type AiOcrCheckResponse = {
  ok: boolean
  check: AiOcrCheckResult
}

export default function SettingsPage() {
  const [settings, setSettings] = React.useState<Settings>(defaultSettings)
  const [settingsHydrated, setSettingsHydrated] = React.useState(false)
  const [lastSavedAt, setLastSavedAt] = React.useState<number | null>(null)
  const [showAdvanced, setShowAdvanced] = React.useState(false)
  const [apiOrigin, setApiOrigin] = React.useState("")
  const [apiOriginInput, setApiOriginInput] = React.useState("")
  const [apiOriginOverrideEnabled, setApiOriginOverrideEnabled] = React.useState(false)
  const [apiOriginResolving, setApiOriginResolving] = React.useState(false)
  const [apiOriginError, setApiOriginError] = React.useState<string | null>(null)
  const [ocrModelOptions, setOcrModelOptions] = React.useState<string[]>([])
  const [ocrModelLoading, setOcrModelLoading] = React.useState(false)
  const [ocrModelError, setOcrModelError] = React.useState<string | null>(null)
  const [localOcrSuiteChecking, setLocalOcrSuiteChecking] = React.useState(false)
  const [localOcrSuite, setLocalOcrSuite] = React.useState<LocalOcrCheckSuiteResult | null>(
    null
  )
  const [localOcrSuiteError, setLocalOcrSuiteError] = React.useState<string | null>(null)
  const [aiOcrChecking, setAiOcrChecking] = React.useState(false)
  const [aiOcrCheck, setAiOcrCheck] = React.useState<AiOcrCheckResponse | null>(null)
  const [aiOcrCheckError, setAiOcrCheckError] = React.useState<string | null>(null)
  const skipNextAutoSaveRef = React.useRef(false)

  React.useEffect(() => {
    setSettings(loadStoredSettings())
    const storedOrigin = getStoredApiOrigin() || ""
    setApiOriginInput(storedOrigin)
    setApiOriginOverrideEnabled(Boolean(storedOrigin))
    setSettingsHydrated(true)
  }, [])

  React.useEffect(() => {
    let mounted = true
    const hasManualOverride = Boolean(getStoredApiOrigin())
    setApiOriginResolving(true)
    setApiOriginError(null)

    void resolveApiOrigin()
      .then((origin) => {
        if (!mounted) return
        setApiOrigin(origin)
        if (!hasManualOverride) {
          setApiOriginInput(origin)
        }
      })
      .catch((e) => {
        if (!mounted) return
        setApiOriginError(normalizeFetchError(e, "API 地址自动探测失败"))
      })
      .finally(() => {
        if (mounted) setApiOriginResolving(false)
      })

    return () => {
      mounted = false
    }
  }, [])

  const ocrState = React.useMemo(() => resolveOcrSettingsState(settings), [settings])
  const isMineruProvider = ocrState.isMineruProvider
  const isBaiduDocParseMode = ocrState.isBaiduDocParseMode
  const isOcrEnabledForCurrentEngine = ocrState.isOcrEnabledForCurrentEngine
  const canUseAiOcr = ocrState.canUseAiOcr
  const selectedOcrProvider = ocrState.selectedOcrProvider
  const parseEngineMode = ocrState.parseEngineMode
  const currentOcrAiChainMode = ocrState.runConfig.ocrAiChainMode
  const currentOcrAiLayoutModel = ocrState.runConfig.ocrAiLayoutModel
  const isOcrProviderPaddleLocal = ocrState.isOcrProviderPaddleLocal
  const isOcrProviderBaidu = ocrState.isOcrProviderBaidu
  const isOcrProviderTesseract = ocrState.isOcrProviderTesseract
  const isOcrAiChainDirect = ocrState.isOcrAiChainDirect
  const isOcrAiChainDocParser = ocrState.isOcrAiChainDocParser
  const isOcrAiChainLayoutBlock = ocrState.isOcrAiChainLayoutBlock
  const needsRequiredOcrAiConfig = ocrState.needsRequiredOcrAiConfig
  const shouldShowLocalOcrCheck = ocrState.shouldShowLocalOcrCheck
  const tesseractSuite = localOcrSuite?.tesseract ?? null
  const paddleSuite = localOcrSuite?.paddle ?? null
  const hasTesseractSuite = Boolean(
    tesseractSuite?.runtime ||
      tesseractSuite?.runtimeError ||
      tesseractSuite?.models ||
      tesseractSuite?.modelsError
  )
  const hasPaddleSuite = Boolean(
    paddleSuite?.runtime ||
      paddleSuite?.runtimeError ||
      paddleSuite?.models ||
      paddleSuite?.modelsError
  )
  const tesseractSuiteReady = Boolean(tesseractSuite?.runtime?.ok && tesseractSuite?.models?.ok)
  const paddleSuiteReady = Boolean(paddleSuite?.runtime?.ok && paddleSuite?.models?.ok)
  const shouldShowOcrProviderSelector = ocrState.shouldShowOcrProviderSelector
  const shouldShowBaiduConfig = ocrState.shouldShowBaiduConfig
  const shouldShowTesseractConfig = ocrState.shouldShowTesseractConfig
  const shouldShowAiVendorAdapter = ocrState.shouldShowAiVendorAdapter
  const mainConfig = getMainProviderConfig(settings)
  const mainModelsApiKeyRaw = mainConfig.apiKey
  const ocrModelsApiKey = ocrState.ocrModelsApiKey
  const ocrModelsBaseUrl = ocrState.ocrModelsBaseUrl
  const ocrModelCapability = isOcrAiChainLayoutBlock ? "vision" : "ocr"
  const visibleOcrModelOptions = React.useMemo(() => {
    if (isOcrAiChainDocParser) {
      return ocrModelOptions.filter((model) => isPaddleOcrVlModelName(model))
    }
    if (isOcrAiChainDirect) {
      return ocrModelOptions.filter((model) => !isPaddleOcrVlModelName(model))
    }
    return ocrModelOptions
  }, [isOcrAiChainDirect, isOcrAiChainDocParser, ocrModelOptions])

  const canLoadOcrModels =
    canUseAiOcr &&
    isOcrEnabledForCurrentEngine &&
    needsRequiredOcrAiConfig &&
    Boolean(ocrModelsApiKey)

  const visibleSectionItems = React.useMemo(
    () =>
      settingsSectionItems.filter((section) => section.id !== "ocr" || isOcrEnabledForCurrentEngine),
    [isOcrEnabledForCurrentEngine]
  )
  const observableSectionItems = visibleSectionItems
  const [activeSection, setActiveSection] = React.useState<SettingsSectionId>("api")

  React.useEffect(() => {
    if (visibleSectionItems.some((section) => section.id === activeSection)) {
      return
    }
    if (visibleSectionItems.length > 0) {
      setActiveSection(visibleSectionItems[0].id)
    }
  }, [activeSection, visibleSectionItems])

  const selectSection = React.useCallback(
    (sectionId: SettingsSectionId) => {
      setActiveSection(sectionId)
      const target = document.getElementById(`settings-section-${sectionId}`)
      if (!target) return
      target.scrollIntoView({ behavior: "smooth", block: "start" })
    },
    []
  )

  React.useEffect(() => {
    if (typeof window === "undefined") return

    let ticking = false

    const syncActiveSectionWithScroll = () => {
      const viewportHeight =
        window.innerHeight || document.documentElement.clientHeight || 0
      const activationLine = Math.min(Math.max(viewportHeight * 0.38, 110), 220)
      const scrolledToBottom =
        window.scrollY + viewportHeight >= document.documentElement.scrollHeight - 8

      if (scrolledToBottom && observableSectionItems.length > 0) {
        const bottomSection = observableSectionItems[observableSectionItems.length - 1]
        setActiveSection((prev) =>
          prev === bottomSection.id ? prev : bottomSection.id
        )
        return
      }

      let nextActive: SettingsSectionId | null = null

      for (const section of observableSectionItems) {
        const target = document.getElementById(`settings-section-${section.id}`)
        if (!target) continue
        if (target.offsetHeight <= 0) continue
        const rect = target.getBoundingClientRect()

        if (rect.top <= activationLine) {
          nextActive = section.id
          continue
        }

        if (!nextActive) {
          nextActive = section.id
        }
        break
      }

      if (!nextActive && observableSectionItems.length > 0) {
        nextActive = observableSectionItems[0].id
      }

      if (nextActive) {
        setActiveSection((prev) => (prev === nextActive ? prev : nextActive))
      }
    }

    const onScrollOrResize = () => {
      if (ticking) return
      ticking = true
      window.requestAnimationFrame(() => {
        ticking = false
        syncActiveSectionWithScroll()
      })
    }

    syncActiveSectionWithScroll()
    window.addEventListener("scroll", onScrollOrResize, { passive: true })
    window.addEventListener("resize", onScrollOrResize)

    return () => {
      window.removeEventListener("scroll", onScrollOrResize)
      window.removeEventListener("resize", onScrollOrResize)
    }
  }, [observableSectionItems])

  React.useEffect(() => {
    if (!canLoadOcrModels) {
      setOcrModelOptions([])
      setOcrModelError(null)
      return
    }

    let mounted = true
    const controller = new AbortController()
    const timer = setTimeout(async () => {
      setOcrModelLoading(true)
      setOcrModelError(null)

      try {
        const payload: Record<string, string> = {
          provider: "openai",
          api_key: ocrModelsApiKey,
          capability: ocrModelCapability,
        }
        if (ocrModelsBaseUrl) {
          payload.base_url = ocrModelsBaseUrl
        }

        const response = await apiFetch("/models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        })

        if (!response.ok) {
          const body = await response.json().catch(() => null)
          throw new Error(body?.message || "OCR 模型列表加载失败")
        }

        const body = await response.json().catch(() => null)
        const models = Array.isArray(body?.models)
          ? body.models.filter((m: unknown) => typeof m === "string")
          : []

        if (!mounted) return
        setOcrModelOptions(models)
      } catch (e) {
        if (!mounted || controller.signal.aborted) return
        setOcrModelError(normalizeFetchError(e, "OCR 模型列表加载失败"))
        setOcrModelOptions([])
      } finally {
        if (mounted) setOcrModelLoading(false)
      }
    }, 400)

    return () => {
      mounted = false
      controller.abort()
      clearTimeout(timer)
    }
  }, [
    canLoadOcrModels,
    ocrModelsApiKey,
    ocrModelsBaseUrl,
    ocrModelCapability,
    settings.ocrAiModel,
    selectedOcrProvider,
    isOcrEnabledForCurrentEngine,
  ])

  React.useEffect(() => {
    if (ocrModelLoading || !canLoadOcrModels) return
    if (!settings.ocrAiModel.trim()) return
    if (!visibleOcrModelOptions.length) return
    if (visibleOcrModelOptions.includes(settings.ocrAiModel)) return
    setSettings((prev) => ({ ...prev, ocrAiModel: "" }))
  }, [canLoadOcrModels, ocrModelLoading, settings.ocrAiModel, visibleOcrModelOptions])

  React.useEffect(() => {
    if (!settingsHydrated) return
    if (skipNextAutoSaveRef.current) {
      skipNextAutoSaveRef.current = false
      return
    }
    const timer = window.setTimeout(() => {
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings))
      setLastSavedAt(Date.now())
    }, 350)
    return () => {
      window.clearTimeout(timer)
    }
  }, [settings, settingsHydrated])

  const onSave = React.useCallback(() => {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings))
    setLastSavedAt(Date.now())
    toast.success("设置已保存")
  }, [settings])

  const onResetScannedImageTuning = React.useCallback(() => {
    setSettings((s) => ({
      ...s,
      imageBgClearExpandMinPt: defaultSettings.imageBgClearExpandMinPt,
      imageBgClearExpandMaxPt: defaultSettings.imageBgClearExpandMaxPt,
      imageBgClearExpandRatio: defaultSettings.imageBgClearExpandRatio,
      scannedImageRegionMinAreaRatio: defaultSettings.scannedImageRegionMinAreaRatio,
      scannedImageRegionMaxAreaRatio: defaultSettings.scannedImageRegionMaxAreaRatio,
      scannedImageRegionMaxAspectRatio: defaultSettings.scannedImageRegionMaxAspectRatio,
    }))
    toast.success("图片阈值已恢复默认值")
  }, [])

  const onClear = React.useCallback(() => {
    localStorage.removeItem(SETTINGS_STORAGE_KEY)
    skipNextAutoSaveRef.current = true
    setSettings(defaultSettings)
    setLastSavedAt(null)
    setLocalOcrSuite(null)
    setLocalOcrSuiteError(null)
    setAiOcrCheck(null)
    setAiOcrCheckError(null)
    toast("已清空本地设置")
  }, [])

  const onSaveApiOrigin = React.useCallback(async () => {
    setApiOriginError(null)
    setApiOriginResolving(true)
    try {
      if (apiOriginInput.trim()) {
        const normalized = setStoredApiOrigin(apiOriginInput)
        setApiOriginInput(normalized)
        setApiOriginOverrideEnabled(true)
      } else {
        clearStoredApiOrigin()
        setApiOriginOverrideEnabled(false)
      }
      const resolved = await resolveApiOrigin({ force: true })
      setApiOrigin(resolved)
      if (!apiOriginInput.trim()) {
        setApiOriginInput(resolved)
      }
      toast.success("API 地址已更新")
    } catch (e) {
      const message = normalizeFetchError(e, "API 地址更新失败")
      setApiOriginError(message)
      toast.error(message)
    } finally {
      setApiOriginResolving(false)
    }
  }, [apiOriginInput])

  const onAutoDetectApiOrigin = React.useCallback(async () => {
    setApiOriginError(null)
    setApiOriginResolving(true)
    try {
      clearStoredApiOrigin()
      setApiOriginOverrideEnabled(false)
      const resolved = await resolveApiOrigin({ force: true })
      setApiOrigin(resolved)
      setApiOriginInput(resolved)
      toast.success("已切换为自动探测 API 地址")
    } catch (e) {
      const message = normalizeFetchError(e, "自动探测失败")
      setApiOriginError(message)
      toast.error(message)
    } finally {
      setApiOriginResolving(false)
    }
  }, [])

  const requestLocalOcrCheck = React.useCallback(
    async (provider: string, language: string, fallbackError: string) => {
      const response = await apiFetch("/jobs/ocr/local/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, language }),
      })
      const body = (await response.json().catch(() => null)) as
        | LocalOcrCheckResponse
        | { message?: string }
        | null
      if (!response.ok) {
        throw new Error((body as { message?: string } | null)?.message || fallbackError)
      }
      return body as LocalOcrCheckResponse
    },
    []
  )

  const runSingleLocalOcrSuite = React.useCallback(
    async (provider: "tesseract" | "paddle"): Promise<LocalOcrCheckSuiteEntry> => {
      const providerLabel = provider === "paddle" ? "PaddleOCR" : "Tesseract"
      const language =
        provider === "paddle"
          ? "ch"
          : settings.ocrTesseractLanguage.trim() || "chi_sim+eng"
      const modelProvider = provider === "paddle" ? "paddle_models" : "tesseract_models"

      const [runtimeResult, modelResult] = await Promise.allSettled([
        requestLocalOcrCheck(provider, language, `${providerLabel} 运行环境检测失败`),
        requestLocalOcrCheck(modelProvider, language, `${providerLabel} 模型检测失败`),
      ])

      const runtime = runtimeResult.status === "fulfilled" ? runtimeResult.value : null
      const runtimeError =
        runtimeResult.status === "rejected"
          ? normalizeFetchError(runtimeResult.reason, `${providerLabel} 运行环境检测失败`)
          : null
      const models = modelResult.status === "fulfilled" ? modelResult.value : null
      const modelsError =
        modelResult.status === "rejected"
          ? normalizeFetchError(modelResult.reason, `${providerLabel} 模型检测失败`)
          : null

      return { runtime, runtimeError, models, modelsError }
    },
    [requestLocalOcrCheck, settings.ocrTesseractLanguage]
  )

  const onRunLocalOcrSuite = React.useCallback(async () => {
    setLocalOcrSuiteChecking(true)
    setLocalOcrSuiteError(null)

    try {
      const [tesseract, paddle] = await Promise.all([
        runSingleLocalOcrSuite("tesseract"),
        runSingleLocalOcrSuite("paddle"),
      ])

      setLocalOcrSuite({ tesseract, paddle })

      const tesseractReady = Boolean(tesseract.runtime?.ok && tesseract.models?.ok)
      const paddleReady = Boolean(paddle.runtime?.ok && paddle.models?.ok)
      const hasAnyHardError = Boolean(
        tesseract.runtimeError || tesseract.modelsError || paddle.runtimeError || paddle.modelsError
      )

      if (tesseractReady && paddleReady) {
        toast.success("本地 OCR 运行环境与模型均已就绪")
      } else if (hasAnyHardError) {
        setLocalOcrSuiteError("综合检测部分失败，请查看各卡片错误详情。")
        toast.error("本地 OCR 综合检测部分失败")
      } else {
        toast("综合检测完成，请根据结果补齐缺失项")
      }
    } catch (e) {
      const message = normalizeFetchError(e, "本地 OCR 综合检测失败")
      setLocalOcrSuiteError(message)
      toast.error(message)
    } finally {
      setLocalOcrSuiteChecking(false)
    }
  }, [runSingleLocalOcrSuite])

  const onCheckAiOcrModel = React.useCallback(async () => {
    const apiKey = ocrModelsApiKey.trim()
    const baseUrl = ocrModelsBaseUrl.trim()
    const model = settings.ocrAiModel.trim()
    const provider = (settings.ocrAiProvider || "auto").trim() || "auto"
    const sourceLabel = getOcrConfigSourceLabel(ocrState.ocrModelsConfigSource)

    if (!apiKey) {
      const message =
        needsRequiredOcrAiConfig
          ? "AIOCR 需要独立 OCR API Key"
          : `请先补充可用的 AIOCR 配置（当前来源：${sourceLabel}）`
      setAiOcrCheck(null)
      setAiOcrCheckError(message)
      toast.error(message)
      return
    }
    if (!model) {
      const message = "请先选择 OCR 模型"
      setAiOcrCheck(null)
      setAiOcrCheckError(message)
      toast.error(message)
      return
    }

    setAiOcrChecking(true)
    setAiOcrCheckError(null)
    try {
      const payload: Record<string, string | number> = {
        provider,
        api_key: apiKey,
        model,
        ocr_ai_chain_mode: currentOcrAiChainMode,
        ocr_ai_layout_model: currentOcrAiLayoutModel,
      }
      if (baseUrl) payload.base_url = baseUrl
      const paddleDocMaxSidePx = Number(settings.ocrPaddleVlDocparserMaxSidePx.trim())
      if (Number.isFinite(paddleDocMaxSidePx) && paddleDocMaxSidePx >= 0) {
        payload.ocr_paddle_vl_docparser_max_side_px = Math.round(paddleDocMaxSidePx)
      }
      const blockConcurrency = Number(settings.ocrAiBlockConcurrency.trim())
      if (Number.isFinite(blockConcurrency) && blockConcurrency > 0) {
        payload.ocr_ai_block_concurrency = Math.round(blockConcurrency)
      }
      const requestsPerMinute = Number(settings.ocrAiRequestsPerMinute.trim())
      if (Number.isFinite(requestsPerMinute) && requestsPerMinute > 0) {
        payload.ocr_ai_requests_per_minute = Math.round(requestsPerMinute)
      }
      const tokensPerMinute = Number(settings.ocrAiTokensPerMinute.trim())
      if (Number.isFinite(tokensPerMinute) && tokensPerMinute > 0) {
        payload.ocr_ai_tokens_per_minute = Math.round(tokensPerMinute)
      }
      const maxRetries = Number(settings.ocrAiMaxRetries.trim())
      if (Number.isFinite(maxRetries) && maxRetries >= 0) {
        payload.ocr_ai_max_retries = Math.round(maxRetries)
      }

      const response = await apiFetch("/jobs/ocr/ai/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      const body = (await response.json().catch(() => null)) as
        | AiOcrCheckResponse
        | { message?: string }
        | null

      if (!response.ok) {
        throw new Error((body as { message?: string } | null)?.message || "OCR 能力验证失败")
      }

      const result = body as AiOcrCheckResponse
      setAiOcrCheck(result)
      if (result.ok) {
        toast.success("OCR 能力验证通过")
      } else {
        toast.error("OCR 能力验证未通过")
      }
    } catch (e) {
      const message = normalizeFetchError(e, "OCR 能力验证失败")
      setAiOcrCheck(null)
      setAiOcrCheckError(message)
      toast.error(message)
    } finally {
      setAiOcrChecking(false)
    }
  }, [
    currentOcrAiChainMode,
    currentOcrAiLayoutModel,
    needsRequiredOcrAiConfig,
    ocrModelsApiKey,
    ocrModelsBaseUrl,
    ocrState.ocrModelsConfigSource,
    settings.ocrAiBlockConcurrency,
    settings.ocrAiMaxRetries,
    settings.ocrAiModel,
    settings.ocrAiProvider,
    settings.ocrPaddleVlDocparserMaxSidePx,
    settings.ocrAiRequestsPerMinute,
    settings.ocrAiTokensPerMinute,
  ])

  return (
    <div className="min-h-dvh bg-background">
      <div className="mx-auto w-full max-w-[1440px] px-4 py-6 md:px-6 md:py-10">
        <header className="editorial-page-header newsprint-texture page-enter border border-border bg-background p-5 md:p-6">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <div className="font-mono text-xs uppercase tracking-[0.16em] text-muted-foreground">
                  参数设置
                </div>
                <Badge variant="outline" className="font-sans text-[11px] uppercase tracking-[0.12em]">
                  参数管理
                </Badge>
              </div>
              <div>
                <h1 className="font-serif text-4xl leading-[0.95] tracking-tight md:text-5xl">
                  处理设置
                </h1>
                <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground md:text-base">
                  常用设置默认直接显示，调参与诊断项收在高级参数里。设置会保存在当前浏览器，方便下次继续使用。
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">浏览器保存</Badge>
              <Badge className="editorial-pill">可选 OCR</Badge>
              <Badge variant="outline">与首页联动</Badge>
            </div>
          </div>
        </header>

        <div className="page-enter page-enter-delay-1 mt-4 grid gap-4 xl:grid-cols-[15rem_minmax(0,1fr)] xl:gap-6">
          <aside className="hidden xl:block">
            <div className="sticky top-24 space-y-1 pr-3">
              <div className="space-y-1 pt-1">
                <div className="pl-3 font-sans text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                  设置目录
                </div>
                <div className="grid gap-1">
                  {visibleSectionItems.map((section) => {
                    const active = activeSection === section.id
                    return (
                      <button
                        key={section.id}
                        type="button"
                        onClick={() => selectSection(section.id)}
                        className={`nav-highlight group border-l-2 px-3 py-2.5 text-left ${
                          active
                            ? "nav-highlight-active border-primary text-foreground"
                            : "nav-highlight-inactive border-transparent text-muted-foreground hover:border-border hover:text-foreground"
                        }`}
                      >
                        <div className="text-sm font-medium">{section.label}</div>
                        <div className="mt-0.5 text-xs opacity-80 transition-colors">
                          {section.description}
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>
          </aside>

          <div className="page-enter page-enter-delay-2 min-w-0 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 border border-border bg-background/90 px-3 py-2">
              <div className="font-mono text-xs uppercase tracking-[0.14em] text-muted-foreground">
                配置操作
              </div>
              <div className="flex items-center gap-2">
                {lastSavedAt ? (
                  <div className="hidden font-mono text-xs uppercase tracking-[0.14em] text-muted-foreground sm:block">
                    自动保存已开启
                  </div>
                ) : null}
                <Button type="button" variant="outline" size="sm" onClick={onClear}>
                  清空本地配置
                </Button>
                <Button type="button" size="sm" onClick={onSave}>
                  立即保存
                </Button>
              </div>
            </div>

            <Card className="border border-border py-0 hard-shadow-hover">
              <CardContent className="grid gap-3 p-5">
                <div className="flex items-center justify-between">
                  <div className="font-sans text-sm font-semibold uppercase tracking-[0.14em]">
                    解析引擎
                  </div>
                  <Badge variant="secondary">{PARSE_ENGINE_MODE_LABELS[parseEngineMode]}</Badge>
                </div>

                <div className="grid grid-cols-1 gap-2 sm:grid-cols-4">
                  {PARSE_ENGINE_OPTIONS.map((p) => (
                    <Button
                      key={p.id}
                      type="button"
                      variant={p.id === parseEngineMode ? "default" : "outline"}
                      onClick={() => setSettings((s) => applyParseEngineMode(s, p.id))}
                      className="justify-center"
                    >
                      {p.id === parseEngineMode ? (
                        <CheckIcon className="size-4" />
                      ) : (
                        <KeyRoundIcon className="size-4" />
                      )}
                      {p.label}
                    </Button>
                  ))}
                </div>

                <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-muted/20 px-3 py-2">
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-foreground">高级参数与诊断</div>
                    <div className="text-xs text-muted-foreground">
                      默认隐藏调参、联调与检测项。
                    </div>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => setShowAdvanced((previous) => !previous)}
                  >
                    {showAdvanced ? "收起" : "展开"}
                  </Button>
                </div>
              </CardContent>
            </Card>

            <div className="flex items-center gap-2 overflow-x-auto rounded-md border border-border bg-background p-2 xl:hidden">
              {visibleSectionItems.map((section) => {
                const active = activeSection === section.id
                return (
                  <button
                    key={section.id}
                    type="button"
                    onClick={() => selectSection(section.id)}
                    className={`shrink-0 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                      active
                        ? "border-primary bg-primary/10"
                        : "border-border bg-background hover:bg-muted/50"
                    }`}
                  >
                    {section.label}
                  </button>
                )
              })}
            </div>

            <Card className="border border-border py-0">
              <CardContent className="!px-0">
            <section
              id="settings-section-api"
              className="scroll-mt-24 grid gap-4 border-b border-border p-5"
            >
              <SectionTitle
                hint={
                  isMineruProvider
                    ? undefined
                    : isBaiduDocParseMode
                      ? "当前模式将使用下方文档解析配置中的参数。"
                      : "当前模式将使用下方 OCR 配置中的参数。"
                }
              >
                接口配置
              </SectionTitle>

              <AdvancedReveal show={showAdvanced}>
                <div className="grid gap-2 border border-border bg-muted/30 p-3">
                  <FieldLabel
                    htmlFor="api-origin-input"
                    hint="仅在本地联调或特殊部署时需要修改。"
                  >
                    后端 API 地址
                  </FieldLabel>
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="font-mono break-all text-xs">{apiOrigin}</div>
                    <Badge variant="outline">
                      {apiOriginOverrideEnabled ? "手动覆盖" : "自动探测"}
                    </Badge>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto]">
                    <Input
                      id="api-origin-input"
                      type="text"
                      autoComplete="off"
                      value={apiOriginInput}
                      onChange={(e) => setApiOriginInput(e.target.value)}
                      placeholder="留空为自动探测"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => void onSaveApiOrigin()}
                      disabled={apiOriginResolving}
                    >
                      {apiOriginResolving ? "应用中..." : "应用地址"}
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      onClick={() => void onAutoDetectApiOrigin()}
                      disabled={apiOriginResolving}
                    >
                      自动探测
                    </Button>
                  </div>
                  {apiOriginError ? (
                    <div className="text-xs text-destructive">{apiOriginError}</div>
                  ) : null}
                </div>
              </AdvancedReveal>

              {isMineruProvider ? (
                <>
                  <div className="grid gap-2">
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="mineru-token"
                    >
                      MinerU Token（主 Key）
                    </label>
                    <Input
                      id="mineru-token"
                      type="password"
                      autoComplete="off"
                      value={settings.mineruApiToken}
                      onChange={(e) =>
                        setSettings((s) => ({ ...s, mineruApiToken: e.target.value }))
                      }
                      placeholder="官网申请 Token"
                    />
                  </div>
                  <AdvancedReveal show={showAdvanced}>
                    <div className="grid gap-4">
                      <div className="grid gap-2">
                        <label
                          className="text-muted-foreground text-xs"
                          htmlFor="mineru-base-url"
                        >
                          MinerU Base URL（可选）
                        </label>
                        <Input
                          id="mineru-base-url"
                          type="text"
                          autoComplete="off"
                          value={settings.mineruBaseUrl}
                          onChange={(e) =>
                            setSettings((s) => ({ ...s, mineruBaseUrl: e.target.value }))
                          }
                          placeholder="https://mineru.net"
                        />
                      </div>

                      <div className="grid gap-2">
                        <label
                          className="text-muted-foreground text-xs"
                          htmlFor="mineru-model-version"
                        >
                          MinerU 模型版本
                        </label>
                        <Select
                          id="mineru-model-version"
                          value={settings.mineruModelVersion}
                          onChange={(e) =>
                            setSettings((s) => ({
                              ...s,
                              mineruModelVersion: e.target.value as Settings["mineruModelVersion"],
                            }))
                          }
                        >
                          <option value="pipeline">pipeline</option>
                          <option value="vlm">vlm</option>
                          <option value="MinerU-HTML">MinerU-HTML</option>
                        </Select>
                      </div>

                      <div className="grid gap-2">
                        <label
                          className="text-muted-foreground text-xs"
                          htmlFor="mineru-language"
                        >
                          MinerU 语言（可选）
                        </label>
                        <Input
                          id="mineru-language"
                          type="text"
                          autoComplete="off"
                          value={settings.mineruLanguage}
                          onChange={(e) =>
                            setSettings((s) => ({ ...s, mineruLanguage: e.target.value }))
                          }
                          placeholder="ch"
                        />
                      </div>

                      <label className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          className="h-4 w-4 accent-[#111111]"
                          checked={settings.mineruEnableFormula}
                          onChange={(e) =>
                            setSettings((s) => ({
                              ...s,
                              mineruEnableFormula: e.target.checked,
                            }))
                          }
                        />
                        启用 MinerU 公式识别
                      </label>

                      <label className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          className="h-4 w-4 accent-[#111111]"
                          checked={settings.mineruEnableTable}
                          onChange={(e) =>
                            setSettings((s) => ({
                              ...s,
                              mineruEnableTable: e.target.checked,
                            }))
                          }
                        />
                        启用 MinerU 表格识别
                      </label>

                      <label className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          className="h-4 w-4 accent-[#111111]"
                          checked={settings.mineruIsOcr}
                          onChange={(e) =>
                            setSettings((s) => ({
                              ...s,
                              mineruIsOcr: e.target.checked,
                            }))
                          }
                        />
                        启用 MinerU OCR
                      </label>
                    </div>
                  </AdvancedReveal>
                </>
              ) : (
                <></>
              )}
            </section>

            <section
              id="settings-section-strategy"
              className="scroll-mt-24 grid gap-4 border-b border-border p-5"
            >
              <SectionTitle
                hint={
                  isMineruProvider
                    ? "MinerU 使用自身解析链路。"
                    : "这里只保留会直接影响结果的常用选项。"
                }
              >
                处理策略
              </SectionTitle>

              {isMineruProvider ? (
                <>
                  <AdvancedReveal show={showAdvanced}>
                    <div className="grid gap-2 pt-1">
                      <FieldLabel
                        htmlFor="text-erase-mode"
                        hint="推荐使用纯色填充；智能消除仅适合特殊页面。"
                      >
                        文字消除模式
                      </FieldLabel>
                      <Select
                        id="text-erase-mode"
                        value={settings.textEraseMode}
                        onChange={(e) =>
                          setSettings((s) => ({
                            ...s,
                            textEraseMode: e.target.value as Settings["textEraseMode"],
                          }))
                        }
                      >
                        <option value="fill">纯色填充（推荐）</option>
                        <option value="smart">智能消除</option>
                      </Select>
                    </div>
                  </AdvancedReveal>
                </>
              ) : null}

              <AdvancedReveal show={!isMineruProvider && showAdvanced}>
                <div className="grid gap-2 pt-1">
                  <FieldLabel
                    htmlFor="text-erase-mode"
                    hint="推荐使用纯色填充；智能消除仅适合特殊页面。"
                  >
                    文字消除模式
                  </FieldLabel>
                  <Select
                    id="text-erase-mode"
                    value={settings.textEraseMode}
                    onChange={(e) =>
                      setSettings((s) => ({
                        ...s,
                        textEraseMode: e.target.value as Settings["textEraseMode"],
                      }))
                    }
                  >
                    <option value="fill">纯色填充（推荐）</option>
                    <option value="smart">智能消除</option>
                  </Select>
                </div>
              </AdvancedReveal>

              <div className="grid gap-2">
                <FieldLabel
                  htmlFor="scanned-page-mode"
                  hint={
                    settings.pptGenerationMode === "fast"
                      ? "快速模式固定保留整页背景；此项仅在精准模式下生效。"
                      : "决定图片是拆成可编辑元素，还是保留在整页背景中。"
                  }
                >
                  扫描页图片处理方式
                </FieldLabel>
                <Select
                  id="scanned-page-mode"
                  value={settings.scannedPageMode}
                  onChange={(e) =>
                    setSettings((s) => ({
                      ...s,
                      scannedPageMode: e.target.value as Settings["scannedPageMode"],
                    }))
                  }
                >
                  <option value="segmented">图片拆出来（可单独编辑）</option>
                  <option value="fullpage">图片留在整页背景里（更像原图）</option>
                </Select>
              </div>

              <div className="flex items-center gap-1.5">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-[#111111]"
                    checked={settings.removeFooterNotebooklm}
                    onChange={(e) =>
                      setSettings((s) => ({
                        ...s,
                        removeFooterNotebooklm: e.target.checked,
                      }))
                    }
                  />
                  <span>删除页脚 NotebookLM</span>
                </label>
                <HoverHint text="仅删除页脚里的 NotebookLM 字样，避免误删正文内容。" />
              </div>

              <AdvancedReveal show={showAdvanced}>
                <div className="grid gap-3 rounded-md border border-border/70 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 font-sans text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                      <span>图片底图清除与图块阈值</span>
                      <HoverHint text="仅在图片底图残留或图块误判时再微调。" />
                    </div>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-7 px-2 text-[11px]"
                      onClick={onResetScannedImageTuning}
                    >
                      恢复默认阈值
                    </Button>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="grid gap-1.5">
                      <FieldLabel
                        htmlFor="image-bg-clear-min-pt"
                        hint={`默认值：${defaultSettings.imageBgClearExpandMinPt}`}
                      >
                        清除扩边最小值（pt）
                      </FieldLabel>
                      <Input
                        id="image-bg-clear-min-pt"
                        type="number"
                        inputMode="decimal"
                        step="0.01"
                        value={settings.imageBgClearExpandMinPt}
                        onChange={(e) =>
                          setSettings((s) => ({ ...s, imageBgClearExpandMinPt: e.target.value }))
                        }
                      />
                    </div>

                    <div className="grid gap-1.5">
                      <FieldLabel
                        htmlFor="image-bg-clear-max-pt"
                        hint={`默认值：${defaultSettings.imageBgClearExpandMaxPt}`}
                      >
                        清除扩边最大值（pt）
                      </FieldLabel>
                      <Input
                        id="image-bg-clear-max-pt"
                        type="number"
                        inputMode="decimal"
                        step="0.01"
                        value={settings.imageBgClearExpandMaxPt}
                        onChange={(e) =>
                          setSettings((s) => ({ ...s, imageBgClearExpandMaxPt: e.target.value }))
                        }
                      />
                    </div>

                    <div className="grid gap-1.5">
                      <FieldLabel
                        htmlFor="image-bg-clear-ratio"
                        hint={`默认值：${defaultSettings.imageBgClearExpandRatio}`}
                      >
                        清除扩边比例
                      </FieldLabel>
                      <Input
                        id="image-bg-clear-ratio"
                        type="number"
                        inputMode="decimal"
                        step="0.001"
                        value={settings.imageBgClearExpandRatio}
                        onChange={(e) =>
                          setSettings((s) => ({ ...s, imageBgClearExpandRatio: e.target.value }))
                        }
                      />
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="grid gap-1.5">
                      <FieldLabel
                        htmlFor="scanned-min-area-ratio"
                        hint={`默认值：${defaultSettings.scannedImageRegionMinAreaRatio}`}
                      >
                        图块最小面积比例
                      </FieldLabel>
                      <Input
                        id="scanned-min-area-ratio"
                        type="number"
                        inputMode="decimal"
                        step="0.0001"
                        value={settings.scannedImageRegionMinAreaRatio}
                        onChange={(e) =>
                          setSettings((s) => ({
                            ...s,
                            scannedImageRegionMinAreaRatio: e.target.value,
                          }))
                        }
                      />
                    </div>

                    <div className="grid gap-1.5">
                      <FieldLabel
                        htmlFor="scanned-max-area-ratio"
                        hint={`默认值：${defaultSettings.scannedImageRegionMaxAreaRatio}`}
                      >
                        图块最大面积比例
                      </FieldLabel>
                      <Input
                        id="scanned-max-area-ratio"
                        type="number"
                        inputMode="decimal"
                        step="0.01"
                        value={settings.scannedImageRegionMaxAreaRatio}
                        onChange={(e) =>
                          setSettings((s) => ({
                            ...s,
                            scannedImageRegionMaxAreaRatio: e.target.value,
                          }))
                        }
                      />
                    </div>

                    <div className="grid gap-1.5">
                      <FieldLabel
                        htmlFor="scanned-max-aspect-ratio"
                        hint={`默认值：${defaultSettings.scannedImageRegionMaxAspectRatio}`}
                      >
                        图块最大长宽比
                      </FieldLabel>
                      <Input
                        id="scanned-max-aspect-ratio"
                        type="number"
                        inputMode="decimal"
                        step="0.1"
                        value={settings.scannedImageRegionMaxAspectRatio}
                        onChange={(e) =>
                          setSettings((s) => ({
                            ...s,
                            scannedImageRegionMaxAspectRatio: e.target.value,
                          }))
                        }
                      />
                    </div>
                  </div>
                </div>
              </AdvancedReveal>
            </section>

            {isOcrEnabledForCurrentEngine ? (
              <section
                id="settings-section-ocr"
                className="scroll-mt-24 grid gap-4 p-5"
              >
              <SectionTitle
                hint={
                  isBaiduDocParseMode
                    ? "百度解析会直接返回结构化结果。"
                    : undefined
                }
              >
                {isBaiduDocParseMode ? "文档解析配置" : "OCR 配置"}
              </SectionTitle>

              {shouldShowOcrProviderSelector ? (
                <div className="grid gap-2">
                  <FieldLabel
                    htmlFor="ocr-provider"
                    hint={
                      isOcrProviderPaddleLocal
                        ? "使用本地 OCR（PaddleOCR）链路，适合纯本地部署。"
                        : isOcrProviderBaidu
                          ? "使用百度 OCR 直接识别。"
                          : "使用本地 OCR（Tesseract）链路，适合纯本地部署。"
                    }
                  >
                    OCR 提供方
                  </FieldLabel>
                  <Select
                    id="ocr-provider"
                    value={selectedOcrProvider}
                    onChange={(e) =>
                      setSettings((s) => ({
                        ...s,
                        ocrProvider: e.target.value as Settings["ocrProvider"],
                      }))
                    }
                  >
                    {ocrState.availableOcrProviders.map((providerId) => (
                      <option key={providerId} value={providerId}>
                        {ocrProviderLabels[providerId]}
                      </option>
                    ))}
                  </Select>
                </div>
              ) : isBaiduDocParseMode ? (
                <div className="flex flex-wrap gap-2 text-xs">
                  <Badge variant="outline">{PARSE_ENGINE_MODE_LABELS[parseEngineMode]}</Badge>
                  <HoverHint text="百度解析会直接返回结构化结果，这里只需填写对应凭据。" />
                </div>
              ) : null}

              <AdvancedReveal show={showAdvanced && !isBaiduDocParseMode}>
                <>
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      className="h-4 w-4 accent-[#111111]"
                      checked={settings.ocrStrictMode}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          ocrStrictMode: e.target.checked,
                        }))
                      }
                    />
                    <span>OCR 严格模式</span>
                    <HoverHint text="关闭后会尽量继续完成任务，但识别失败的页面可能按图片页保留。" />
                  </label>
                </>
              </AdvancedReveal>

              <AdvancedReveal show={showAdvanced && shouldShowAiVendorAdapter}>
                <div className="grid gap-3">
                  <div className="grid gap-2">
                    <FieldLabel htmlFor="ocr-ai-provider" hint="仅在切换服务商时需要修改。">
                      AIOCR 厂商适配
                    </FieldLabel>
                    <Select
                      id="ocr-ai-provider"
                      value={settings.ocrAiProvider}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          ocrAiProvider: e.target.value as OcrAiProvider,
                        }))
                      }
                    >
                      {ocrAiProviderOptions.map((option) => (
                        <option key={option.id} value={option.id}>
                          {option.label}
                        </option>
                      ))}
                    </Select>
                  </div>
                </div>
              </AdvancedReveal>

              {needsRequiredOcrAiConfig ? (
                <div className="grid gap-3 border border-border bg-muted/20 p-3">
                  <div className="flex items-center gap-2 font-sans text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                    <span>专用 OCR 接口参数</span>
                    <HoverHint text="这里只保留当前链路需要的字段。" />
                  </div>

                  <div className="grid gap-2">
                    <FieldLabel htmlFor="ocr-ai-api-key">OCR API Key</FieldLabel>
                    <Input
                      id="ocr-ai-api-key"
                      type="password"
                      autoComplete="off"
                      value={settings.ocrAiApiKey}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          ocrAiApiKey: e.target.value,
                        }))
                      }
                      placeholder="sk-..."
                    />
                  </div>

                  <AdvancedReveal show={showAdvanced}>
                    <div className="grid gap-3">
                      <div className="flex flex-wrap gap-2 text-xs">
                        <Badge variant="outline">
                          配置来源：{getOcrConfigSourceLabel(ocrState.ocrModelsConfigSource)}
                        </Badge>
                      </div>

                      <div className="grid gap-2">
                        <FieldLabel htmlFor="ocr-ai-base-url" hint="仅在自定义接口地址时需要修改。">
                          OCR Base URL（可选）
                        </FieldLabel>
                        <Input
                          id="ocr-ai-base-url"
                          type="text"
                          autoComplete="off"
                          value={settings.ocrAiBaseUrl}
                          onChange={(e) =>
                            setSettings((s) => ({
                              ...s,
                              ocrAiBaseUrl: e.target.value,
                            }))
                          }
                          placeholder="https://api.siliconflow.cn/v1"
                        />
                      </div>

                      <div className="grid gap-2">
                        <FieldLabel
                          htmlFor="ocr-ai-chain-mode"
                          hint="模型直出适合整页识别；内置文档解析适合 PaddleOCR-VL；本地切块识别会先分块再识别。"
                        >
                          AIOCR 识别链路
                        </FieldLabel>
                        <Select
                          id="ocr-ai-chain-mode"
                          value={settings.ocrAiChainMode}
                          onChange={(e) =>
                            setSettings((s) => ({
                              ...s,
                              ocrAiChainMode: e.target.value as Settings["ocrAiChainMode"],
                              ocrAiModel: "",
                            }))
                          }
                        >
                          {ocrAiChainModeOptions.map((option) => (
                            <option key={option.id} value={option.id}>
                              {option.label}
                            </option>
                          ))}
                        </Select>
                      </div>

                      {isOcrAiChainLayoutBlock ? (
                        <div className="grid gap-2">
                          <FieldLabel
                            htmlFor="ocr-ai-layout-model"
                            hint="仅在本地切块识别时使用。"
                          >
                            版面切块模型
                          </FieldLabel>
                          <Select
                            id="ocr-ai-layout-model"
                            value={settings.ocrAiLayoutModel}
                            onChange={(e) =>
                              setSettings((s) => ({
                                ...s,
                                ocrAiLayoutModel: e.target.value as Settings["ocrAiLayoutModel"],
                              }))
                            }
                          >
                            {ocrAiLayoutModelOptions.map((option) => (
                              <option key={option.id} value={option.id}>
                                {option.label}
                              </option>
                            ))}
                          </Select>
                        </div>
                      ) : null}

                      {isOcrAiChainDocParser ? (
                        <div className="grid gap-2">
                          <FieldLabel
                            htmlFor="ocr-paddle-vl-docparser-max-side"
                            hint="仅对 PaddleOCR-VL 链路生效。0 表示不缩图。"
                          >
                            PaddleOCR-VL 长边上限
                          </FieldLabel>
                          <Input
                            id="ocr-paddle-vl-docparser-max-side"
                            type="number"
                            min={0}
                            step={100}
                            value={settings.ocrPaddleVlDocparserMaxSidePx}
                            onChange={(e) =>
                              setSettings((s) => ({
                                ...s,
                                ocrPaddleVlDocparserMaxSidePx: e.target.value,
                              }))
                            }
                            placeholder="2200"
                          />
                        </div>
                      ) : null}
                    </div>
                  </AdvancedReveal>

                  <div className="grid gap-2">
                    <FieldLabel
                      htmlFor="ocr-ai-model"
                      hint={
                        isOcrAiChainDocParser
                          ? "此链路只显示 PaddleOCR-VL 模型。"
                          : isOcrAiChainDirect
                            ? "适合整页 OCR 模型。"
                            : "适合通用视觉模型。"
                      }
                    >
                      {isOcrAiChainLayoutBlock ? "AI 视觉识别模型（必填）" : "专用 OCR 模型（必填）"}
                    </FieldLabel>
                    <Select
                      id="ocr-ai-model"
                      value={settings.ocrAiModel}
                      onChange={(e) =>
                        setSettings((s) => ({ ...s, ocrAiModel: e.target.value }))
                      }
                      disabled={ocrModelLoading}
                    >
                      <option value="">请选择 OCR 模型</option>
                      {ocrModelLoading ? (
                        <option value="__loading__" disabled>
                          正在加载模型...
                        </option>
                      ) : visibleOcrModelOptions.length ? null : (
                        <option value="__none__" disabled>
                          暂无可用 OCR 模型
                        </option>
                      )}
                      {visibleOcrModelOptions.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </Select>
                    {ocrModelError ? (
                      <div className="text-xs text-destructive">{ocrModelError}</div>
                    ) : null}
                  </div>

                  <AdvancedReveal show={showAdvanced}>
                    <div className="grid gap-3 border border-border/70 bg-muted/10 p-3">
                      <div className="flex items-center gap-2 font-sans text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                        <span>并发与限流</span>
                        <HoverHint text="仅对模型直出和本地切块识别生效。" />
                      </div>

                      <div className="grid gap-3 md:grid-cols-2">
                        <div className="grid gap-2">
                          <FieldLabel
                            htmlFor="ocr-ai-page-concurrency"
                            hint="仅对模型直出和本地切块识别生效。1 表示串行。"
                          >
                            多页并发数
                          </FieldLabel>
                          <Input
                            id="ocr-ai-page-concurrency"
                            type="number"
                            min={1}
                            max={8}
                            step={1}
                            value={settings.ocrAiPageConcurrency}
                            disabled={!isOcrAiChainDirect && !isOcrAiChainLayoutBlock}
                            onChange={(e) =>
                              setSettings((s) => ({
                                ...s,
                                ocrAiPageConcurrency: e.target.value,
                              }))
                            }
                            placeholder="1"
                          />
                        </div>

                        <div className="grid gap-2">
                          <FieldLabel
                            htmlFor="ocr-ai-block-concurrency"
                            hint="仅对本地切块识别生效。"
                          >
                            单页切块并发
                          </FieldLabel>
                          <Input
                            id="ocr-ai-block-concurrency"
                            type="number"
                            min={1}
                            max={8}
                            step={1}
                            value={settings.ocrAiBlockConcurrency}
                            disabled={!isOcrAiChainLayoutBlock}
                            onChange={(e) =>
                              setSettings((s) => ({
                                ...s,
                                ocrAiBlockConcurrency: e.target.value,
                              }))
                            }
                            placeholder="自动"
                          />
                        </div>
                      </div>

                      <div className="grid gap-3 md:grid-cols-3">
                        <div className="grid gap-2">
                          <FieldLabel
                            htmlFor="ocr-ai-rpm"
                            hint="留空表示不主动限流。"
                          >
                            RPM 上限
                          </FieldLabel>
                          <Input
                            id="ocr-ai-rpm"
                            type="number"
                            min={1}
                            step={1}
                            value={settings.ocrAiRequestsPerMinute}
                            disabled={isOcrAiChainDocParser}
                            onChange={(e) =>
                              setSettings((s) => ({
                                ...s,
                                ocrAiRequestsPerMinute: e.target.value,
                              }))
                            }
                            placeholder="不限"
                          />
                        </div>

                        <div className="grid gap-2">
                          <FieldLabel
                            htmlFor="ocr-ai-tpm"
                            hint="不同模型的消耗会有差异。"
                          >
                            TPM 上限
                          </FieldLabel>
                          <Input
                            id="ocr-ai-tpm"
                            type="number"
                            min={1}
                            step={1000}
                            value={settings.ocrAiTokensPerMinute}
                            disabled={isOcrAiChainDocParser}
                            onChange={(e) =>
                              setSettings((s) => ({
                                ...s,
                                ocrAiTokensPerMinute: e.target.value,
                              }))
                            }
                            placeholder="不限"
                          />
                        </div>

                        <div className="grid gap-2">
                          <FieldLabel
                            htmlFor="ocr-ai-max-retries"
                            hint="仅对 AIOCR 请求生效。"
                          >
                            失败重试次数
                          </FieldLabel>
                          <Input
                            id="ocr-ai-max-retries"
                            type="number"
                            min={0}
                            max={8}
                            step={1}
                            value={settings.ocrAiMaxRetries}
                            disabled={isOcrAiChainDocParser}
                            onChange={(e) =>
                              setSettings((s) => ({
                                ...s,
                                ocrAiMaxRetries: e.target.value,
                              }))
                            }
                            placeholder="0"
                          />
                        </div>
                      </div>

                      {isOcrAiChainDocParser ? (
                        <div className="text-xs text-muted-foreground">
                          当前是内置文档解析链路，并发与限流不会生效。
                        </div>
                      ) : null}
                    </div>
                  </AdvancedReveal>

                  <AdvancedReveal show={showAdvanced}>
                    <div className="grid gap-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => void onCheckAiOcrModel()}
                          disabled={aiOcrChecking}
                        >
                          {aiOcrChecking ? "检测中..." : "检测 OCR 配置"}
                        </Button>
                        <HoverHint
                          text={
                            isOcrAiChainLayoutBlock
                              ? "检查当前分块链路和视觉模型是否可用。"
                              : "检查当前模型是否能返回可用识别结果。"
                          }
                        />
                      </div>
                      {aiOcrCheckError ? (
                        <div className="text-xs text-destructive">{aiOcrCheckError}</div>
                      ) : null}
                      {aiOcrCheck ? (
                        <div
                          className={
                            aiOcrCheck.ok
                              ? "border border-emerald-500/40 bg-emerald-50 px-3 py-2 text-xs text-emerald-900"
                              : "border border-amber-500/40 bg-amber-50 px-3 py-2 text-xs text-amber-900"
                          }
                        >
                          <div>
                            状态：{aiOcrCheck.check.ready ? "通过" : "未通过"} ·
                            耗时：{aiOcrCheck.check.elapsed_ms}ms
                          </div>
                          <div>模型：{aiOcrCheck.check.model}</div>
                          <div>
                            结果：{aiOcrCheck.check.valid_bbox_items}/{aiOcrCheck.check.items_count} 条有效结果
                          </div>
                          <div>{aiOcrCheck.check.message}</div>
                          {aiOcrCheck.check.error ? <div>错误：{aiOcrCheck.check.error}</div> : null}
                        </div>
                      ) : null}
                    </div>
                  </AdvancedReveal>

                  {needsRequiredOcrAiConfig &&
                  !settings.ocrAiApiKey.trim() &&
                  Boolean(mainModelsApiKeyRaw.trim()) ? (
                    <div className="text-muted-foreground text-xs">
                      请单独填写 OCR API Key 与 OCR 模型。
                    </div>
                  ) : null}
                </div>
              ) : null}

              {shouldShowBaiduConfig && (showAdvanced || isOcrProviderBaidu) ? (
                <>
                  {isBaiduDocParseMode ? (
                    <div className="grid gap-2">
                      <FieldLabel
                        htmlFor="baidu-doc-parse-type"
                        hint="普通模式适合常规文档，PaddleOCR-VL 更适合复杂版式。"
                      >
                        文档解析类型
                      </FieldLabel>
                      <Select
                        id="baidu-doc-parse-type"
                        value={settings.baiduDocParseType}
                        onChange={(e) =>
                          setSettings((s) => ({
                            ...s,
                            baiduDocParseType: e.target.value as BaiduDocParseType,
                          }))
                        }
                      >
                        {baiduDocParseTypeOptions.map((option) => (
                          <option key={option.id} value={option.id}>
                            {option.label}
                          </option>
                        ))}
                      </Select>
                    </div>
                  ) : null}
                  <div className="grid gap-2">
                    <FieldLabel htmlFor="ocr-baidu-api-key">
                      {isBaiduDocParseMode ? "百度解析 API Key" : "百度 OCR API Key"}
                    </FieldLabel>
                    <Input
                      id="ocr-baidu-api-key"
                      type="password"
                      autoComplete="off"
                      value={settings.ocrBaiduApiKey}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          ocrBaiduApiKey: e.target.value,
                        }))
                      }
                      placeholder="..."
                    />
                  </div>
                  <div className="grid gap-2">
                    <FieldLabel htmlFor="ocr-baidu-secret-key">
                      {isBaiduDocParseMode ? "百度解析 Secret Key" : "百度 OCR Secret Key"}
                    </FieldLabel>
                    <Input
                      id="ocr-baidu-secret-key"
                      type="password"
                      autoComplete="off"
                      value={settings.ocrBaiduSecretKey}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          ocrBaiduSecretKey: e.target.value,
                        }))
                      }
                      placeholder="..."
                    />
                  </div>
                  <AdvancedReveal show={showAdvanced}>
                    <div className="grid gap-2">
                      <FieldLabel htmlFor="ocr-baidu-app-id" hint="兼容字段，可留空。">
                        {isBaiduDocParseMode ? "百度解析 App ID（可选）" : "百度 OCR App ID（可选）"}
                      </FieldLabel>
                      <Input
                        id="ocr-baidu-app-id"
                        type="text"
                        autoComplete="off"
                        value={settings.ocrBaiduAppId}
                        onChange={(e) =>
                          setSettings((s) => ({
                            ...s,
                            ocrBaiduAppId: e.target.value,
                          }))
                        }
                        placeholder="..."
                      />
                    </div>
                  </AdvancedReveal>
                </>
              ) : null}

              {shouldShowTesseractConfig && (showAdvanced || isOcrProviderTesseract) ? (
                <div className="grid gap-2">
                  <FieldLabel htmlFor="ocr-tesseract-min-conf">
                    Tesseract 最低置信度（0-100）
                  </FieldLabel>
                  <Input
                    id="ocr-tesseract-min-conf"
                    type="number"
                    min={0}
                    max={100}
                    value={settings.ocrTesseractMinConfidence}
                    onChange={(e) =>
                      setSettings((s) => ({
                        ...s,
                        ocrTesseractMinConfidence: e.target.value,
                      }))
                    }
                    placeholder="50"
                  />
                </div>
              ) : null}

              {shouldShowTesseractConfig && (showAdvanced || isOcrProviderTesseract) ? (
                <div className="grid gap-2">
                  <FieldLabel htmlFor="ocr-tesseract-lang">
                    Tesseract 语言（例如 eng、chi_sim）
                  </FieldLabel>
                  <Input
                    id="ocr-tesseract-lang"
                    type="text"
                    autoComplete="off"
                    value={settings.ocrTesseractLanguage}
                    onChange={(e) =>
                      setSettings((s) => ({
                        ...s,
                        ocrTesseractLanguage: e.target.value,
                      }))
                    }
                    placeholder="chi_sim+eng"
                  />
                </div>
              ) : null}

              {shouldShowLocalOcrCheck ? (
                <AdvancedReveal show={showAdvanced}>
                  <div className="grid gap-3 border border-border bg-muted/20 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2 font-sans text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                        <span>本地 OCR 综合检测</span>
                        <HoverHint text="检查本地运行环境与模型文件，不会触发自动下载。" />
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={onRunLocalOcrSuite}
                          disabled={localOcrSuiteChecking}
                        >
                          {localOcrSuiteChecking ? "检测中..." : "检测本地 OCR"}
                        </Button>
                      </div>
                    </div>
                    {localOcrSuiteError ? (
                      <div className="text-xs text-destructive">{localOcrSuiteError}</div>
                    ) : null}

                    <div className="grid gap-2 md:grid-cols-2">
                    <div
                      className={
                        !hasTesseractSuite
                          ? "border border-border bg-background px-3 py-2 text-xs text-muted-foreground"
                          : tesseractSuiteReady
                            ? "border border-emerald-500/40 bg-emerald-50 px-3 py-2 text-xs text-emerald-900"
                            : "border border-amber-500/40 bg-amber-50 px-3 py-2 text-xs text-amber-900"
                      }
                    >
                      <div className="font-medium">Tesseract</div>
                      <div>
                        运行环境：
                        {tesseractSuite?.runtime
                          ? tesseractSuite.runtime.check.ready
                            ? "就绪"
                            : "未就绪"
                          : "未检测"}
                        {tesseractSuite?.runtime
                          ? ` · ${tesseractSuite.runtime.check.message}`
                          : ""}
                      </div>
                      {tesseractSuite?.runtimeError ? (
                        <div>运行环境错误：{tesseractSuite.runtimeError}</div>
                      ) : null}
                      <div>
                        模型文件：
                        {tesseractSuite?.models
                          ? tesseractSuite.models.check.ready
                            ? "齐全"
                            : "缺失"
                          : "未检测"}
                        {tesseractSuite?.models
                          ? ` · ${tesseractSuite.models.check.message}`
                          : ""}
                      </div>
                      {tesseractSuite?.models?.check.missing_models?.length ? (
                        <div>
                          缺失：{tesseractSuite.models.check.missing_models.join(", ")}
                        </div>
                      ) : null}
                      {tesseractSuite?.modelsError ? (
                        <div>模型错误：{tesseractSuite.modelsError}</div>
                      ) : null}
                    </div>

                    <div
                      className={
                        !hasPaddleSuite
                          ? "border border-border bg-background px-3 py-2 text-xs text-muted-foreground"
                          : paddleSuiteReady
                            ? "border border-emerald-500/40 bg-emerald-50 px-3 py-2 text-xs text-emerald-900"
                            : "border border-amber-500/40 bg-amber-50 px-3 py-2 text-xs text-amber-900"
                      }
                    >
                      <div className="font-medium">PaddleOCR</div>
                      <div>
                        运行环境：
                        {paddleSuite?.runtime
                          ? paddleSuite.runtime.check.ready
                            ? "就绪"
                            : "未就绪"
                          : "未检测"}
                        {paddleSuite?.runtime ? ` · ${paddleSuite.runtime.check.message}` : ""}
                      </div>
                      {paddleSuite?.runtimeError ? (
                        <div>运行环境错误：{paddleSuite.runtimeError}</div>
                      ) : null}
                      <div>
                        模型文件：
                        {paddleSuite?.models
                          ? paddleSuite.models.check.ready
                            ? "齐全"
                            : "缺失"
                          : "未检测"}
                        {paddleSuite?.models ? ` · ${paddleSuite.models.check.message}` : ""}
                      </div>
                      {paddleSuite?.models?.check.missing_models?.length ? (
                        <div>缺失：{paddleSuite.models.check.missing_models.join(", ")}</div>
                      ) : null}
                      {paddleSuite?.models?.check.model_root_dir ? (
                        <div>目录：{paddleSuite.models.check.model_root_dir}</div>
                      ) : null}
                      {paddleSuite?.modelsError ? (
                        <div>模型错误：{paddleSuite.modelsError}</div>
                      ) : null}
                    </div>
                    </div>
                  </div>
                </AdvancedReveal>
              ) : null}
              </section>
            ) : null}

          </CardContent>

            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
