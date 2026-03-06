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
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/select"

import {
  type LayoutAssistMode,
  SILICONFLOW_BASE_URL,
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
  resolveOcrSettingsState,
  type ParseEngineMode,
} from "@/lib/run-config"
import {
  apiFetch,
  clearStoredApiOrigin,
  getStoredApiOrigin,
  normalizeFetchError,
  resolveApiOrigin,
  setStoredApiOrigin,
} from "@/lib/api"

type SettingsSectionId = "api" | "strategy" | "ocr" | "vision"

const parseEngineOptions: Array<{ id: ParseEngineMode; label: string }> = [
  { id: "local_ocr", label: "本地 OCR" },
  { id: "remote_ocr", label: "远程 OCR" },
  { id: "mineru_cloud", label: "云端 MinerU" },
]

const parseEngineModeLabels: Record<ParseEngineMode, string> = {
  local_ocr: "本地 OCR",
  remote_ocr: "远程 OCR",
  mineru_cloud: "云端 MinerU",
}

const settingsSectionItems: Array<{
  id: SettingsSectionId
  label: string
  description: string
}> = [
  { id: "api", label: "接口配置", description: "密钥、模型与接口地址" },
  { id: "strategy", label: "处理策略", description: "输出方式与版式选项" },
  { id: "ocr", label: "OCR 配置", description: "OCR 来源与能力检测" },
  { id: "vision", label: "版式辅助", description: "页面结构修正与图片区建议" },
]

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

const aiProviderOptions: Array<{ id: "openai" | "siliconflow" | "claude"; label: string }> = [
  { id: "openai", label: "OpenAI 兼容" },
  { id: "siliconflow", label: "SiliconFlow" },
  { id: "claude", label: "Claude" },
]

const ocrAiProviderOptions: Array<{ id: OcrAiProvider; label: string }> = [
  { id: "auto", label: "自动识别（推荐）" },
  { id: "openai", label: "OpenAI" },
  { id: "siliconflow", label: "SiliconFlow" },
  { id: "deepseek", label: "DeepSeek" },
  { id: "ppio", label: "PPIO" },
  { id: "novita", label: "Novita" },
]

const ocrProviderLabels: Record<Settings["ocrProvider"], string> = {
  auto: "自动（混合）",
  aiocr: "AI OCR（OpenAI 兼容）",
  paddle_local: "PaddleOCR（本地）",
  baidu: "百度 OCR",
  tesseract: "Tesseract（本地）",
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

function isOpenAiCompatibleEndpoint(baseUrl: string): boolean {
  const normalized = baseUrl.trim().toLowerCase()
  if (!normalized) return false
  return !normalized.includes("api.openai.com")
}

function isOcrSpecializedModel(modelId: string): boolean {
  const normalized = modelId.trim().toLowerCase()
  if (!normalized) return false
  return normalized.includes("ocr") || normalized.includes("paddleocr") || normalized.includes("mineru")
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
  const [modelOptions, setModelOptions] = React.useState<string[]>([])
  const [modelLoading, setModelLoading] = React.useState(false)
  const [modelError, setModelError] = React.useState<string | null>(null)
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
  const isOcrEnabledForCurrentEngine = ocrState.isOcrEnabledForCurrentEngine
  const canUseAiOcr = ocrState.canUseAiOcr
  const selectedOcrProvider = ocrState.selectedOcrProvider
  const parseEngineMode = ocrState.parseEngineMode
  const currentLayoutAssistMode: LayoutAssistMode = ocrState.currentLayoutAssistMode
  const isLayoutAssistEnabledForCurrentEngine = ocrState.isLayoutAssistEnabledForCurrentEngine
  const currentOcrLinebreakAssistMode = ocrState.currentOcrLinebreakAssistMode
  const canConfigureOcrLinebreakAssist = ocrState.canConfigureOcrLinebreakAssist
  const isOcrProviderAi = ocrState.isOcrProviderAi
  const isOcrProviderPaddleLocal = ocrState.isOcrProviderPaddleLocal
  const isOcrProviderBaidu = ocrState.isOcrProviderBaidu
  const isOcrProviderTesseract = ocrState.isOcrProviderTesseract
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
  const shouldShowBaiduConfig = ocrState.shouldShowBaiduConfig
  const shouldShowTesseractConfig = ocrState.shouldShowTesseractConfig
  const shouldShowAiVendorAdapter = ocrState.shouldShowAiVendorAdapter
  const mainConfig = getMainProviderConfig(settings)
  const aiProvider =
    settings.provider === "mineru" ? settings.preferredMainProvider : settings.provider
  const mainModelsApiKeyRaw = mainConfig.apiKey
  const mainModelsBaseUrlRaw =
    aiProvider === "siliconflow"
      ? mainConfig.baseUrl || SILICONFLOW_BASE_URL
      : mainConfig.baseUrl
  const mainModelsSelectedRaw = mainConfig.model
  const modelsApiKey = mainModelsApiKeyRaw
  const modelsBaseUrl = mainModelsBaseUrlRaw
  const modelsSelected = mainModelsSelectedRaw
  const isCompatGatewayMode =
    !isMineruProvider &&
    aiProvider === "openai" &&
    isOpenAiCompatibleEndpoint(modelsBaseUrl)
  const localOcrAiPostprocessRequested =
    !isMineruProvider &&
    isOcrEnabledForCurrentEngine &&
    !isOcrProviderAi &&
    currentOcrLinebreakAssistMode === "on"
  const shouldShowVisionModelConfig =
    isLayoutAssistEnabledForCurrentEngine || localOcrAiPostprocessRequested
  const layoutAssistRequested =
    settings.visualAssistModeLocal !== "off" ||
    settings.visualAssistModeRemote !== "off" ||
    settings.visualAssistModeMineru !== "off"
  const canLoadModels =
    aiProvider !== "claude" &&
    Boolean(modelsApiKey) &&
    (layoutAssistRequested || localOcrAiPostprocessRequested)
  const ocrModelsApiKey = ocrState.ocrModelsApiKey
  const ocrModelsBaseUrl = ocrState.ocrModelsBaseUrl

  React.useEffect(() => {
    if (!canLoadModels) {
      setModelOptions([])
      setModelError(null)
      return
    }

    let mounted = true
    const controller = new AbortController()
    const timer = setTimeout(async () => {
      setModelLoading(true)
      setModelError(null)

      try {
        const payload: Record<string, string> = {
          provider: "openai",
          api_key: modelsApiKey,
          capability: "vision",
        }
        if (modelsBaseUrl) {
          payload.base_url = modelsBaseUrl
        }

        const response = await apiFetch("/models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        })

        if (!response.ok) {
          const body = await response.json().catch(() => null)
          throw new Error(body?.message || "模型列表加载失败")
        }

        const body = await response.json().catch(() => null)
        const models = Array.isArray(body?.models)
          ? body.models.filter((m: unknown) => typeof m === "string")
          : []
        const filteredModels = models.filter((model: string) => !isOcrSpecializedModel(model))

        if (!mounted) return
        setModelOptions(filteredModels)
        if (filteredModels.length && !filteredModels.includes(modelsSelected)) {
          setSettings((prev) =>
            aiProvider === "siliconflow"
              ? { ...prev, siliconflowModel: "" }
              : { ...prev, openaiModel: "" }
          )
        }
      } catch (e) {
        if (!mounted || controller.signal.aborted) return
        setModelError(normalizeFetchError(e, "模型列表加载失败"))
        setModelOptions([])
      } finally {
        if (mounted) setModelLoading(false)
      }
    }, 400)

    return () => {
      mounted = false
      controller.abort()
      clearTimeout(timer)
    }
  }, [
    canLoadModels,
    modelsApiKey,
    modelsBaseUrl,
    modelsSelected,
    aiProvider,
    isMineruProvider,
    layoutAssistRequested,
  ])

  const canLoadOcrModels =
    canUseAiOcr &&
    isOcrEnabledForCurrentEngine &&
    needsRequiredOcrAiConfig &&
    Boolean(ocrModelsApiKey)

  const visibleSectionItems = React.useMemo(
    () =>
      settingsSectionItems.filter(
        (section) =>
          (section.id !== "ocr" || isOcrEnabledForCurrentEngine) &&
          (section.id !== "vision" || showAdvanced)
      ),
    [isOcrEnabledForCurrentEngine, showAdvanced]
  )
  const observableSectionItems = React.useMemo(
    () =>
      visibleSectionItems.filter(
        (section) => section.id !== "vision" || showAdvanced
      ),
    [showAdvanced, visibleSectionItems]
  )
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
      if (sectionId === "vision" && !showAdvanced) {
        setShowAdvanced(true)
      }

      setActiveSection(sectionId)
      const scrollToTarget = () => {
        const target = document.getElementById(`settings-section-${sectionId}`)
        if (!target) return
        target.scrollIntoView({ behavior: "smooth", block: "start" })
      }
      if (sectionId === "vision" && !showAdvanced) {
        window.setTimeout(scrollToTarget, 180)
      } else {
        scrollToTarget()
      }
    },
    [showAdvanced]
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
          capability: "ocr",
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
        if (models.length && !models.includes(settings.ocrAiModel)) {
          setSettings((prev) => ({ ...prev, ocrAiModel: "" }))
        }
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
    settings.ocrAiModel,
    selectedOcrProvider,
    isOcrEnabledForCurrentEngine,
  ])

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
    toast.success("图片阈值已恢复为后端默认值")
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
    const provider =
      ocrState.ocrModelsConfigSource === "main"
        ? (mainConfig.ocrAdapter ?? "auto")
        : (settings.ocrAiProvider || "auto").trim() || "auto"
    const sourceLabel = getOcrConfigSourceLabel(ocrState.ocrModelsConfigSource)

    if (!apiKey) {
      const message =
        needsRequiredOcrAiConfig
          ? "显式 AI OCR 需要独立 OCR API Key"
          : `请先补充可用的 OCR AI 配置（当前来源：${sourceLabel}）`
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
      const payload: Record<string, string> = {
        provider,
        api_key: apiKey,
        model,
      }
      if (baseUrl) payload.base_url = baseUrl

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
    mainConfig.ocrAdapter,
    needsRequiredOcrAiConfig,
    ocrModelsApiKey,
    ocrModelsBaseUrl,
    ocrState.ocrModelsConfigSource,
    settings.ocrAiModel,
    settings.ocrAiProvider,
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
                  在这里管理模型、OCR 和处理策略。设置会保存在当前浏览器，方便下次继续使用。
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
                  <Badge variant="secondary">{parseEngineModeLabels[parseEngineMode]}</Badge>
                </div>

                <div className="grid grid-cols-1 gap-2 sm:grid-cols-4">
                  {parseEngineOptions.map((p) => (
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

                <div className="flex items-center justify-between rounded-md border border-border bg-muted/20 px-3 py-2">
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      className="h-4 w-4 accent-[#111111]"
                      checked={showAdvanced}
                      onChange={(e) => setShowAdvanced(e.target.checked)}
                    />
                    显示高级设置
                  </label>
                  <div className="text-muted-foreground text-xs">
                    {showAdvanced ? "已展开" : "已折叠"}
                  </div>
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
              <div className="font-sans text-sm font-semibold uppercase tracking-[0.14em]">
                接口配置
              </div>

              <div className="grid gap-2 border border-border bg-muted/30 p-3">
                <label className="text-muted-foreground text-xs">后端 API 地址（当前生效）</label>
                <div className="flex flex-wrap items-center gap-2">
                  <div className="font-mono break-all text-xs">{apiOrigin}</div>
                  <Badge variant="outline">
                    {apiOriginOverrideEnabled ? "手动覆盖" : "自动探测"}
                  </Badge>
                </div>
                <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto]">
                  <Input
                    type="text"
                    autoComplete="off"
                    value={apiOriginInput}
                    onChange={(e) => setApiOriginInput(e.target.value)}
                    placeholder="留空=自动探测，或输入 http://127.0.0.1:8001"
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
                <div className="text-muted-foreground text-xs">
                  自动模式在公开部署优先使用同源地址；本地开发会探测 `:8000` 与 `:8001`。如需固定地址请设置 `NEXT_PUBLIC_API_URL`。
                </div>
                {apiOriginError ? (
                  <div className="text-xs text-destructive">{apiOriginError}</div>
                ) : null}
              </div>

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
                </>
              ) : (
                <>
                  <div className="text-muted-foreground text-sm">
                    当前模式将使用下方 OCR 配置中的参数。
                  </div>
                </>
              )}
            </section>

            <section
              id="settings-section-strategy"
              className="scroll-mt-24 grid gap-4 border-b border-border p-5"
            >
              <div className="font-sans text-sm font-semibold uppercase tracking-[0.14em]">
                处理策略
              </div>

              {isMineruProvider ? (
                <>
                  <AdvancedReveal show={showAdvanced}>
                    <div className="grid gap-2 pt-1">
                      <label
                        className="text-muted-foreground text-xs"
                        htmlFor="text-erase-mode"
                      >
                        文字消除模式
                      </label>
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
                        <option value="smart">智能消除（实验）</option>
                      </Select>
                      <div className="text-muted-foreground text-xs">
                        默认推荐纯色填充，输出更稳定；智能消除保留为实验模式。
                      </div>
                    </div>
                  </AdvancedReveal>

                  <div className="text-muted-foreground text-sm">
                    MinerU 现在只使用自身解析与 OCR，不再叠加本地/远程 OCR 定位。
                  </div>
                </>
              ) : (
                <>
                  <div className="text-muted-foreground text-sm">
                    本地/远程 OCR 模式默认启用 OCR。
                  </div>
                </>
              )}

              <AdvancedReveal show={!isMineruProvider && showAdvanced}>
                <div className="grid gap-2 pt-1">
                  <label
                    className="text-muted-foreground text-xs"
                    htmlFor="text-erase-mode"
                  >
                    文字消除模式
                  </label>
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
                    <option value="smart">智能消除（实验）</option>
                  </Select>
                  <div className="text-muted-foreground text-xs">
                    默认推荐纯色填充，输出更稳定；智能消除保留为实验模式。
                  </div>
                </div>
              </AdvancedReveal>

              <div className="grid gap-2">
                <label className="text-muted-foreground text-xs" htmlFor="scanned-page-mode">
                  扫描页合成模式
                </label>
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
                  <option value="segmented">分块（图片可编辑）</option>
                  <option value="fullpage">全页（更像原图）</option>
                </Select>
                <div className="text-muted-foreground text-xs">
                  分块模式会尝试把截图/图表等区域单独裁剪为可编辑图片；全页模式只保留一张整页背景并覆盖文字，通常更接近原图但图片不可单独编辑。
                </div>
              </div>

              <AdvancedReveal show={showAdvanced}>
                <div className="grid gap-3 rounded-md border border-border/70 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-sans text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                      图片底图清除与图块阈值
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
                      <label className="text-muted-foreground text-xs" htmlFor="image-bg-clear-min-pt">
                        清除扩边最小值（pt）
                      </label>
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
                      <div className="text-muted-foreground text-[11px]">
                        后端默认：{defaultSettings.imageBgClearExpandMinPt}
                      </div>
                    </div>

                    <div className="grid gap-1.5">
                      <label className="text-muted-foreground text-xs" htmlFor="image-bg-clear-max-pt">
                        清除扩边最大值（pt）
                      </label>
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
                      <div className="text-muted-foreground text-[11px]">
                        后端默认：{defaultSettings.imageBgClearExpandMaxPt}
                      </div>
                    </div>

                    <div className="grid gap-1.5">
                      <label className="text-muted-foreground text-xs" htmlFor="image-bg-clear-ratio">
                        清除扩边比例
                      </label>
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
                      <div className="text-muted-foreground text-[11px]">
                        后端默认：{defaultSettings.imageBgClearExpandRatio}
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="grid gap-1.5">
                      <label className="text-muted-foreground text-xs" htmlFor="scanned-min-area-ratio">
                        图块最小面积比例
                      </label>
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
                      <div className="text-muted-foreground text-[11px]">
                        后端默认：{defaultSettings.scannedImageRegionMinAreaRatio}
                      </div>
                    </div>

                    <div className="grid gap-1.5">
                      <label className="text-muted-foreground text-xs" htmlFor="scanned-max-area-ratio">
                        图块最大面积比例
                      </label>
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
                      <div className="text-muted-foreground text-[11px]">
                        后端默认：{defaultSettings.scannedImageRegionMaxAreaRatio}
                      </div>
                    </div>

                    <div className="grid gap-1.5">
                      <label className="text-muted-foreground text-xs" htmlFor="scanned-max-aspect-ratio">
                        图块最大长宽比
                      </label>
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
                      <div className="text-muted-foreground text-[11px]">
                        后端默认：{defaultSettings.scannedImageRegionMaxAspectRatio}
                      </div>
                    </div>
                  </div>

                  <div className="text-muted-foreground text-xs">
                    建议先使用默认值；仅在出现图片底图残留或图块误判时微调。修改后建议用 1 页样本先验证。
                  </div>
                </div>
              </AdvancedReveal>
            </section>

            {isOcrEnabledForCurrentEngine ? (
              <section
                id="settings-section-ocr"
                className="scroll-mt-24 grid gap-4 p-5"
              >
              <div className="font-sans text-sm font-semibold uppercase tracking-[0.14em]">
                OCR 配置
              </div>

              <div className="grid gap-2">
                <label className="text-muted-foreground text-xs" htmlFor="ocr-provider">
                  OCR 提供方
                </label>
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
                  <div className="text-muted-foreground text-xs">
                    当前模式：{ocrProviderLabels[selectedOcrProvider]}。
                    {isOcrProviderAi
                      ? " 显式 AI OCR 只使用专用 OCR 模型本身的识别与 bbox。"
                      : isOcrProviderPaddleLocal
                        ? " 使用本地 PaddleOCR；如需 AI 修字/拆行，可在下方开启 AI OCR 后处理。"
                        : isOcrProviderBaidu
                          ? " 使用百度 OCR；如需 AI 修字/拆行，可在下方开启 AI OCR 后处理。"
                          : " 使用本地 Tesseract；如需 AI 修字/拆行，可在下方开启 AI OCR 后处理。"}
                  </div>
              </div>

              <AdvancedReveal show={showAdvanced}>
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
                    OCR 严格模式（默认开启；开启后禁止隐式 OCR provider 回退，初始化/运行失败即报错）
                  </label>
                  <div className="text-muted-foreground text-xs">
                    默认开启。关闭后才启用最佳努力策略：允许 provider 降级，或在 OCR 失败时按图片页继续。
                  </div>
                </>
              </AdvancedReveal>

              {shouldShowAiVendorAdapter ? (
                <div className="grid gap-3">
                  <div className="grid gap-2">
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="ocr-ai-provider"
                    >
                      AI OCR 厂商适配
                    </label>
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
                    <div className="text-muted-foreground text-xs">
                      厂商适配会自动处理常见参数差异。
                    </div>

                  </div>
                </div>
              ) : null}

              {needsRequiredOcrAiConfig ? (
                <div className="grid gap-3 border border-border bg-muted/20 p-3">
                  <div className="font-sans text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                    专用 OCR 接口参数
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <Badge variant="outline">
                      配置来源：{getOcrConfigSourceLabel(ocrState.ocrModelsConfigSource)}
                    </Badge>
                  </div>

                  <div className="grid gap-2">
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="ocr-ai-api-key"
                    >
                      OCR API Key
                    </label>
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

                  <div className="grid gap-2">
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="ocr-ai-base-url"
                    >
                      OCR Base URL（可选）
                    </label>
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
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="ocr-ai-model"
                    >
                      专用 OCR 模型（必填）
                    </label>
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
                      ) : ocrModelOptions.length ? null : (
                        <option value="__none__" disabled>
                          暂无可用 OCR 模型
                        </option>
                      )}
                      {ocrModelOptions.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </Select>
                    {ocrModelError ? (
                      <div className="text-xs text-destructive">{ocrModelError}</div>
                    ) : null}
                  </div>

                  <div className="text-muted-foreground text-xs">
                    显式 AI OCR 现在固定走纯 AI OCR 链路，不再混用本地 Tesseract 几何定位；这里仅列专门的 OCR 模型。
                  </div>

                  <div className="grid gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => void onCheckAiOcrModel()}
                        disabled={aiOcrChecking}
                      >
                        {aiOcrChecking ? "验证中..." : "验证 OCR 能力"}
                      </Button>
                      <span className="text-muted-foreground text-xs">
                        检查模型是否返回有效 bbox
                      </span>
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
                          结果：{aiOcrCheck.check.valid_bbox_items}/{aiOcrCheck.check.items_count} 条有效
                          bbox
                        </div>
                        <div>{aiOcrCheck.check.message}</div>
                        {aiOcrCheck.check.error ? <div>错误：{aiOcrCheck.check.error}</div> : null}
                      </div>
                    ) : null}
                  </div>

                  {needsRequiredOcrAiConfig &&
                  !settings.ocrAiApiKey.trim() &&
                  Boolean(mainModelsApiKeyRaw.trim()) ? (
                    <div className="text-muted-foreground text-xs">
                      显式 AI OCR 不再复用主 AI Key。请填写独立 OCR API Key 与 OCR 模型后再执行。
                    </div>
                  ) : null}

                </div>
              ) : null}

              {!showAdvanced && canConfigureOcrLinebreakAssist ? (
                <div className="text-muted-foreground text-xs">
                  AI OCR 后处理属于高级功能，请展开“高级设置”后配置。启用后，非显式 AI OCR 会复用下方视觉模型做文字修正与逐行拆分。
                </div>
              ) : null}

              <AdvancedReveal show={showAdvanced}>
                {canConfigureOcrLinebreakAssist ? (
                  <div className="grid gap-2">
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="ocr-ai-linebreak-mode"
                    >
                      AI OCR 后处理
                    </label>
                    <Select
                      id="ocr-ai-linebreak-mode"
                      value={currentOcrLinebreakAssistMode}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          ocrAiLinebreakAssistMode: e.target.value as Settings["ocrAiLinebreakAssistMode"],
                        }))
                      }
                    >
                      <option value="auto">自动（显式 AI OCR 推荐）</option>
                      <option value="on">开启（本地 OCR 也可用）</option>
                      <option value="off">关闭</option>
                    </Select>
                    <div className="text-muted-foreground text-xs">
                      这是 OCR 后处理，不属于版式辅助。显式 AI OCR：自动模式会按 OCR 模型特性决定是否拆行。非显式 AI OCR：开启后会尽量复用下方视觉模型做文字修正与逐行拆分；若没有可用视觉模型，则退回启发式处理。
                    </div>
                  </div>
                ) : null}
              </AdvancedReveal>

              {!canUseAiOcr ? null : !needsRequiredOcrAiConfig ? (
                <div className="text-muted-foreground text-xs">
                  当前链路不需要单独的专用 OCR 模型；如需给本地 OCR 增加 AI 修字/拆行，请开启上面的 AI OCR 后处理，并在下方配置视觉模型。
                </div>
              ) : null}

              {shouldShowBaiduConfig && (showAdvanced || isOcrProviderBaidu) ? (
                <>
                  <div className="grid gap-2">
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="ocr-baidu-app-id"
                    >
                      百度 OCR App ID
                    </label>
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
                  <div className="grid gap-2">
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="ocr-baidu-api-key"
                    >
                      百度 OCR API Key
                    </label>
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
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="ocr-baidu-secret-key"
                    >
                      百度 OCR Secret Key
                    </label>
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
                </>
              ) : null}

              {shouldShowTesseractConfig && (showAdvanced || isOcrProviderTesseract) ? (
                <div className="grid gap-2">
                  <label
                    className="text-muted-foreground text-xs"
                    htmlFor="ocr-tesseract-min-conf"
                  >
                    Tesseract 最低置信度（0-100）
                  </label>
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
                  <label
                    className="text-muted-foreground text-xs"
                    htmlFor="ocr-tesseract-lang"
                  >
                    Tesseract 语言（例如 eng、chi_sim）
                  </label>
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
                <div className="grid gap-3 border border-border bg-muted/20 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="font-sans text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                      本地 OCR 综合检测
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={onRunLocalOcrSuite}
                        disabled={localOcrSuiteChecking}
                      >
                        {localOcrSuiteChecking ? "检测中..." : "检测本地 OCR（Tesseract + PaddleOCR）"}
                      </Button>
                    </div>
                  </div>
                  <div className="text-muted-foreground text-xs">
                    一次检测同时验证运行环境与模型文件，不会触发自动下载。
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
              ) : null}
              </section>
            ) : null}

            <AdvancedReveal show={showAdvanced}>
              <section
                id="settings-section-vision"
                className="scroll-mt-24 grid gap-4 border-t border-border p-5"
              >
                <div className="font-sans text-sm font-semibold uppercase tracking-[0.14em]">
                  AI 视觉辅助
                </div>

                <div className="rounded-md border border-border bg-muted/20 p-3 text-xs text-muted-foreground">
                  当前链路：{parseEngineModeLabels[parseEngineMode]}。这里配置页面结构分析使用的视觉模型；当本地 OCR 开启“AI OCR 后处理”时，也会复用这里的模型做文字修正与拆行。
                </div>

                <div className="grid gap-2">
                  <label className="text-muted-foreground text-xs" htmlFor="vision-mode-current">
                    版式辅助模式
                  </label>
                  <Select
                    id="vision-mode-current"
                    value={currentLayoutAssistMode}
                    onChange={(e) => {
                      const nextMode = e.target.value as LayoutAssistMode
                      setSettings((s) =>
                        parseEngineMode === "local_ocr"
                          ? { ...s, visualAssistModeLocal: nextMode }
                          : parseEngineMode === "remote_ocr"
                            ? { ...s, visualAssistModeRemote: nextMode }
                            : { ...s, visualAssistModeMineru: nextMode }
                      )
                    }}
                  >
                    <option value="off">关闭（默认）</option>
                    <option value="on">强制开启</option>
                    <option value="auto">自动（按模型能力）</option>
                  </Select>
                  <div className="text-muted-foreground text-xs">
                    关闭：禁用当前链路的 AI 版式辅助。强制开启：始终启用版式辅助。自动：由后端按模型能力决定。它不会改变 OCR 提供方，也不会替你切换成另一条 OCR 链路。
                  </div>
                </div>

                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-[#111111]"
                    checked={settings.layoutAssistApplyImageRegions}
                    disabled={!isLayoutAssistEnabledForCurrentEngine}
                    onChange={(e) =>
                      setSettings((s) => ({
                        ...s,
                        layoutAssistApplyImageRegions: e.target.checked,
                      }))
                    }
                  />
                  应用 AI 图片区域建议（高风险实验）
                </label>
                <div className="text-muted-foreground text-xs">
                  仅当当前链路的版式辅助模式不为“关闭”时生效。开启后可能让图片更干净，也可能误判导致图片缺失；默认关闭更稳妥。
                </div>

                <div className="rounded-md border border-border bg-muted/20 p-3 text-xs text-muted-foreground">
                  {isMineruProvider
                    ? "当前为云端 MinerU 链路：这里只控制 AI 版式辅助。"
                    : "当前为 OCR 链路：这里控制页面结构修正与视觉模型；OCR 识字由 OCR 提供方决定，AI 修字/拆行在 OCR 配置中单独开关。"}
                </div>

                {shouldShowVisionModelConfig ? (
                  <>
                    <div className="grid gap-2">
                      <label className="text-muted-foreground text-xs" htmlFor="ai-provider-2">
                        视觉模型提供方
                      </label>
                      <Select
                        id="ai-provider-2"
                        value={aiProvider}
                        onChange={(e) =>
                          setSettings((s) => ({
                            ...s,
                            preferredMainProvider: e.target.value as "openai" | "siliconflow" | "claude",
                            provider:
                              s.provider === "mineru"
                                ? "mineru"
                                : (e.target.value as "openai" | "siliconflow" | "claude"),
                          }))
                        }
                      >
                        {aiProviderOptions.map((option) => (
                          <option key={option.id} value={option.id}>
                            {option.label}
                          </option>
                        ))}
                      </Select>
                    </div>

                    {aiProvider === "openai" ? (
                      <>
                        {isCompatGatewayMode ? (
                          <div className="border border-amber-500/40 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                            当前处于 OpenAI 兼容模式（第三方网关）。建议后续升级为原生接口接入，以提升稳定性和兼容性。
                          </div>
                        ) : null}
                        <div className="grid gap-2">
                          <label
                            className="font-mono text-xs uppercase tracking-[0.14em] text-muted-foreground"
                            htmlFor="openai-key"
                          >
                            OpenAI API Key
                          </label>
                          <Input
                            id="openai-key"
                            type="password"
                            autoComplete="off"
                            value={settings.openaiApiKey}
                            onChange={(e) =>
                              setSettings((s) => ({ ...s, openaiApiKey: e.target.value }))
                            }
                            placeholder="sk-..."
                          />
                        </div>

                        <div className="grid gap-2">
                          <label
                            className="text-muted-foreground text-xs"
                            htmlFor="openai-base-url"
                          >
                            OpenAI 兼容 Base URL（可选）
                          </label>
                          <Input
                            id="openai-base-url"
                            type="text"
                            autoComplete="off"
                            value={settings.openaiBaseUrl}
                            onChange={(e) =>
                              setSettings((s) => ({ ...s, openaiBaseUrl: e.target.value }))
                            }
                            placeholder="https://api.openai.com/v1"
                          />
                        </div>

                        <div className="grid gap-2">
                          <label
                            className="text-muted-foreground text-xs"
                            htmlFor="openai-model"
                          >
                            OpenAI 兼容视觉模型
                          </label>
                          <Select
                            id="openai-model"
                            value={settings.openaiModel}
                            onChange={(e) =>
                              setSettings((s) => ({ ...s, openaiModel: e.target.value }))
                            }
                            disabled={modelLoading || modelOptions.length === 0}
                          >
                            <option value="">
                              {modelLoading
                                ? "正在加载模型..."
                                : modelOptions.length
                                  ? "请选择模型"
                                  : "暂无可用模型"}
                            </option>
                            {modelOptions.map((model) => (
                              <option key={model} value={model}>
                                {model}
                              </option>
                            ))}
                          </Select>
                          {modelError ? (
                            <div className="text-xs text-destructive">{modelError}</div>
                          ) : null}
                        </div>
                      </>
                    ) : aiProvider === "siliconflow" ? (
                      <>
                        <div className="border border-cyan-500/35 bg-cyan-50 px-3 py-2 text-xs text-cyan-900">
                          SiliconFlow 使用 OpenAI 兼容接口；该通道建议优先填写专用 Base URL 与模型。
                        </div>
                        <div className="grid gap-2">
                          <label
                            className="font-mono text-xs uppercase tracking-[0.14em] text-muted-foreground"
                            htmlFor="siliconflow-key"
                          >
                            SiliconFlow API Key
                          </label>
                          <Input
                            id="siliconflow-key"
                            type="password"
                            autoComplete="off"
                            value={settings.siliconflowApiKey}
                            onChange={(e) =>
                              setSettings((s) => ({ ...s, siliconflowApiKey: e.target.value }))
                            }
                            placeholder="sk-..."
                          />
                        </div>

                        <div className="grid gap-2">
                          <label
                            className="text-muted-foreground text-xs"
                            htmlFor="siliconflow-base-url"
                          >
                            SiliconFlow Base URL
                          </label>
                          <Input
                            id="siliconflow-base-url"
                            type="text"
                            autoComplete="off"
                            value={settings.siliconflowBaseUrl}
                            onChange={(e) =>
                              setSettings((s) => ({ ...s, siliconflowBaseUrl: e.target.value }))
                            }
                            placeholder={SILICONFLOW_BASE_URL}
                          />
                        </div>

                        <div className="grid gap-2">
                          <label
                            className="text-muted-foreground text-xs"
                            htmlFor="siliconflow-model"
                          >
                            SiliconFlow 视觉模型
                          </label>
                          <Select
                            id="siliconflow-model"
                            value={settings.siliconflowModel}
                            onChange={(e) =>
                              setSettings((s) => ({ ...s, siliconflowModel: e.target.value }))
                            }
                            disabled={modelLoading || modelOptions.length === 0}
                          >
                            <option value="">
                              {modelLoading
                                ? "正在加载模型..."
                                : modelOptions.length
                                  ? "请选择模型"
                                  : "暂无可用模型"}
                            </option>
                            {modelOptions.map((model) => (
                              <option key={model} value={model}>
                                {model}
                              </option>
                            ))}
                          </Select>
                          {modelError ? (
                            <div className="text-xs text-destructive">{modelError}</div>
                          ) : null}
                        </div>
                      </>
                    ) : (
                      <div className="grid gap-2">
                        <label
                          className="font-mono text-xs uppercase tracking-[0.14em] text-muted-foreground"
                          htmlFor="claude-key"
                        >
                          Claude API Key
                        </label>
                        <Input
                          id="claude-key"
                          type="password"
                          autoComplete="off"
                          value={settings.claudeApiKey}
                          onChange={(e) =>
                            setSettings((s) => ({ ...s, claudeApiKey: e.target.value }))
                          }
                          placeholder="sk-ant-..."
                        />
                      </div>
                    )}
                    {localOcrAiPostprocessRequested ? (
                      <div className="rounded-md border border-emerald-500/25 bg-emerald-50 px-3 py-2 text-xs text-emerald-900">
                        当前 OCR 已开启 AI OCR 后处理：会复用这里的视觉模型做文字修正与逐行拆分；它不改变 OCR 提供方，也不等同于 AI 版式辅助。
                      </div>
                    ) : null}

                    {localOcrAiPostprocessRequested && aiProvider === "claude" ? (
                      <div className="rounded-md border border-amber-500/35 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                        Claude 目前只用于版式辅助，不会参与当前 OCR 链路的 AI 后处理。若要使用 AI 修字/拆行，请切换到 OpenAI 兼容视觉模型提供方。
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="text-muted-foreground text-sm">
                    当前未启用 AI 版式辅助，也未开启本地 OCR 的 AI 后处理，已折叠视觉模型配置。
                  </div>
                )}
              </section>
            </AdvancedReveal>
          </CardContent>

            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
