"use client"

import * as React from "react"
import Link from "next/link"
import { ArrowLeftIcon, CheckIcon, KeyRoundIcon } from "lucide-react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/select"

import {
  SILICONFLOW_BASE_URL,
  SETTINGS_STORAGE_KEY,
  defaultSettings,
  loadStoredSettings,
  type OcrAiProvider,
  type Settings,
} from "@/lib/settings"
import {
  apiFetch,
  clearStoredApiOrigin,
  getStoredApiOrigin,
  normalizeFetchError,
  resolveApiOrigin,
  setStoredApiOrigin,
} from "@/lib/api"

type ParseEngineMode = "local_ocr" | "remote_ocr" | "mineru_cloud"

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

  const isMineruProvider = settings.provider === "mineru"
  const isOcrEnabledForCurrentEngine = isMineruProvider
    ? settings.mineruHybridOcr
    : true
  const hasBaiduCredentials =
    Boolean(settings.ocrBaiduAppId.trim()) &&
    Boolean(settings.ocrBaiduApiKey.trim()) &&
    Boolean(settings.ocrBaiduSecretKey.trim())
  const canUseAiOcr = !isMineruProvider
  const selectedOcrProvider: Settings["ocrProvider"] =
    isMineruProvider
      ? settings.ocrProvider === "aiocr" ||
          settings.ocrProvider === "paddle_local"
        ? "auto"
        : settings.ocrProvider
      : settings.ocrProvider === "auto"
        ? "tesseract"
        : settings.ocrProvider
  const parseEngineMode: ParseEngineMode = isMineruProvider
    ? "mineru_cloud"
    : (selectedOcrProvider === "aiocr" ||
        selectedOcrProvider === "baidu")
      ? "remote_ocr"
      : "local_ocr"
  const isRemoteOcrMode = parseEngineMode === "remote_ocr"
  const isOcrProviderAuto = selectedOcrProvider === "auto"
  const isOcrProviderAi = selectedOcrProvider === "aiocr"
  const isOcrProviderPaddleLocal = selectedOcrProvider === "paddle_local"
  const isOcrProviderBaidu = selectedOcrProvider === "baidu"
  const isOcrProviderTesseract = selectedOcrProvider === "tesseract"
  const isAiOcrProviderSelected = isOcrProviderAi || isOcrProviderAuto
  const needsRequiredOcrAiConfig = !isMineruProvider && isOcrProviderAi
  const supportsOptionalOcrAiConfig = !isMineruProvider && isOcrProviderAuto
  const hasAnyOcrAiConfigValue =
    Boolean(settings.ocrAiApiKey.trim()) ||
    Boolean(settings.ocrAiBaseUrl.trim()) ||
    Boolean(settings.ocrAiModel.trim())
  const shouldExpandOptionalOcrAiConfig =
    supportsOptionalOcrAiConfig && hasAnyOcrAiConfigValue
  const shouldShowLocalOcrCheck =
    !isMineruProvider &&
    (isOcrProviderAuto || isOcrProviderTesseract || isOcrProviderPaddleLocal)
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
  const shouldShowBaiduConfig = isOcrProviderBaidu || isOcrProviderAuto
  const shouldShowTesseractConfig = isOcrProviderTesseract || isOcrProviderAuto
  const shouldShowAiVendorAdapter =
    canUseAiOcr &&
    (isOcrProviderAi ||
      (isOcrProviderAuto && hasAnyOcrAiConfigValue))
  const aiProvider =
    settings.provider === "claude"
      ? "claude"
      : settings.provider === "siliconflow"
        ? "siliconflow"
        : "openai"
  const mainModelsApiKeyRaw =
    aiProvider === "siliconflow"
      ? settings.siliconflowApiKey
      : settings.openaiApiKey
  const mainModelsBaseUrlRaw =
    aiProvider === "siliconflow"
      ? (settings.siliconflowBaseUrl || SILICONFLOW_BASE_URL)
      : settings.openaiBaseUrl
  const mainModelsSelectedRaw =
    aiProvider === "siliconflow"
      ? settings.siliconflowModel
      : settings.openaiModel
  const ocrPrimaryApiKey = settings.ocrAiApiKey.trim()
  const ocrPrimaryBaseUrl = settings.ocrAiBaseUrl.trim()
  const ocrPrimaryModel = settings.ocrAiModel.trim()
  const canReuseOcrForMain =
    !isMineruProvider && aiProvider !== "claude" && Boolean(ocrPrimaryApiKey)
  const modelsApiKey = mainModelsApiKeyRaw || (canReuseOcrForMain ? ocrPrimaryApiKey : "")
  const modelsBaseUrl =
    mainModelsBaseUrlRaw || (canReuseOcrForMain ? ocrPrimaryBaseUrl : "")
  const modelsSelected =
    mainModelsSelectedRaw || (canReuseOcrForMain ? ocrPrimaryModel : "")
  const isMainUsingOcrKeyFallback =
    !mainModelsApiKeyRaw.trim() && canReuseOcrForMain
  const isCompatGatewayMode =
    !isMineruProvider &&
    aiProvider === "openai" &&
    isOpenAiCompatibleEndpoint(modelsBaseUrl)
  const aiFeaturesRequested = !isMineruProvider && settings.enableLayoutAssist
  const canLoadModels =
    !isMineruProvider &&
    aiProvider !== "claude" &&
    aiFeaturesRequested &&
    Boolean(modelsApiKey)
  const ocrModelsApiKey = ocrPrimaryApiKey || mainModelsApiKeyRaw.trim()
  const ocrModelsBaseUrl = ocrPrimaryBaseUrl || mainModelsBaseUrlRaw

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

        if (!mounted) return
        setModelOptions(models)
        if (models.length && !models.includes(modelsSelected)) {
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
    aiFeaturesRequested,
  ])

  const canLoadOcrModels =
    canUseAiOcr &&
    isOcrEnabledForCurrentEngine &&
    isAiOcrProviderSelected &&
    Boolean(ocrModelsApiKey)

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
    const apiKey = ocrPrimaryApiKey || mainModelsApiKeyRaw.trim()
    const baseUrl = ocrPrimaryBaseUrl || mainModelsBaseUrlRaw.trim()
    const model = settings.ocrAiModel.trim()
    const provider = (settings.ocrAiProvider || "auto").trim() || "auto"

    if (!apiKey) {
      const message = "请先填写 OCR API Key"
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
    mainModelsApiKeyRaw,
    mainModelsBaseUrlRaw,
    ocrPrimaryApiKey,
    ocrPrimaryBaseUrl,
    settings.ocrAiModel,
    settings.ocrAiProvider,
  ])

  const [editionDate, setEditionDate] = React.useState("----/--/--")

  React.useEffect(() => {
    setEditionDate(
      new Intl.DateTimeFormat("zh-CN", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        timeZone: "Asia/Shanghai",
      }).format(new Date())
    )
  }, [])

  return (
    <div className="min-h-dvh bg-background">
      <div className="mx-auto w-full max-w-screen-xl px-4 py-6 md:py-10">
        <header className="newsprint-texture border border-border bg-background">
          <div className="grid md:grid-cols-12">
            <div className="border-b border-border px-4 py-5 md:col-span-9 md:border-b-0 md:border-r md:px-6 md:py-6">
              <p className="font-mono text-[11px] uppercase tracking-[0.24em] text-neutral-600">
                第 1 卷 | {editionDate} | 配置专刊
              </p>
              <h1 className="mt-3 font-serif text-5xl leading-[0.92] tracking-tight md:text-7xl">
                设置中心
              </h1>
              <p className="drop-cap mt-4 max-w-3xl text-sm leading-relaxed text-justify md:text-base">
                配置模型提供方、接口密钥与 OCR 策略。设置仅保存在当前浏览器 localStorage。
              </p>
            </div>

            <div className="flex flex-col justify-between gap-4 px-4 py-5 md:col-span-3 md:px-6 md:py-6">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">本地保存</Badge>
                <Badge className="border-[#cc0000] bg-[#cc0000] text-[#f9f9f7]">
                  OCR 可选
                </Badge>
              </div>

              <Button asChild variant="outline" className="w-full justify-center">
                <Link href="/" aria-label="返回上传页">
                  <ArrowLeftIcon className="size-4" />
                  返回上传页
                </Link>
              </Button>
            </div>
          </div>
        </header>

        <Card className="mt-4 border border-border py-0">
          <CardHeader className="border-b border-border py-5">
            <CardTitle>接口与处理配置</CardTitle>
            <CardDescription>
              先选择处理模式，再配置 OCR 与 AI 参数。配置仅保存在本地浏览器。
            </CardDescription>
          </CardHeader>

          <CardContent className="!px-0">
            <section className="flex flex-col gap-3 border-b border-border p-5">
              <div className="flex items-center justify-between">
                <div className="font-sans text-sm font-semibold uppercase tracking-[0.14em]">
                  解析引擎
                </div>
                <Badge variant="secondary">
                  {parseEngineModeLabels[parseEngineMode]}
                </Badge>
              </div>

              <div className="grid grid-cols-1 gap-2 sm:grid-cols-4">
                {parseEngineOptions.map((p) => (
                  <Button
                    key={p.id}
                    type="button"
                    variant={p.id === parseEngineMode ? "default" : "outline"}
                    onClick={() =>
                      setSettings((s) => {
                        const hasBaiduCredsInState =
                          Boolean(s.ocrBaiduAppId.trim()) &&
                          Boolean(s.ocrBaiduApiKey.trim()) &&
                          Boolean(s.ocrBaiduSecretKey.trim())

                        if (p.id === "mineru_cloud") {
                          const nextOcrProvider: Settings["ocrProvider"] =
                            s.ocrProvider === "aiocr" ||
                              s.ocrProvider === "paddle_local"
                              ? hasBaiduCredsInState
                                ? "baidu"
                                : "auto"
                              : s.ocrProvider === "auto" && hasBaiduCredsInState
                                ? "baidu"
                                : s.ocrProvider
                          return {
                            ...s,
                            provider: "mineru",
                            ocrProvider: nextOcrProvider,
                          }
                        }

                        if (p.id === "remote_ocr") {
                          const nextOcrProvider: Settings["ocrProvider"] =
                            s.ocrProvider === "aiocr" ||
                              s.ocrProvider === "baidu"
                              ? s.ocrProvider
                              : hasBaiduCredsInState
                                ? "baidu"
                                : "aiocr"
                          return {
                            ...s,
                            provider:
                              s.provider === "mineru"
                                ? "openai"
                                : s.provider,
                            ocrProvider: nextOcrProvider,
                          }
                        }

                        const nextOcrProvider: Settings["ocrProvider"] =
                          s.ocrProvider === "paddle_local" ? "paddle_local" : "tesseract"
                        return {
                          ...s,
                          provider:
                            s.provider === "mineru"
                              ? "openai"
                              : s.provider,
                          ocrProvider: nextOcrProvider,
                        }
                      })
                    }
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

            </section>

            <section className="grid gap-4 border-b border-border p-5">
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
                  自动模式会探测 `:8000` 与 `:8001`；公开部署可用 `NEXT_PUBLIC_API_URL` 固定地址。
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

            <section className="grid gap-4 border-b border-border p-5">
              <div className="font-sans text-sm font-semibold uppercase tracking-[0.14em]">
                处理策略
              </div>

              {isMineruProvider ? (
                <>
                  {showAdvanced ? (
                    <div className="grid gap-2">
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
                      {settings.mineruHybridOcr && settings.textEraseMode === "fill" ? (
                        <div className="border border-amber-500/40 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                          当前为 MinerU 混合 OCR。纯色填充已启用“精细框定位”策略（不再自动切回智能模式），建议先用一页样本确认效果。
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      className="h-4 w-4 accent-[#111111]"
                      checked={settings.mineruHybridOcr}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          mineruHybridOcr: e.target.checked,
                        }))
                      }
                    />
                    启用混合 OCR 定位（实验）
                  </label>
                  <div className="text-muted-foreground text-sm">
                    仅在 MinerU 模式使用本地 OCR 辅助定位，OCR 提供方可在下方单独选择。
                  </div>
                </>
              ) : (
                <>
                  <div className="text-muted-foreground text-sm">
                    本地/远程 OCR 模式默认启用 OCR。
                  </div>
                </>
              )}

              {!isMineruProvider && showAdvanced ? (
                <div className="grid gap-2">
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
              ) : null}

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
            </section>

            {isOcrEnabledForCurrentEngine ? (
              <section className="grid gap-4 p-5">
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
                  {isRemoteOcrMode ? (
                    <>
                      <option value="aiocr">AI OCR（OpenAI 兼容）</option>
                      <option value="baidu">百度 OCR</option>
                    </>
                  ) : parseEngineMode === "local_ocr" ? (
                    <>
                      <option value="tesseract">Tesseract（本地）</option>
                      <option value="paddle_local">PaddleOCR（本地）</option>
                    </>
                  ) : (
                    <>
                      <option value="auto">自动（推荐）</option>
                      <option value="baidu">百度 OCR</option>
                      <option value="tesseract">Tesseract（本地）</option>
                    </>
                  )}
                </Select>
                <div className="text-muted-foreground text-xs">
                  当前模式：{ocrProviderLabels[selectedOcrProvider]}。
                  {isOcrProviderAi
                    ? " 显式 AI OCR 为严格执行：失败即报错，不自动回退。"
                    : isOcrProviderAuto
                      ? " 自动模式优先本地 OCR，可选百度。"
                      : isOcrProviderPaddleLocal
                        ? " 使用本地 PaddleOCR。"
                        : isOcrProviderBaidu
                          ? " 使用百度 OCR。"
                          : " 使用本地 Tesseract。"}
                </div>
              {isMineruProvider && selectedOcrProvider === "auto" && hasBaiduCredentials ? (
                  <div className="text-muted-foreground text-xs">
                    已检测到百度 OCR 凭证。MinerU 混合 OCR 任务会默认优先使用百度 OCR。
                  </div>
                ) : null}
              </div>

              {showAdvanced ? (
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
                    OCR 严格模式（失败即报错，不自动降级/回退）
                  </label>
                  <div className="text-muted-foreground text-xs">
                    关闭后会启用最佳努力策略：失败时允许降级或以图片页继续。
                  </div>
                </>
              ) : null}

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

              {(needsRequiredOcrAiConfig || shouldExpandOptionalOcrAiConfig) ? (
                <div className="grid gap-3 border border-border bg-muted/20 p-3">
                  <div className="font-sans text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                    OCR AI 接口参数
                  </div>

                  <div className="grid gap-2">
                    <label
                      className="text-muted-foreground text-xs"
                      htmlFor="ocr-ai-api-key"
                    >
                      OCR API Key（主 Key）
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
                      OCR 模型{needsRequiredOcrAiConfig ? "（必填）" : "（可选）"}
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

                  {isMainUsingOcrKeyFallback ? (
                    <div className="text-muted-foreground text-xs">
                      主 AI Key 为空：主模型拉取与可选 AI 辅助会复用 OCR 主 Key（不影响 OCR Key 生效）。
                    </div>
                  ) : null}

                </div>
              ) : null}

              {supportsOptionalOcrAiConfig && !shouldExpandOptionalOcrAiConfig ? (
                <div className="text-muted-foreground text-xs">
                  自动模式下 OCR AI 参数可留空。
                </div>
              ) : null}

              {!canUseAiOcr ? null : !isAiOcrProviderSelected ? (
                <div className="text-muted-foreground text-xs">
                  当前为纯本地 OCR，AI OCR 厂商配置已折叠。
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

            {!isMineruProvider && showAdvanced ? (
              <section className="grid gap-4 border-t border-border p-5">
                <div className="font-sans text-sm font-semibold uppercase tracking-[0.14em]">
                  AI 辅助（可选）
                </div>

                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-[#111111]"
                    checked={settings.enableLayoutAssist}
                    onChange={(e) =>
                      setSettings((s) => ({
                        ...s,
                        enableLayoutAssist: e.target.checked,
                      }))
                    }
                  />
                  启用 AI 版式辅助（实验）
                </label>
                <div className="text-muted-foreground text-xs">
                  默认仅优化阅读顺序和表格网格；图片区域建议默认关闭，按需手动开启。
                </div>

                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-[#111111]"
                    checked={settings.layoutAssistApplyImageRegions}
                    disabled={!settings.enableLayoutAssist}
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
                  开启后可能让图片更干净，也可能误判导致图片缺失；默认关闭更稳妥。
                </div>

                <div className="grid gap-2">
                  <label
                    className="text-muted-foreground text-xs"
                    htmlFor="ocr-linebreak-assist"
                  >
                    OCR 行级拆分辅助（实验）
                  </label>
                  <Select
                    id="ocr-linebreak-assist"
                    value={settings.ocrAiLinebreakAssistMode}
                    onChange={(e) =>
                      setSettings((s) => ({
                        ...s,
                        ocrAiLinebreakAssistMode: e.target.value as Settings["ocrAiLinebreakAssistMode"],
                      }))
                    }
                  >
                    <option value="auto">自动（推荐）</option>
                    <option value="on">强制开启</option>
                    <option value="off">强制关闭</option>
                  </Select>
                  <div className="text-muted-foreground text-xs">
                    使用视觉模型辅助判断块内换行，把粗粒度文本框拆为行级文本框，可改善字号、颜色和段落对齐稳定性。自动模式下后端会按 OCR 引擎能力自行决定（例如 AI OCR 场景可自动启用）。
                  </div>
                </div>

                {settings.enableLayoutAssist ? (
                  <>
                    <div className="grid gap-2">
                      <label className="text-muted-foreground text-xs" htmlFor="ai-provider-2">
                        AI 提供方
                      </label>
                      <Select
                        id="ai-provider-2"
                        value={aiProvider}
                        onChange={(e) =>
                          setSettings((s) => ({
                            ...s,
                            provider: e.target.value as "openai" | "siliconflow" | "claude",
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
                            OpenAI 兼容模型
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
                            SiliconFlow 模型
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
                  </>
                ) : (
                  <div className="text-muted-foreground text-sm">
                    当前未启用 AI 辅助，已折叠 AI 相关配置。
                  </div>
                )}
              </section>
            ) : null}
          </CardContent>

          <CardFooter className="border-t border-border justify-between gap-3 py-5">
            <Button type="button" variant="outline" onClick={onClear}>
              清空本地配置
            </Button>

            <div className="flex items-center gap-3">
              {lastSavedAt ? (
                <div className="hidden font-mono text-xs uppercase tracking-[0.14em] text-muted-foreground sm:block">
                  自动保存已开启
                </div>
              ) : null}
              <Button type="button" onClick={onSave}>
                立即保存
              </Button>
            </div>
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}
