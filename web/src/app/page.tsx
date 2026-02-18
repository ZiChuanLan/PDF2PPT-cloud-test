"use client"

import * as React from "react"
import Link from "next/link"
import {
  ArrowRightIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  DownloadIcon,
  FileTextIcon,
  ListChecksIcon,
  Settings2Icon,
  UploadCloudIcon,
  XCircleIcon,
} from "lucide-react"
import { useDropzone } from "react-dropzone"
import { toast } from "sonner"

import { cn } from "@/lib/utils"
import { apiFetch, normalizeFetchError } from "@/lib/api"
import { SILICONFLOW_BASE_URL, defaultSettings, loadStoredSettings, type Settings } from "@/lib/settings"
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
import { Progress } from "@/components/ui/progress"
import { PdfCanvasPreview } from "@/components/pdf-canvas-preview"
import { useUploadSession } from "@/components/upload-session-provider"

type JobStatusValue = "pending" | "processing" | "completed" | "failed" | "cancelled"
type JobQueueState = "queued" | "running" | "waiting" | "done"

type JobListItem = {
  job_id: string
  status: JobStatusValue
  stage: string
  progress: number
  created_at: string
  expires_at: string
  message?: string | null
  error?: { code?: string; message?: string } | null
  queue_position?: number | null
  queue_state?: JobQueueState | string | null
}

type JobListResponse = {
  jobs: JobListItem[]
  queue_size: number
  returned: number
}

type JobStatusResponse = {
  job_id: string
  status: JobStatusValue
  stage: string
  progress: number
  created_at: string
  expires_at: string
  message?: string | null
  error?: { code?: string; message?: string } | null
}

type RunConfig = {
  parseProvider: "local" | "mineru"
  llmProvider: "openai" | "claude"
  mainApiKey: string
  mainBaseUrl: string
  mainModel: string
  effectiveOcrProvider: string
  effectiveOcrAiKey: string
  effectiveOcrAiBaseUrl: string
  effectiveOcrAiModel: string
  effectiveOcrAiProvider: string
}

type ValidationResult = {
  ok: boolean
  message?: string
}

const TERMINAL_STATUSES = new Set<JobStatusValue>(["completed", "failed", "cancelled"])

const jobStatusLabels: Record<JobStatusValue, string> = {
  pending: "排队中",
  processing: "处理中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
}

const jobStageLabels: Record<string, string> = {
  upload_received: "上传接收",
  queued: "队列等待",
  parsing: "解析 PDF",
  ocr: "OCR 识别",
  layout_assist: "版式辅助",
  pptx_generating: "生成 PPTX",
  packaging: "打包",
  cleanup: "清理",
  done: "已完成",
}

const jobStageFlow = [
  "queued",
  "parsing",
  "ocr",
  "layout_assist",
  "pptx_generating",
  "packaging",
  "done",
] as const

const homeTickerItems = [
  "结构优先：先排版，再装饰",
  "单任务单状态：避免多重反馈噪音",
  "高级配置全部收敛到设置页",
  "跟踪中心负责可观测性与排障",
]
const HOME_ACTIVE_JOB_STORAGE_KEY = "ppt-opencode:home:active-job-id"

function toJobStatusResponse(row: JobListItem): JobStatusResponse {
  return {
    job_id: row.job_id,
    status: row.status,
    stage: row.stage,
    progress: row.progress,
    created_at: row.created_at,
    expires_at: row.expires_at,
    message: row.message ?? null,
    error: row.error ?? null,
  }
}

function formatDateTime(iso: string) {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Shanghai",
  }).format(date)
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB"] as const
  const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / Math.pow(1024, idx)
  return `${value.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`
}

function getMainProviderConfig(settings: Settings) {
  if (settings.provider === "siliconflow") {
    return {
      provider: "openai" as const,
      apiKey: settings.siliconflowApiKey.trim(),
      baseUrl: settings.siliconflowBaseUrl.trim() || SILICONFLOW_BASE_URL,
      model: settings.siliconflowModel.trim(),
    }
  }
  if (settings.provider === "claude") {
    return {
      provider: "claude" as const,
      apiKey: settings.claudeApiKey.trim(),
      baseUrl: "",
      model: "",
    }
  }
  return {
    provider: "openai" as const,
    apiKey: settings.openaiApiKey.trim(),
    baseUrl: settings.openaiBaseUrl.trim(),
    model: settings.openaiModel.trim(),
  }
}

function resolveRunConfig(settings: Settings): RunConfig {
  const parseProvider: RunConfig["parseProvider"] =
    settings.provider === "mineru" ? "mineru" : "local"
  const main = getMainProviderConfig(settings)
  const canReuseMainForOcr = Boolean(main.apiKey) && main.provider === "openai"

  const rawOcrProvider = (settings.ocrProvider || "auto").trim().toLowerCase()
  const effectiveOcrProvider =
    parseProvider === "mineru"
      ? (rawOcrProvider === "aiocr" ? "auto" : rawOcrProvider)
      : rawOcrProvider

  const effectiveOcrAiKey =
    settings.ocrAiApiKey.trim() || (canReuseMainForOcr ? main.apiKey : "")
  const effectiveOcrAiBaseUrl =
    settings.ocrAiBaseUrl.trim() || (canReuseMainForOcr ? main.baseUrl : "")
  const effectiveOcrAiModel =
    settings.ocrAiModel.trim() || (canReuseMainForOcr ? main.model : "")
  const effectiveOcrAiProvider = (settings.ocrAiProvider || "auto").trim() || "auto"


  return {
    parseProvider,
    llmProvider: main.provider,
    mainApiKey: main.apiKey,
    mainBaseUrl: main.baseUrl,
    mainModel: main.model,
    effectiveOcrProvider,
    effectiveOcrAiKey,
    effectiveOcrAiBaseUrl,
    effectiveOcrAiModel,
    effectiveOcrAiProvider,
  }
}

function validateBeforeRun(settings: Settings): ValidationResult {
  const run = resolveRunConfig(settings)

  if (run.parseProvider === "mineru" && !settings.mineruApiToken.trim()) {
    return { ok: false, message: "当前为 MinerU 解析，请先在设置页填写 MinerU API Token。" }
  }

  if (run.parseProvider === "local") {
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
      if (!run.effectiveOcrAiKey) {
        return { ok: false, message: "当前 OCR 需要 AI Key，请在设置页补充 OCR API Key。" }
      }
    }
  }

  return { ok: true }
}

function toIntOrUndefined(value: string): number | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const n = Number(trimmed)
  if (!Number.isFinite(n)) return undefined
  const i = Math.floor(n)
  if (i <= 0) return undefined
  return i
}

function clampPositiveInt(value: number, max?: number) {
  const normalized = Number.isFinite(value) ? Math.max(1, Math.floor(value)) : 1
  if (!max || max <= 0) return normalized
  return Math.min(normalized, max)
}

function toFiniteFloatStringOrUndefined(value: string): string | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const n = Number(trimmed)
  if (!Number.isFinite(n)) return undefined
  return String(n)
}

function createFormData(
  file: File,
  settings: Settings,
  pageStart?: number,
  pageEnd?: number
): FormData {
  const run = resolveRunConfig(settings)
  const form = new FormData()

  form.append("file", file)
  form.append("parse_provider", run.parseProvider)
  form.append("provider", run.llmProvider)

  if (run.mainApiKey) form.append("api_key", run.mainApiKey)
  if (run.mainBaseUrl) form.append("base_url", run.mainBaseUrl)
  if (run.mainModel) form.append("model", run.mainModel)

  form.append("enable_layout_assist", String(Boolean(settings.enableLayoutAssist)))
  form.append(
    "layout_assist_apply_image_regions",
    String(Boolean(settings.layoutAssistApplyImageRegions))
  )
  form.append("enable_ocr", String(Boolean(settings.enableOcr)))
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
    form.append("mineru_hybrid_ocr", String(Boolean(settings.mineruHybridOcr)))
    if (settings.mineruBaseUrl.trim()) form.append("mineru_base_url", settings.mineruBaseUrl.trim())
    if (settings.mineruLanguage.trim()) form.append("mineru_language", settings.mineruLanguage.trim())
  }

  if (run.parseProvider === "local") {
    form.append("ocr_provider", run.effectiveOcrProvider)

    const shouldAttachOcrAiParams =
      run.effectiveOcrProvider === "aiocr" ||
      Boolean(settings.ocrAiLinebreakAssistMode === "on" && run.effectiveOcrAiKey) ||
      Boolean(settings.enableLayoutAssist && run.effectiveOcrAiKey)
    if (shouldAttachOcrAiParams) {
      if (run.effectiveOcrAiKey) form.append("ocr_ai_api_key", run.effectiveOcrAiKey)
      if (run.effectiveOcrAiBaseUrl) form.append("ocr_ai_base_url", run.effectiveOcrAiBaseUrl)
      if (run.effectiveOcrAiModel) form.append("ocr_ai_model", run.effectiveOcrAiModel)
      form.append("ocr_ai_provider", run.effectiveOcrAiProvider)
    }
    if (settings.ocrAiLinebreakAssistMode === "on") {
      form.append("ocr_ai_linebreak_assist", "true")
    } else if (settings.ocrAiLinebreakAssistMode === "off") {
      form.append("ocr_ai_linebreak_assist", "false")
    }

    if (run.effectiveOcrProvider === "baidu") {
      form.append("ocr_baidu_app_id", settings.ocrBaiduAppId.trim())
      form.append("ocr_baidu_api_key", settings.ocrBaiduApiKey.trim())
      form.append("ocr_baidu_secret_key", settings.ocrBaiduSecretKey.trim())
    }

    if (run.effectiveOcrProvider === "tesseract" || run.effectiveOcrProvider === "auto") {
      if (settings.ocrTesseractLanguage.trim()) {
        form.append("ocr_tesseract_language", settings.ocrTesseractLanguage.trim())
      }
      const minConf = Number(settings.ocrTesseractMinConfidence)
      if (Number.isFinite(minConf)) {
        form.append("ocr_tesseract_min_confidence", String(minConf))
      }
    }
  }

  if (pageStart && pageEnd) {
    form.append("page_start", String(pageStart))
    form.append("page_end", String(pageEnd))
  }

  return form
}

export default function Home() {
  const [settingsSnapshot, setSettingsSnapshot] = React.useState<Settings>(defaultSettings)
  const {
    file,
    setFile,
    pageStartInput,
    setPageStartInput,
    pageEndInput,
    setPageEndInput,
    clearUpload,
  } = useUploadSession()

  const [jobId, setJobId] = React.useState<string | null>(null)
  const [activeJob, setActiveJob] = React.useState<JobStatusResponse | null>(null)
  const [isSubmitting, setIsSubmitting] = React.useState(false)
  const [actionError, setActionError] = React.useState<string | null>(null)
  const [previewPageInput, setPreviewPageInput] = React.useState("1")
  const [previewPageCount, setPreviewPageCount] = React.useState(0)
  const [usePageRange, setUsePageRange] = React.useState(
    Boolean(pageStartInput.trim() || pageEndInput.trim())
  )

  const [jobs, setJobs] = React.useState<JobListItem[]>([])
  const [queueSize, setQueueSize] = React.useState(0)
  const [jobsLoading, setJobsLoading] = React.useState(false)
  const [isJobIdHydrated, setIsJobIdHydrated] = React.useState(false)
  const jobIdRef = React.useRef<string | null>(null)
  const lastTerminalToastRef = React.useRef<{
    jobId: string | null
    status: JobStatusValue | null
  }>({
    jobId: null,
    status: null,
  })

  const runConfig = React.useMemo(() => resolveRunConfig(settingsSnapshot), [settingsSnapshot])
  const runModelLabel = React.useMemo(() => {
    if (runConfig.parseProvider === "local" && runConfig.effectiveOcrProvider !== "aiocr") {
      if (settingsSnapshot.ocrAiLinebreakAssistMode === "on" && runConfig.effectiveOcrAiModel) {
        return `${runConfig.effectiveOcrAiModel}（仅用于行级拆分辅助）`
      }
      return "本地 OCR（无需远程模型）"
    }
    return runConfig.effectiveOcrAiModel || runConfig.mainModel || "未设置"
  }, [runConfig, settingsSnapshot.ocrAiLinebreakAssistMode])

  const refreshSettingsSnapshot = React.useCallback(() => {
    setSettingsSnapshot(loadStoredSettings())
  }, [])

  const fetchJobs = React.useCallback(async (silent = true) => {
    if (!silent) setJobsLoading(true)
    try {
      const response = await apiFetch("/jobs?limit=50")
      if (!response.ok) {
        throw new Error("加载任务列表失败")
      }
      const body = (await response.json().catch(() => null)) as JobListResponse | null
      const rows = Array.isArray(body?.jobs) ? body.jobs : []
      setJobs(rows)
      setQueueSize(typeof body?.queue_size === "number" ? Math.max(0, body.queue_size) : 0)

      const currentJobId = jobIdRef.current
      if (currentJobId) {
        const matched = rows.find((row) => row.job_id === currentJobId)
        if (matched) {
          setActiveJob(toJobStatusResponse(matched))
          if (TERMINAL_STATUSES.has(matched.status)) {
            setIsSubmitting(false)
          }
        }
      }
    } catch (e) {
      if (!silent) {
        setActionError(normalizeFetchError(e, "加载任务列表失败"))
      }
    } finally {
      if (!silent) setJobsLoading(false)
    }
  }, [])

  const fetchJobStatus = React.useCallback(async (targetJobId: string) => {
    const response = await apiFetch(`/jobs/${targetJobId}`)
    if (!response.ok) {
      throw new Error("查询任务状态失败")
    }
    const body = (await response.json().catch(() => null)) as JobStatusResponse | null
    if (!body || typeof body !== "object") {
      throw new Error("任务状态响应异常")
    }
    return body
  }, [])

  const onDrop = React.useCallback((accepted: File[]) => {
    const next = accepted[0] ?? null
    setFile(next)
    setActionError(null)
    if (next) {
      setPageStartInput("")
      setPageEndInput("")
      setPreviewPageInput("1")
      setPreviewPageCount(0)
      setUsePageRange(false)
    } else {
      clearUpload()
      setPreviewPageInput("1")
      setPreviewPageCount(0)
      setUsePageRange(false)
    }
  }, [clearUpload, setFile, setPageEndInput, setPageStartInput])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
    disabled: isSubmitting,
    onDrop,
  })

  const handleConvert = React.useCallback(async () => {
    if (!file) return

    setActionError(null)

    const validation = validateBeforeRun(settingsSnapshot)
    if (!validation.ok) {
      setActionError(validation.message || "配置校验失败")
      return
    }

    const pageStart = usePageRange ? toIntOrUndefined(pageStartInput) : undefined
    const pageEnd = usePageRange ? toIntOrUndefined(pageEndInput) : undefined
    if (usePageRange && ((pageStart && !pageEnd) || (!pageStart && pageEnd))) {
      setActionError("页码范围请同时填写起始页和结束页")
      return
    }
    if (usePageRange && pageStart && pageEnd && pageStart > pageEnd) {
      setActionError("页码范围错误：起始页不能大于结束页")
      return
    }

    setIsSubmitting(true)
    setJobId(null)
    setActiveJob(null)

    try {
      const formData = createFormData(file, settingsSnapshot, pageStart, pageEnd)
      const response = await apiFetch("/jobs", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.message || "创建任务失败")
      }

      const body = (await response.json().catch(() => null)) as { job_id?: string } | null
      const nextJobId = typeof body?.job_id === "string" ? body.job_id : ""
      if (!nextJobId) {
        throw new Error("创建任务失败：未返回任务号")
      }

      setJobId(nextJobId)
      toast.success("任务创建成功，正在处理中")

      try {
        const status = await fetchJobStatus(nextJobId)
        setActiveJob(status)
      } catch {
        // ignore immediate poll failure
      }

      void fetchJobs(true)
    } catch (e) {
      setActionError(normalizeFetchError(e, "创建任务失败"))
      setIsSubmitting(false)
    }
  }, [fetchJobStatus, fetchJobs, file, pageEndInput, pageStartInput, settingsSnapshot, usePageRange])

  const handleCancelCurrentJob = React.useCallback(async () => {
    if (!jobId) return
    try {
      await apiFetch(`/jobs/${jobId}/cancel`, { method: "POST" })
      toast("已发送取消请求")
      void fetchJobs(true)
    } catch {
      toast.error("取消请求失败")
    }
  }, [fetchJobs, jobId])

  const handleDownload = React.useCallback(async (targetJobId: string) => {
    const response = await apiFetch(`/jobs/${targetJobId}/download`)
    if (!response.ok) {
      const body = await response.json().catch(() => null)
      throw new Error(body?.message || `下载失败（HTTP ${response.status}）`)
    }
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `output-${targetJobId.slice(0, 8)}.pptx`
    document.body.appendChild(a)
    a.click()
    a.remove()
    window.URL.revokeObjectURL(url)
  }, [])

  const handleCancelById = React.useCallback(
    async (targetJobId: string) => {
      try {
        await apiFetch(`/jobs/${targetJobId}/cancel`, { method: "POST" })
        toast("取消请求已发送")
        void fetchJobs(true)
      } catch {
        toast.error("取消失败")
      }
    },
    [fetchJobs]
  )

  const handleResetAll = React.useCallback(() => {
    clearUpload()
    setJobId(null)
    setActiveJob(null)
    setIsSubmitting(false)
    setActionError(null)
    setPreviewPageInput("1")
    setPreviewPageCount(0)
    setUsePageRange(false)
    setPageStartInput("")
    setPageEndInput("")
  }, [clearUpload, setPageEndInput, setPageStartInput])

  React.useEffect(() => {
    jobIdRef.current = jobId
  }, [jobId])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    const storedJobId = window.localStorage.getItem(HOME_ACTIVE_JOB_STORAGE_KEY)
    if (storedJobId) {
      setJobId((prev) => prev || storedJobId)
    }
    setIsJobIdHydrated(true)
  }, [])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    if (!isJobIdHydrated) return
    if (jobId) {
      window.localStorage.setItem(HOME_ACTIVE_JOB_STORAGE_KEY, jobId)
    } else {
      window.localStorage.removeItem(HOME_ACTIVE_JOB_STORAGE_KEY)
    }
  }, [isJobIdHydrated, jobId])

  React.useEffect(() => {
    refreshSettingsSnapshot()
    void fetchJobs(false)

    const onFocus = () => {
      refreshSettingsSnapshot()
      void fetchJobs(true)
    }

    window.addEventListener("focus", onFocus)
    const timer = window.setInterval(() => {
      void fetchJobs(true)
    }, 4000)

    return () => {
      window.removeEventListener("focus", onFocus)
      window.clearInterval(timer)
    }
  }, [fetchJobs, refreshSettingsSnapshot])

  React.useEffect(() => {
    if (!jobId) return

    let mounted = true
    let timer: number | null = null

    const stopPolling = () => {
      if (timer !== null) {
        window.clearInterval(timer)
        timer = null
      }
    }

    const poll = async () => {
      try {
        const status = await fetchJobStatus(jobId)
        if (!mounted) return
        setActiveJob(status)
        if (TERMINAL_STATUSES.has(status.status)) {
          setIsSubmitting(false)
          stopPolling()
          void fetchJobs(true)
        }
      } catch {
        // ignore transient polling error
      }
    }

    void poll()
    timer = window.setInterval(() => {
      void poll()
    }, 2000)

    return () => {
      mounted = false
      stopPolling()
    }
  }, [fetchJobStatus, fetchJobs, jobId])

  React.useEffect(() => {
    if (!activeJob) return
    if (!TERMINAL_STATUSES.has(activeJob.status)) return

    const hasNotified =
      lastTerminalToastRef.current.jobId === activeJob.job_id &&
      lastTerminalToastRef.current.status === activeJob.status
    if (hasNotified) return

    lastTerminalToastRef.current = {
      jobId: activeJob.job_id,
      status: activeJob.status,
    }

    if (activeJob.status === "completed") {
      toast.success("转换完成，可下载 PPTX")
    } else if (activeJob.status === "failed") {
      setActionError(activeJob.error?.message || "转换失败")
      toast.error(activeJob.error?.message || "转换失败")
    } else if (activeJob.status === "cancelled") {
      toast("任务已取消")
    }
  }, [activeJob])

  const progressValue = Math.max(0, Math.min(100, Number(activeJob?.progress || 0)))
  const currentStatus = activeJob?.status || (isSubmitting ? "processing" : "pending")
  const currentStageCode = activeJob?.stage || (isSubmitting ? "queued" : "")
  const currentStageLabel = activeJob?.stage
    ? (jobStageLabels[activeJob.stage] ?? activeJob.stage)
    : "等待开始"
  const stageFlowIndex = currentStageCode
    ? jobStageFlow.findIndex((stage) => stage === currentStageCode)
    : -1
  const stageLiveText = activeJob
    ? `任务状态 ${jobStatusLabels[currentStatus as JobStatusValue] || currentStatus}，阶段 ${currentStageLabel}，进度 ${progressValue}%`
    : "尚无进行中的任务"
  const inFlightJobs = jobs.filter((row) => row.status === "pending" || row.status === "processing").length
  const failedJobs = jobs.filter((row) => row.status === "failed").length
  const completedJobs = jobs.filter((row) => row.status === "completed").length
  const canStart = Boolean(file) && !isSubmitting
  const [filePreviewUrl, setFilePreviewUrl] = React.useState("")
  React.useEffect(() => {
    if (!file) {
      setFilePreviewUrl("")
      return
    }

    const nextUrl = URL.createObjectURL(file)
    setFilePreviewUrl(nextUrl)

    return () => {
      URL.revokeObjectURL(nextUrl)
    }
  }, [file])
  const previewPage = clampPositiveInt(toIntOrUndefined(previewPageInput) || 1, previewPageCount || undefined)
  const handlePreviewPageCommit = React.useCallback(
    (value: string) => {
      const raw = toIntOrUndefined(value) || 1
      const normalized = clampPositiveInt(raw, previewPageCount || undefined)
      setPreviewPageInput(String(normalized))
    },
    [previewPageCount]
  )
  const handlePreviewPageCountChange = React.useCallback((count: number) => {
    setPreviewPageCount(count)
    setPreviewPageInput((prev) =>
      String(clampPositiveInt(toIntOrUndefined(prev) || 1, count))
    )
  }, [])
  const editionDate = new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    timeZone: "Asia/Shanghai",
  }).format(new Date())

  return (
    <div className="min-h-dvh bg-background">
      <div className="mx-auto w-full max-w-screen-xl px-4 py-6 md:py-10">
        <header className="newsprint-texture border border-border bg-background">
          <div className="grid md:grid-cols-12">
            <div className="border-b border-border px-4 py-5 md:col-span-8 md:border-b-0 md:border-r md:px-6 md:py-6">
              <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
                第 1 版 · {editionDate} · PDF 编排台
              </div>
              <h1 className="mt-3 font-serif text-4xl leading-[0.9] tracking-tight md:text-7xl">
                上传即转换
              </h1>
              <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted-foreground md:text-base">
                首页仅保留任务启动与状态观察。复杂参数集中在设置页，追踪与排障集中在跟踪中心。
              </p>
            </div>
            <div className="flex flex-col justify-between gap-3 px-4 py-5 md:col-span-4 md:px-6 md:py-6">
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">任务面板</Badge>
                <Badge className="border-[#cc0000] bg-[#cc0000] text-[#f9f9f7]">稳定模式</Badge>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button type="button" variant="outline" asChild>
                  <Link href="/tracking">
                    <ListChecksIcon className="size-4" />
                    跟踪中心
                  </Link>
                </Button>
                <Button type="button" asChild>
                  <Link href="/settings">
                    <Settings2Icon className="size-4" />
                    设置页
                  </Link>
                </Button>
              </div>
            </div>
          </div>
          <div className="news-ticker relative overflow-hidden border-t border-border bg-muted/40 py-2">
            <div className="news-ticker-track flex min-w-max gap-8 px-4 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground md:px-6">
              {[...homeTickerItems, ...homeTickerItems].map((item, idx) => (
                <span key={`${item}-${idx}`} className="whitespace-nowrap">
                  {item}
                </span>
              ))}
            </div>
          </div>
        </header>

        <p className="sr-only" role="status" aria-live="polite">
          {stageLiveText}
        </p>

        <main className="mt-6 grid gap-4 lg:grid-cols-2">
          <Card className="hard-shadow-hover border-border">
            <CardHeader>
              <CardTitle className="text-lg">上传与执行</CardTitle>
              <CardDescription>
                首页只做核心操作：选文件、选页码、启动转换。
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                {...getRootProps()}
                className={cn(
                  "cursor-pointer border border-dashed border-border bg-muted/40 p-5 text-center transition-colors",
                  isDragActive && !isDragReject && "bg-accent/50",
                  isDragReject && "border-destructive bg-destructive/10",
                  isSubmitting && "pointer-events-none opacity-60"
                )}
              >
                <input {...getInputProps()} />
                <UploadCloudIcon className="mx-auto size-8 text-muted-foreground" />
                <p className="mt-2 text-sm font-medium">
                  {isDragActive ? "松开以上传 PDF" : "拖拽 PDF 到这里，或点击选择文件"}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">仅支持 .pdf</p>
              </div>

              {file ? (
                <div className="flex items-center justify-between gap-3 border border-border p-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium">{file.name}</div>
                    <div className="text-xs text-muted-foreground">{formatBytes(file.size)}</div>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => {
                      clearUpload()
                      setPreviewPageInput("1")
                      setPreviewPageCount(0)
                      setUsePageRange(false)
                    }}
                  >
                    清空
                  </Button>
                </div>
              ) : null}

              <div className="border border-dashed border-border bg-muted/15 p-3 text-xs text-muted-foreground">
                页码范围与“单页试跑/开始转换”已放到下方 PDF 预览区域，操作路径更连贯。
              </div>

              <div className="border border-border bg-muted/30 p-3">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <Badge variant="outline">解析：{runConfig.parseProvider}</Badge>
                  <Badge variant="outline">OCR：{runConfig.effectiveOcrProvider}</Badge>
                  <Badge variant="outline">模型：{runModelLabel}</Badge>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  当前配置已通过基础校验，参数微调请在设置页完成。
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button type="button" variant="ghost" asChild>
                    <Link href="/settings">
                      进入设置页精调
                      <ArrowRightIcon className="size-4" />
                    </Link>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {filePreviewUrl ? (
            <Card className="border-border">
              <CardHeader className="pb-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <CardTitle className="text-lg">PDF 预览</CardTitle>
                    <CardDescription>自定义预览器。当前页与“单页试跑”始终保持一致。</CardDescription>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Button
                      type="button"
                      variant="outline"
                      size="icon-xs"
                      disabled={previewPage <= 1}
                      onClick={() => {
                        setPreviewPageInput(String(clampPositiveInt(previewPage - 1, previewPageCount || undefined)))
                      }}
                      aria-label="预览上一页"
                    >
                      <ChevronLeftIcon className="size-3" />
                    </Button>
                    <Input
                      inputMode="numeric"
                      value={previewPageInput}
                      onChange={(e) => setPreviewPageInput(e.target.value)}
                      onBlur={(e) => handlePreviewPageCommit(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault()
                          handlePreviewPageCommit((e.target as HTMLInputElement).value)
                        }
                      }}
                      className="h-8 w-20 text-center"
                      aria-label="当前预览页"
                    />
                    <span className="w-14 text-right font-mono text-xs text-muted-foreground">
                      / {previewPageCount || "?"}
                    </span>
                    <Button
                      type="button"
                      variant="outline"
                      size="icon-xs"
                      disabled={previewPageCount > 0 ? previewPage >= previewPageCount : true}
                      onClick={() => {
                        setPreviewPageInput(String(clampPositiveInt(previewPage + 1, previewPageCount || undefined)))
                      }}
                      aria-label="预览下一页"
                    >
                      <ChevronRightIcon className="size-3" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <PdfCanvasPreview
                  fileUrl={filePreviewUrl}
                  page={previewPage}
                  className="mx-auto max-w-[840px]"
                  onPageCountChange={handlePreviewPageCountChange}
                />
                <div className="grid gap-3 border border-border bg-muted/20 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        className="h-4 w-4 accent-[#111111]"
                        checked={usePageRange}
                        onChange={(e) => {
                          const enabled = e.target.checked
                          setUsePageRange(enabled)
                          if (!enabled) {
                            setPageStartInput("")
                            setPageEndInput("")
                          }
                        }}
                      />
                      限定页码范围
                    </label>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        disabled={!file}
                        onClick={() => {
                          setUsePageRange(true)
                          const current = String(previewPage)
                          setPageStartInput(current)
                          setPageEndInput(current)
                        }}
                      >
                        单页试跑（当前页）
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="xs"
                        onClick={() => {
                          setUsePageRange(false)
                          setPageStartInput("")
                          setPageEndInput("")
                        }}
                      >
                        整份处理
                      </Button>
                    </div>
                  </div>

                  {usePageRange ? (
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="grid gap-2">
                        <label className="font-mono text-xs uppercase tracking-[0.14em] text-muted-foreground">
                          起始页
                        </label>
                        <Input
                          inputMode="numeric"
                          placeholder="例如 1"
                          value={pageStartInput}
                          onChange={(e) => setPageStartInput(e.target.value)}
                        />
                      </div>
                      <div className="grid gap-2">
                        <label className="font-mono text-xs uppercase tracking-[0.14em] text-muted-foreground">
                          结束页
                        </label>
                        <Input
                          inputMode="numeric"
                          placeholder="例如 5"
                          value={pageEndInput}
                          onChange={(e) => setPageEndInput(e.target.value)}
                        />
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      当前将处理整份文档。若只想做快速验证，点击“单页试跑（当前页）”即可。
                    </p>
                  )}

                  <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3">
                    <div className="flex gap-2">
                      <Button type="button" onClick={handleConvert} disabled={!canStart}>
                        开始转换
                      </Button>
                      <Button type="button" variant="outline" onClick={handleResetAll}>
                        重置
                      </Button>
                    </div>
                    {jobId && !TERMINAL_STATUSES.has(currentStatus as JobStatusValue) ? (
                      <Button type="button" variant="destructive" onClick={handleCancelCurrentJob}>
                        <XCircleIcon className="size-4" />
                        取消当前任务
                      </Button>
                    ) : null}
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="border-border bg-muted/20">
              <CardHeader>
                <CardTitle className="text-lg">PDF 预览</CardTitle>
                <CardDescription>上传 PDF 后可在此预览并执行单页试跑。</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex min-h-[320px] items-center justify-center border border-dashed border-border bg-background px-4 text-sm text-muted-foreground">
                  暂无可预览文件
                </div>
              </CardContent>
            </Card>
          )}
        </main>

        <Card className="mt-6 hard-shadow-hover border-border">
          <CardHeader>
            <CardTitle className="text-lg">当前任务状态</CardTitle>
            <CardDescription>实时轮询后端状态，稳定且便于排查问题。</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={currentStatus === "failed" ? "destructive" : currentStatus === "completed" ? "secondary" : "outline"}>
                {jobStatusLabels[currentStatus as JobStatusValue] || currentStatus}
              </Badge>
              <Badge variant="outline">阶段：{currentStageLabel}</Badge>
              {jobId ? <Badge variant="outline">任务号：{jobId}</Badge> : null}
            </div>

            <Progress value={progressValue} className="h-2" />

            {(jobId || isSubmitting || activeJob) ? (
              <div className="grid gap-1 sm:grid-cols-2">
                {jobStageFlow.map((stage, index) => {
                  const isDone = stageFlowIndex >= index && stageFlowIndex >= 0
                  const isCurrent = currentStageCode === stage
                  return (
                    <div
                      key={stage}
                      className={cn(
                        "border px-2 py-1 text-[11px] font-mono uppercase tracking-[0.12em]",
                        isCurrent
                          ? "border-[#cc0000] bg-[#cc0000] text-[#f9f9f7]"
                          : isDone
                            ? "border-border bg-muted/60 text-foreground"
                            : "border-border/70 bg-background text-muted-foreground"
                      )}
                    >
                      {jobStageLabels[stage] || stage}
                    </div>
                  )
                })}
              </div>
            ) : null}

            <div className="text-sm text-muted-foreground">
              {activeJob?.message || (isSubmitting ? "任务已提交，正在等待状态更新…" : "尚未开始任务")}
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="border border-border bg-muted/30 p-2">队列总数：{queueSize}</div>
              <div className="border border-border bg-muted/30 p-2">执行中：{inFlightJobs}</div>
              <div className="border border-border bg-muted/30 p-2">已完成：{completedJobs}</div>
              <div className="border border-border bg-muted/30 p-2">失败：{failedJobs}</div>
            </div>

            {!jobId ? (
              <div className="border border-dashed border-border bg-muted/20 p-3 text-xs text-muted-foreground">
                暂无当前任务。上传 PDF 后点击“开始转换”，状态会在这里实时更新。
              </div>
            ) : null}

            <div className="border border-dashed border-border p-3 text-xs text-muted-foreground">
              历史任务请查看下方“最近任务”表格，避免信息重复。
            </div>

            {activeJob?.error?.message ? (
              <div className="border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
                {activeJob.error.message}
              </div>
            ) : null}

            {actionError ? (
              <div className="border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
                {actionError}
              </div>
            ) : null}

            {jobId && currentStatus === "completed" ? (
              <Button
                type="button"
                onClick={async () => {
                  try {
                    await handleDownload(jobId)
                  } catch (e) {
                    toast.error(normalizeFetchError(e, "下载失败"))
                  }
                }}
              >
                <DownloadIcon className="size-4" />
                下载 PPTX
              </Button>
            ) : null}
          </CardContent>
        </Card>

        <Card className="mt-6 border-border">
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle className="text-lg">最近任务</CardTitle>
                <CardDescription>队列总数：{queueSize} · 保留独立跟踪页用于深度排查</CardDescription>
              </div>
              <Button type="button" variant="outline" onClick={() => void fetchJobs(false)} disabled={jobsLoading}>
                刷新列表
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto border border-border">
              <table className="w-full min-w-[760px] text-sm">
                <thead className="bg-muted/40 text-left text-xs uppercase tracking-[0.08em] text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2">任务</th>
                    <th className="px-3 py-2">状态</th>
                    <th className="px-3 py-2">进度</th>
                    <th className="px-3 py-2">阶段</th>
                    <th className="px-3 py-2">时间</th>
                    <th className="px-3 py-2 text-right">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.length ? (
                    jobs.map((row) => {
                      const stageLabel = jobStageLabels[row.stage] || row.stage
                      const canCancel = row.status === "pending" || row.status === "processing"
                      const canDownload = row.status === "completed"
                      return (
                        <tr
                          key={row.job_id}
                          className={cn("border-t border-border", row.job_id === jobId && "bg-muted/35")}
                        >
                          <td className="px-3 py-2 font-mono text-xs">{row.job_id}</td>
                          <td className="px-3 py-2">
                            <Badge variant={row.status === "failed" ? "destructive" : row.status === "completed" ? "secondary" : "outline"}>
                              {jobStatusLabels[row.status]}
                            </Badge>
                          </td>
                          <td className="px-3 py-2">{Math.max(0, Math.min(100, row.progress || 0))}%</td>
                          <td className="px-3 py-2">{stageLabel}</td>
                          <td className="px-3 py-2 text-muted-foreground">{formatDateTime(row.created_at)}</td>
                          <td className="px-3 py-2">
                            <div className="flex justify-end gap-2">
                              <Button type="button" variant="ghost" asChild>
                                <Link href={`/tracking?job=${encodeURIComponent(row.job_id)}`}>跟踪</Link>
                              </Button>
                              {canDownload ? (
                                <Button
                                  type="button"
                                  variant="outline"
                                  onClick={async () => {
                                    try {
                                      await handleDownload(row.job_id)
                                    } catch (e) {
                                      toast.error(normalizeFetchError(e, "下载失败"))
                                    }
                                  }}
                                >
                                  下载
                                </Button>
                              ) : null}
                              {canCancel ? (
                                <Button
                                  type="button"
                                  variant="destructive"
                                  onClick={() => void handleCancelById(row.job_id)}
                                >
                                  取消
                                </Button>
                              ) : null}
                            </div>
                          </td>
                        </tr>
                      )
                    })
                  ) : (
                    <tr>
                      <td colSpan={6} className="px-3 py-8 text-center text-sm text-muted-foreground">
                        暂无任务记录
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
          <CardFooter className="border-t border-border text-xs text-muted-foreground">
            <FileTextIcon className="mr-2 size-4" />
            首页专注执行；高级参数与模型切换请在设置页管理。
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}
