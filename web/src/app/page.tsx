"use client"

import * as React from "react"
import Link from "next/link"
import {
  ArrowRightIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  DownloadIcon,
  FileTextIcon,
  Trash2Icon,
  UploadCloudIcon,
  XCircleIcon,
} from "lucide-react"
import { useDropzone } from "react-dropzone"
import { toast } from "sonner"

import { cn } from "@/lib/utils"
import { apiFetch, normalizeFetchError } from "@/lib/api"
import {
  defaultSettings,
  loadStoredSettings,
  type Settings,
} from "@/lib/settings"
import {
  createJobFormData,
  getOcrConfigSourceLabel,
  getRunModelLabel,
  getRunParseEngineLabel,
  resolveRunConfig,
  validateRunConfig,
} from "@/lib/run-config"
import {
  getJobStageFlowStage,
  getJobStageFlowIndex,
  JOB_STAGE_COMPACT_LABELS,
  JOB_STAGE_FLOW,
  JOB_STAGE_LABELS,
  JOB_STATUS_LABELS,
  normalizeJobListResponse,
  normalizeJobStatusResponse,
  TERMINAL_JOB_STATUSES,
  type JobListItem,
  type JobListResponse,
  type JobStatusResponse,
  type JobStatusValue,
} from "@/lib/job-status"
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
import { JobDebugPanel } from "@/components/job-debug-panel"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { PdfCanvasPreview } from "@/components/pdf-canvas-preview"
import { useUploadSession } from "@/components/upload-session-provider"

type JobApiErrorBody = {
  code?: string
  message?: string
} | null

type JobStatusFetchError = Error & {
  statusCode?: number
  errorCode?: string
}

const ocrProviderLabels: Record<Settings["ocrProvider"], string> = {
  auto: "自动",
  aiocr: "AI OCR",
  baidu: "百度 OCR",
  tesseract: "本地 OCR（Tesseract）",
  paddle_local: "本地 OCR（PaddleOCR）",
}

const layoutAssistModeLabels: Record<"off" | "on" | "auto", string> = {
  off: "关闭",
  on: "开启",
  auto: "自动",
}

const ocrLinebreakModeLabels: Record<Settings["ocrAiLinebreakAssistMode"], string> = {
  auto: "自动",
  on: "开启",
  off: "关闭",
}

const HOME_ACTIVE_JOB_STORAGE_KEY = "ppt-opencode:home:active-job-id"

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
  const [deletingJobId, setDeletingJobId] = React.useState<string | null>(null)
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
  const runParseEngineLabel = React.useMemo(
    () => getRunParseEngineLabel(runConfig),
    [runConfig]
  )
  const runModelLabel = React.useMemo(() => getRunModelLabel(runConfig), [runConfig])
  const runOcrConfigSourceLabel = React.useMemo(
    () => getOcrConfigSourceLabel(runConfig.ocrAiConfigSource),
    [runConfig.ocrAiConfigSource]
  )
  const runOcrSummaryLabel = React.useMemo(() => {
    if (runConfig.parseProvider === "mineru") {
      return "MinerU 自身"
    }
    if (runConfig.parseProvider === "baidu_doc") {
      return "百度解析"
    }
    return ocrProviderLabels[runConfig.effectiveOcrProvider]
  }, [runConfig])

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
      const normalized = normalizeJobListResponse(body)
      const rows = normalized.jobs
      setJobs(rows)
      setQueueSize(normalized.queueSize)

      const currentJobId = jobIdRef.current
      if (currentJobId) {
        const matched = rows.find((row) => row.job_id === currentJobId)
        if (matched) {
          setActiveJob((previous) => {
            const normalizedStatus = normalizeJobStatusResponse(matched)
            if (previous?.job_id === matched.job_id && previous.debug_events.length) {
              return {
                ...normalizedStatus,
                debug_events: previous.debug_events,
              }
            }
            return normalizedStatus
          })
          if (TERMINAL_JOB_STATUSES.has(matched.status)) {
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
    const body = (await response.json().catch(() => null)) as JobApiErrorBody
    if (!response.ok) {
      const err = new Error(
        body?.message || `查询任务状态失败（HTTP ${response.status}）`
      ) as JobStatusFetchError
      err.statusCode = response.status
      if (typeof body?.code === "string") {
        err.errorCode = body.code
      }
      throw err
    }
    if (!body || typeof body !== "object") {
      throw new Error("任务状态响应异常")
    }
    return normalizeJobStatusResponse(body)
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

    const validation = validateRunConfig(settingsSnapshot)
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
      const formData = createJobFormData(file, settingsSnapshot, pageStart, pageEnd)
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

  const handleDeleteById = React.useCallback(
    async (targetJobId: string) => {
      if (!window.confirm(`确定删除任务 ${targetJobId.slice(0, 8)} 吗？这会移除任务记录和本地产物。`)) {
        return
      }
      setDeletingJobId(targetJobId)
      try {
        const response = await apiFetch(`/jobs/${targetJobId}`, { method: "DELETE" })
        const body = await response.json().catch(() => null)
        if (!response.ok) {
          throw new Error(body?.message || `删除失败（HTTP ${response.status}）`)
        }
        if (jobIdRef.current === targetJobId) {
          setJobId(null)
          setActiveJob(null)
          setIsSubmitting(false)
          setActionError(null)
        }
        setJobs((prev) => prev.filter((row) => row.job_id !== targetJobId))
        toast.success("任务已删除")
        void fetchJobs(true)
      } catch (e) {
        toast.error(normalizeFetchError(e, "删除任务失败"))
      } finally {
        setDeletingJobId((current) => (current === targetJobId ? null : current))
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
        if (TERMINAL_JOB_STATUSES.has(status.status)) {
          setIsSubmitting(false)
          stopPolling()
          void fetchJobs(true)
        }
      } catch (error) {
        if (!mounted) return
        const pollingError = error as JobStatusFetchError
        const isJobNotFound =
          pollingError?.statusCode === 404 || pollingError?.errorCode === "JOB_NOT_FOUND"
        if (!isJobNotFound) {
          return
        }

        stopPolling()
        setIsSubmitting(false)
        setActiveJob(null)
        setJobId(null)
        setActionError("任务状态不存在或已过期，请重新提交任务")
        void fetchJobs(true)
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
    if (!TERMINAL_JOB_STATUSES.has(activeJob.status)) return

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
  const currentStatus: JobStatusValue | null = activeJob?.status || (isSubmitting ? "processing" : null)
  const currentStageCode = activeJob?.stage || (isSubmitting ? "queued" : "")
  const currentStageFlowCode = getJobStageFlowStage(currentStageCode)
  const currentStageLabel = activeJob?.stage
    ? (JOB_STAGE_LABELS[activeJob.stage] ?? activeJob.stage)
    : "等待开始"
  const stageFlowIndex = getJobStageFlowIndex(currentStageCode)
  const stageLiveText = activeJob
    ? `任务状态 ${JOB_STATUS_LABELS[currentStatus as JobStatusValue] || currentStatus}，阶段 ${currentStageLabel}，进度 ${progressValue}%`
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
  const currentStatusLabel = currentStatus ? JOB_STATUS_LABELS[currentStatus] || currentStatus : "空闲中"
  const hasCurrentJob = Boolean(jobId || isSubmitting || activeJob)

  return (
    <div className="min-h-dvh bg-background">
      <div className="mx-auto w-full max-w-screen-xl px-4 py-6 md:py-10">
        <header className="editorial-page-header newsprint-texture page-enter border border-border bg-background">
          <div className="px-5 py-5 md:px-6 md:py-6">
            <div className="flex flex-wrap items-center gap-2">
              <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
                {editionDate} · PDF 工作台
              </div>
              <Badge variant="outline" className="font-sans text-[11px] uppercase tracking-[0.12em]">
                轻量首页
              </Badge>
            </div>
            <h1 className="mt-3 max-w-4xl font-serif text-4xl leading-[0.92] tracking-tight md:text-6xl">
              PDF 处理工作台
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-muted-foreground md:text-[15px]">
              上传文件后直接预览、设定范围并开始处理。复杂参数放在设置页，结果核对放在跟踪页，首页只保留最常用的操作。
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              <Badge variant="outline">当前文件：{file ? file.name : "未选择"}</Badge>
              <Badge variant="outline">默认模型：{runModelLabel}</Badge>
              {runConfig.effectiveOcrProvider === "aiocr" ||
              (runConfig.layoutAssistMode !== "off" && runConfig.effectiveOcrAiKey) ? (
                <Badge variant="outline">AI 配置：{runOcrConfigSourceLabel}</Badge>
              ) : null}
              <Badge className="editorial-pill">{hasCurrentJob ? currentStatusLabel : "等待开始"}</Badge>
            </div>

            <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground">
              <span>解析：{runParseEngineLabel}</span>
              <span>OCR：{runOcrSummaryLabel}</span>
              <span>队列：{queueSize}</span>
              <span>执行中：{inFlightJobs}</span>
            </div>
          </div>
        </header>

        <p className="sr-only" role="status" aria-live="polite">
          {stageLiveText}
        </p>

        <main className="mt-6 grid gap-5 xl:grid-cols-[minmax(0,1fr)_280px] xl:items-stretch">
          <Card className="home-card page-enter page-enter-delay-1 border-border xl:h-full">
            <CardHeader className="pb-3">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <div className="home-section-kicker">开始处理</div>
                  <CardTitle className="mt-2 text-[1.3rem]">上传与预览</CardTitle>
                  <CardDescription className="mt-1 max-w-xl text-sm leading-6">
                    选择 PDF 后可直接预览当前页，并决定是整份处理还是先做单页验证。
                  </CardDescription>
                </div>
                <Badge variant="outline">支持整份与单页试跑</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
                <div
                  {...getRootProps()}
                  className={cn(
                    "home-dropzone cursor-pointer text-center",
                    isDragActive && !isDragReject && "bg-accent/50",
                    isDragReject && "border-destructive bg-destructive/10",
                    isSubmitting && "pointer-events-none opacity-60"
                  )}
                >
                  <input {...getInputProps()} />
                  <UploadCloudIcon className="mx-auto size-8 text-muted-foreground" />
                  <p className="mt-3 text-sm font-medium">
                    {isDragActive ? "松开以上传 PDF" : "拖拽 PDF 到这里，或点击选择文件"}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">仅支持 .pdf</p>
                </div>

                {file ? (
                  <div className="home-inline-panel flex items-center justify-between gap-3 px-4 py-3">
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

                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="home-section-kicker">文档预览</div>
                    <div className="mt-1 text-sm text-muted-foreground">预览页与单页试跑始终保持一致。</div>
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

                {filePreviewUrl ? (
                  <div className="home-preview-stage">
                    <PdfCanvasPreview
                      fileUrl={filePreviewUrl}
                      page={previewPage}
                      className="w-full"
                      onPageCountChange={handlePreviewPageCountChange}
                    />
                  </div>
                ) : (
                  <div className="home-preview-stage home-preview-empty">上传 PDF 后会在这里显示预览</div>
                )}

                <div className="home-inline-panel grid gap-3 px-4 py-3">
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
                    <p className="text-xs leading-6 text-muted-foreground">
                      当前将处理整份文档。如果只是想快速确认效果，直接点击“单页试跑（当前页）”即可。
                    </p>
                  )}

                  <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/70 pt-3">
                    <div className="flex gap-2">
                      <Button type="button" onClick={handleConvert} disabled={!canStart}>
                        开始转换
                      </Button>
                      <Button type="button" variant="outline" onClick={handleResetAll}>
                        重置
                      </Button>
                    </div>
                    {jobId && currentStatus && !TERMINAL_JOB_STATUSES.has(currentStatus) ? (
                      <Button type="button" variant="destructive" onClick={handleCancelCurrentJob}>
                        <XCircleIcon className="size-4" />
                        取消当前任务
                      </Button>
                    ) : null}
                  </div>
                </div>
              </CardContent>
            </Card>

          <div className="grid min-w-0 gap-5 xl:h-full xl:grid-rows-[auto_minmax(0,1fr)]">
            <Card className="home-card-muted page-enter page-enter-delay-2 min-w-0 border border-border">
              <CardHeader className="pb-3">
                <div className="home-section-kicker">当前配置</div>
                <CardTitle className="mt-2 text-xl">处理参数</CardTitle>
                <CardDescription className="mt-1 text-sm leading-6">
                  当前页面只展示必要配置，细调项仍在设置页统一管理。
                </CardDescription>
              </CardHeader>
              <CardContent className="min-w-0 space-y-3">
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <Badge variant="outline">解析：{runParseEngineLabel}</Badge>
                  <Badge variant="outline">OCR：{runOcrSummaryLabel}</Badge>
                  <Badge variant="outline">
                    版式辅助：{layoutAssistModeLabels[runConfig.layoutAssistMode]}
                  </Badge>
                  {runConfig.effectiveOcrProvider === "aiocr" ? (
                    <Badge variant="outline">
                      行拆分：{ocrLinebreakModeLabels[runConfig.ocrLinebreakAssistMode]}
                    </Badge>
                  ) : null}
                  {runConfig.effectiveOcrProvider === "aiocr" ? (
                    <Badge variant="outline">AI 配置：{runOcrConfigSourceLabel}</Badge>
                  ) : null}
                  <Badge variant="outline" className="max-w-full whitespace-normal break-all">
                    模型：{runModelLabel}
                  </Badge>
                </div>
                <Button type="button" variant="ghost" asChild>
                  <Link href="/settings">
                    前往设置页
                    <ArrowRightIcon className="size-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card className="home-card-muted page-enter page-enter-delay-2 min-w-0 border border-border xl:flex xl:min-h-0 xl:flex-col">
              <CardHeader className="pb-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="home-section-kicker">任务状态</div>
                    <CardTitle className="mt-2 text-xl">当前任务</CardTitle>
                    <CardDescription className="mt-1 text-sm leading-6">
                      处理开始后，这里会持续刷新状态和进度。
                    </CardDescription>
                  </div>
                  <Badge
                    variant={currentStatus === "failed" ? "destructive" : currentStatus === "completed" ? "secondary" : "outline"}
                  >
                    {currentStatusLabel}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="min-w-0 space-y-3.5 xl:min-h-0 xl:flex-1 xl:overflow-y-auto">
                <div className="flex min-w-0 flex-wrap items-center gap-2">
                  <Badge variant="outline">阶段：{currentStageLabel}</Badge>
                  {jobId ? (
                    <Badge variant="outline" className="max-w-full whitespace-normal break-all">
                      任务号：{jobId}
                    </Badge>
                  ) : null}
                </div>

                <Progress value={progressValue} className="h-2" />

                {hasCurrentJob ? (
                  <div className="pb-1">
                    <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-4 xl:grid-cols-4">
                      {JOB_STAGE_FLOW.map((stage, index) => {
                        const isDone = stageFlowIndex >= index && stageFlowIndex >= 0
                        const isCurrent = currentStageFlowCode === stage
                        return (
                          <div
                            key={stage}
                            title={JOB_STAGE_LABELS[stage] || stage}
                            aria-label={JOB_STAGE_LABELS[stage] || stage}
                            className={cn(
                              "min-w-0 border px-2 py-1 text-center font-sans text-[11px] font-medium leading-tight tracking-[0.02em] transition-colors",
                              isCurrent
                                ? "border-[#cc0000] bg-[#cc0000] text-[#f9f9f7]"
                                : isDone
                                  ? "border-border bg-muted/60 text-foreground"
                                  : "border-border/70 bg-background/70 text-muted-foreground"
                            )}
                          >
                            {JOB_STAGE_COMPACT_LABELS[stage] || JOB_STAGE_LABELS[stage] || stage}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ) : null}

                <div className="text-sm leading-6 text-muted-foreground">
                  {activeJob?.message || (isSubmitting ? "任务已提交，正在等待状态更新…" : "尚未开始任务")}
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="home-stat-cell">
                    <div className="home-stat-label">队列总数</div>
                    <div className="home-stat-value">{queueSize}</div>
                  </div>
                  <div className="home-stat-cell">
                    <div className="home-stat-label">执行中</div>
                    <div className="home-stat-value">{inFlightJobs}</div>
                  </div>
                  <div className="home-stat-cell">
                    <div className="home-stat-label">已完成</div>
                    <div className="home-stat-value">{completedJobs}</div>
                  </div>
                  <div className="home-stat-cell">
                    <div className="home-stat-label">失败</div>
                    <div className="home-stat-value">{failedJobs}</div>
                  </div>
                </div>

                {!jobId ? (
                  <div className="home-note">
                    暂无当前任务。上传 PDF 后点击“开始转换”，这里会自动显示最新进度。
                  </div>
                ) : null}

                {activeJob?.error?.message ? (
                  <div className="border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
                    {activeJob.error.message}
                  </div>
                ) : null}

                {hasCurrentJob ? (
                  <JobDebugPanel
                    events={activeJob?.debug_events || []}
                    title="任务调试日志"
                    emptyLabel="任务启动后会在这里持续追加后端调试信息"
                    compact
                  />
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
          </div>
        </main>

        <Card className="home-card page-enter page-enter-delay-2 mt-6 border-border">
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="home-section-kicker">任务记录</div>
                <CardTitle className="mt-2 text-[1.3rem]">最近任务</CardTitle>
                <CardDescription className="mt-1 text-sm leading-6">
                  近期任务会保留在这里，需要更细的核对时再进入跟踪页。
                </CardDescription>
              </div>
              <Button type="button" variant="outline" onClick={() => void fetchJobs(false)} disabled={jobsLoading}>
                刷新列表
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="home-table-shell overflow-x-auto border border-border">
              <table className="w-full min-w-[760px] text-sm">
                <thead className="bg-muted/25 text-left text-xs uppercase tracking-[0.08em] text-muted-foreground">
                  <tr>
                    <th className="px-4 py-3">任务</th>
                    <th className="px-4 py-3">状态</th>
                    <th className="px-4 py-3">进度</th>
                    <th className="px-4 py-3">阶段</th>
                    <th className="px-4 py-3">时间</th>
                    <th className="px-4 py-3 text-right">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.length ? (
                    jobs.map((row) => {
                      const stageLabel = JOB_STAGE_LABELS[row.stage] || row.stage
                      const canCancel = row.status === "pending" || row.status === "processing"
                      const canDownload = row.status === "completed"
                      const canDelete =
                        row.status === "completed" ||
                        row.status === "failed" ||
                        row.status === "cancelled"
                      return (
                        <tr
                          key={row.job_id}
                          className={cn(
                            "border-t border-border/80 transition-colors hover:bg-muted/20",
                            row.job_id === jobId && "bg-muted/35"
                          )}
                        >
                          <td className="px-4 py-3 font-mono text-xs">{row.job_id}</td>
                          <td className="px-4 py-3">
                            <Badge variant={row.status === "failed" ? "destructive" : row.status === "completed" ? "secondary" : "outline"}>
                              {JOB_STATUS_LABELS[row.status]}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">{Math.max(0, Math.min(100, row.progress || 0))}%</td>
                          <td className="px-4 py-3">{stageLabel}</td>
                          <td className="px-4 py-3 text-muted-foreground">{formatDateTime(row.created_at)}</td>
                          <td className="px-4 py-3">
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
                              {canDelete ? (
                                <Button
                                  type="button"
                                  variant="outline"
                                  disabled={deletingJobId === row.job_id}
                                  onClick={() => void handleDeleteById(row.job_id)}
                                >
                                  <Trash2Icon className="size-4" />
                                  {deletingJobId === row.job_id ? "删除中..." : "删除"}
                                </Button>
                              ) : null}
                            </div>
                          </td>
                        </tr>
                      )
                    })
                  ) : (
                    <tr>
                      <td colSpan={6} className="px-4 py-10 text-center text-sm text-muted-foreground">
                        暂无任务记录
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
          <CardFooter className="border-t border-border/80 text-xs text-muted-foreground">
            <FileTextIcon className="mr-2 size-4" />
            首页保持轻量入口；需要更细的参数管理或结果核查时，再进入对应页面继续处理。
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}
