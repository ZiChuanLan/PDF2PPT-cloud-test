"use client"

import * as React from "react"
import Image, { type ImageLoader } from "next/image"
import { useRouter } from "next/navigation"
import { useSearchParams } from "next/navigation"
import { toast } from "sonner"

import {
  JOB_STAGE_LABELS,
  JOB_STATUS_LABELS,
  normalizeJobListResponse,
  normalizeJobStatusResponse,
  TERMINAL_JOB_STATUSES,
  QUEUE_STATE_LABELS,
  type JobListItem,
  type JobListResponse,
  type JobStatusResponse,
  type JobStatusValue,
} from "@/lib/job-status"
import {
  formatArtifactPageLabel,
  getArtifactPageIndex,
  normalizeArtifactPages,
  resolveActiveArtifactPage,
} from "@/lib/tracking-artifacts"
import { cn } from "@/lib/utils"
import { apiFetch, normalizeFetchError, resolveApiOrigin } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { JobDebugPanel } from "@/components/job-debug-panel"
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/select"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

type JobArtifactImage = {
  page_index: number
  path: string
  url: string
}

type JobArtifactsResponse = {
  job_id: string
  status?: JobStatusValue | null
  artifacts_retained: boolean
  source_pdf_url?: string | null
  original_images: JobArtifactImage[]
  cleaned_images: JobArtifactImage[]
  final_preview_images: JobArtifactImage[]
  ocr_overlay_images: JobArtifactImage[]
  layout_before_images: JobArtifactImage[]
  layout_after_images: JobArtifactImage[]
  available_pages: number[]
}

const jobStatusFilterOptions: Array<{ value: "all" | JobStatusValue; label: string }> = [
  { value: "all", label: "全部状态" },
  { value: "pending", label: "排队中" },
  { value: "processing", label: "处理中" },
  { value: "completed", label: "已完成" },
  { value: "failed", label: "失败" },
  { value: "cancelled", label: "已取消" },
]

const passthroughImageLoader: ImageLoader = ({ src }) => src

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

function getStatusBadgeClass(status: JobStatusValue) {
  if (status === "completed") return "status-badge status-badge-success"
  if (status === "failed") return "status-badge status-badge-error"
  if (status === "cancelled") return "status-badge status-badge-neutral"
  return "status-badge status-badge-warning"
}

function TrackingArtifactImage({
  src,
  alt,
  className,
  priority = false,
}: {
  src: string
  alt: string
  className?: string
  priority?: boolean
}) {
  return (
    <Image
      loader={passthroughImageLoader}
      unoptimized
      src={src}
      alt={alt}
      fill
      priority={priority}
      sizes="(min-width: 1280px) 720px, (min-width: 1024px) 50vw, 100vw"
      className={className}
    />
  )
}

function TrackingPageContent() {
  const [apiOrigin, setApiOrigin] = React.useState("")
  const searchParams = useSearchParams()
  const requestedJobId = (searchParams.get("job") || "").trim()

  const [jobRecords, setJobRecords] = React.useState<JobListItem[]>([])
  const [queueSize, setQueueSize] = React.useState(0)
  const [isJobsLoading, setIsJobsLoading] = React.useState(false)
  const [jobsError, setJobsError] = React.useState<string | null>(null)
  const [deletingJobId, setDeletingJobId] = React.useState<string | null>(null)

  const [trackedArtifacts, setTrackedArtifacts] = React.useState<JobArtifactsResponse | null>(
    null
  )
  const [trackedJobId, setTrackedJobId] = React.useState<string | null>(requestedJobId || null)
  const [trackedJobStatus, setTrackedJobStatus] = React.useState<JobStatusResponse | null>(null)
  const [trackedArtifactsLoading, setTrackedArtifactsLoading] = React.useState(false)
  const [trackedArtifactsError, setTrackedArtifactsError] = React.useState<string | null>(null)
  const [trackedJobStatusError, setTrackedJobStatusError] = React.useState<string | null>(null)
  const [trackedArtifactsPage, setTrackedArtifactsPage] = React.useState<number | null>(null)
  const [trackingMenu, setTrackingMenu] = React.useState<"frames" | "compare">("compare")
  const [compareSplitRatio, setCompareSplitRatio] = React.useState(0.5)
  const [showInlinePdf, setShowInlinePdf] = React.useState(false)
  const [jobKeyword, setJobKeyword] = React.useState("")
  const [statusFilter, setStatusFilter] = React.useState<"all" | JobStatusValue>("all")
  const router = useRouter()

  const currentTrackedJob = React.useMemo(() => {
    if (!trackedJobId) return null
    return jobRecords.find((r) => r.job_id === trackedJobId) || null
  }, [jobRecords, trackedJobId])

  const trackedJobDetail = React.useMemo<
    (JobStatusResponse & {
      queue_position?: number | null
      queue_state?: string | null
    }) | null
  >(() => {
    if (!currentTrackedJob && !trackedJobStatus) return null
    if (!currentTrackedJob && trackedJobStatus) {
      return {
        ...trackedJobStatus,
        queue_position: null,
        queue_state: null,
      }
    }
    if (currentTrackedJob && !trackedJobStatus) {
      return {
        ...normalizeJobStatusResponse(currentTrackedJob),
        queue_position: currentTrackedJob.queue_position,
        queue_state: currentTrackedJob.queue_state,
      }
    }
    if (!currentTrackedJob || !trackedJobStatus) return null
    return {
      ...trackedJobStatus,
      queue_position: currentTrackedJob.queue_position,
      queue_state: currentTrackedJob.queue_state,
    }
  }, [currentTrackedJob, trackedJobStatus])

  const filteredJobRecords = React.useMemo(() => {
    const keyword = jobKeyword.trim().toLowerCase()
    return jobRecords.filter((record) => {
      if (statusFilter !== "all" && record.status !== statusFilter) return false
      if (!keyword) return true
      return (
        record.job_id.toLowerCase().includes(keyword) ||
        (record.stage || "").toLowerCase().includes(keyword)
      )
    })
  }, [jobKeyword, jobRecords, statusFilter])

  const fetchJobs = React.useCallback(async (silent = true) => {
    if (!silent) setIsJobsLoading(true)
    try {
      const response = await apiFetch("/jobs?limit=60")
      if (!response.ok) {
        throw new Error("加载任务记录失败")
      }
      const body = (await response.json().catch(() => null)) as JobListResponse | null
      const normalized = normalizeJobListResponse(body)
      setJobRecords(normalized.jobs)
      setQueueSize(normalized.queueSize)
      setJobsError(null)
    } catch (e) {
      if (!silent) {
        setJobsError(normalizeFetchError(e, "加载任务记录失败"))
      }
    } finally {
      if (!silent) setIsJobsLoading(false)
    }
  }, [])

  const fetchJobArtifacts = React.useCallback(async (targetJobId: string) => {
    setTrackedJobId(targetJobId)
    setTrackedJobStatus(null)
    setTrackedJobStatusError(null)
    setTrackedArtifactsLoading(true)
    setTrackedArtifactsError(null)
    setShowInlinePdf(false)
    try {
      const response = await apiFetch(`/jobs/${targetJobId}/artifacts`)
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.message || "加载任务产物失败")
      }
      const body = (await response.json().catch(() => null)) as JobArtifactsResponse | null
      if (!body || typeof body !== "object") {
        throw new Error("加载任务产物失败")
      }
      setTrackedArtifacts({
        job_id: body.job_id,
        status: body.status ?? null,
        artifacts_retained: Boolean(body.artifacts_retained),
        source_pdf_url: body.source_pdf_url ?? null,
        original_images: Array.isArray(body.original_images) ? body.original_images : [],
        cleaned_images: Array.isArray(body.cleaned_images) ? body.cleaned_images : [],
        final_preview_images: Array.isArray(body.final_preview_images)
          ? body.final_preview_images
          : [],
        ocr_overlay_images: Array.isArray(body.ocr_overlay_images)
          ? body.ocr_overlay_images
          : [],
        layout_before_images: Array.isArray(body.layout_before_images)
          ? body.layout_before_images
          : [],
        layout_after_images: Array.isArray(body.layout_after_images)
          ? body.layout_after_images
          : [],
        available_pages: normalizeArtifactPages(body.available_pages),
      })
    } catch (e) {
      setTrackedArtifacts(null)
      setTrackedArtifactsError(normalizeFetchError(e, "加载任务产物失败"))
    } finally {
      setTrackedArtifactsLoading(false)
    }
  }, [])

  const fetchTrackedJobStatus = React.useCallback(async (targetJobId: string) => {
    const response = await apiFetch(`/jobs/${targetJobId}`)
    const body = await response.json().catch(() => null)
    if (!response.ok) {
      throw new Error(body?.message || `加载任务状态失败（HTTP ${response.status}）`)
    }
    if (!body || typeof body !== "object") {
      throw new Error("加载任务状态失败")
    }
    return normalizeJobStatusResponse(body)
  }, [])

  const handleDownloadByJobId = React.useCallback(async (targetJobId: string) => {
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

  const handleDeleteJobById = React.useCallback(
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
        if (trackedJobId === targetJobId) {
          setTrackedArtifacts(null)
          setTrackedJobId(null)
          setTrackedJobStatus(null)
          setTrackedJobStatusError(null)
          setTrackedArtifactsError(null)
          setTrackedArtifactsPage(null)
          setShowInlinePdf(false)
          router.replace("/tracking", { scroll: false })
        }
        setJobRecords((prev) => prev.filter((row) => row.job_id !== targetJobId))
        toast.success("任务已删除")
        void fetchJobs(true)
      } catch (e) {
        toast.error(normalizeFetchError(e, "删除任务失败"))
      } finally {
        setDeletingJobId((current) => (current === targetJobId ? null : current))
      }
    },
    [fetchJobs, router, trackedJobId]
  )

  React.useEffect(() => {
    let mounted = true
    void resolveApiOrigin()
      .then((origin) => {
        if (mounted) setApiOrigin(origin)
      })
      .catch(() => {})
    return () => {
      mounted = false
    }
  }, [])

  React.useEffect(() => {
    void fetchJobs(false)
    const timer = window.setInterval(() => {
      void fetchJobs(true)
    }, 3000)
    return () => window.clearInterval(timer)
  }, [fetchJobs])

  React.useEffect(() => {
    if (!requestedJobId) return
    setTrackedJobId(requestedJobId)
    void fetchJobArtifacts(requestedJobId)
  }, [fetchJobArtifacts, requestedJobId])

  React.useEffect(() => {
    if (!trackedJobId) {
      setTrackedJobStatus(null)
      setTrackedJobStatusError(null)
      return
    }

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
        const status = await fetchTrackedJobStatus(trackedJobId)
        if (!mounted) return
        setTrackedJobStatus(status)
        setTrackedJobStatusError(null)
        if (TERMINAL_JOB_STATUSES.has(status.status)) {
          stopPolling()
        }
      } catch (e) {
        if (!mounted) return
        setTrackedJobStatus(null)
        setTrackedJobStatusError(normalizeFetchError(e, "加载任务状态失败"))
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
  }, [fetchTrackedJobStatus, trackedJobId])

  React.useEffect(() => {
    const pages = trackedArtifacts?.available_pages || []
    if (!pages.length) {
      setTrackedArtifactsPage(null)
      return
    }
    if (trackedArtifactsPage === null || !pages.includes(trackedArtifactsPage)) {
      setTrackedArtifactsPage(pages[0] ?? null)
    }
  }, [trackedArtifacts, trackedArtifactsPage])

  const findArtifactByPage = React.useCallback(
    (images: JobArtifactImage[] | undefined, page: number | null) => {
      if (page === null) return null
      if (!Array.isArray(images) || !images.length) return null
      return images.find((item) => item.page_index === page) || null
    },
    []
  )

  const updateCompareSplitRatio = React.useCallback((clientX: number, rect: DOMRect) => {
    if (rect.width <= 0) return
    const ratio = (clientX - rect.left) / rect.width
    setCompareSplitRatio(Math.max(0, Math.min(1, ratio)))
  }, [])

  const handleComparePointerMove = React.useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      updateCompareSplitRatio(event.clientX, event.currentTarget.getBoundingClientRect())
    },
    [updateCompareSplitRatio]
  )

  const handleCompareTouchMove = React.useCallback(
    (event: React.TouchEvent<HTMLDivElement>) => {
      const touch = event.touches[0]
      if (!touch) return
      updateCompareSplitRatio(touch.clientX, event.currentTarget.getBoundingClientRect())
    },
    [updateCompareSplitRatio]
  )

  const trackedPages = trackedArtifacts?.available_pages || []
  const sourcePdfAbsoluteUrl = trackedArtifacts?.source_pdf_url
    ? `${apiOrigin}${trackedArtifacts.source_pdf_url}`
    : null
  const activeTrackedPage = resolveActiveArtifactPage(trackedPages, trackedArtifactsPage)
  const activeTrackedPageIndex = getArtifactPageIndex(trackedPages, activeTrackedPage)
  const activeTrackedPageLabel = formatArtifactPageLabel(activeTrackedPage)
  const trackedOriginal = findArtifactByPage(trackedArtifacts?.original_images, activeTrackedPage)
  const trackedClean = findArtifactByPage(trackedArtifacts?.cleaned_images, activeTrackedPage)
  const trackedFinalPreview = findArtifactByPage(
    trackedArtifacts?.final_preview_images,
    activeTrackedPage
  )
  const trackedOcrOverlay = findArtifactByPage(
    trackedArtifacts?.ocr_overlay_images,
    activeTrackedPage
  )
  const trackedLayoutBefore = findArtifactByPage(
    trackedArtifacts?.layout_before_images,
    activeTrackedPage
  )
  const trackedLayoutAfter = findArtifactByPage(
    trackedArtifacts?.layout_after_images,
    activeTrackedPage
  )
  const trackedBeforeOverlay = trackedLayoutBefore || trackedOcrOverlay
  const trackedAfterOverlay = trackedFinalPreview || trackedLayoutAfter || trackedClean || null
  const trackedCompareBefore = trackedOriginal || trackedLayoutBefore || null
  const trackedCompareAfter =
    trackedFinalPreview || trackedClean || trackedLayoutAfter || trackedOriginal || null
  const trackedCompareBase = trackedCompareBefore || trackedCompareAfter
  const compareSplitPercent = Math.round(compareSplitRatio * 100)
  const hasTrackedVisualArtifacts = trackedPages.length > 0

  return (
    <div className="min-h-dvh bg-background">
      <div className="mx-auto w-full max-w-screen-xl px-4 py-6 md:py-10">
        <header className="editorial-page-header newsprint-texture page-enter border border-border bg-background p-5 md:p-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0 space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <div className="font-mono text-xs uppercase tracking-[0.16em] text-muted-foreground">
                  任务追踪
                </div>
                <Badge variant="outline" className="font-sans text-[11px] uppercase tracking-[0.12em]">
                  结果校验
                </Badge>
              </div>
              <div>
                <div className="font-mono text-xs uppercase tracking-[0.16em] text-muted-foreground">
                  任务记录 / 页面校验 / PDF 回看
                </div>
                <h1 className="mt-2 font-serif text-4xl leading-[0.95] md:text-5xl">
                  任务追踪与结果对比
                </h1>
                <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted-foreground md:text-base">
                  在这里查看任务进度、处理结果和逐页对比，方便快速确认输出是否符合预期。
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">任务详情</Badge>
              <Badge className="editorial-pill">页面对比</Badge>
              <Badge variant="outline">与首页联动</Badge>
            </div>
          </div>
        </header>

        <div className="mt-4 grid gap-4 xl:grid-cols-[23rem_minmax(0,1fr)]">
          <Card className="page-enter page-enter-delay-1 py-0 hard-shadow-hover">
            <CardHeader className="border-b border-border pt-5 md:pt-6">
              <div className="flex items-center justify-between gap-2">
                <CardTitle>任务列表</CardTitle>
                <Badge variant="outline">排队 {queueSize}</Badge>
              </div>
              <CardDescription>可按状态或任务号筛选，选中后在右侧查看结果详情。</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 py-4">
              <div className="grid gap-2 sm:grid-cols-[1fr_150px]">
                <Input
                  value={jobKeyword}
                  onChange={(e) => setJobKeyword(e.target.value)}
                  placeholder="搜索任务号 / 阶段代码"
                />
                <Select
                  value={statusFilter}
                  onChange={(e) =>
                    setStatusFilter((e.target.value || "all") as "all" | JobStatusValue)
                  }
                >
                  {jobStatusFilterOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                当前显示 {filteredJobRecords.length} / {jobRecords.length}
              </div>
              {jobsError && !jobRecords.length ? (
                <div className="text-xs text-destructive">{jobsError}</div>
              ) : null}
              {isJobsLoading && !jobRecords.length ? (
                <div className="text-xs text-muted-foreground">加载中...</div>
              ) : null}
              {filteredJobRecords.length ? (
                <div className="max-h-[70vh] overflow-y-auto border border-border/80 bg-background/80">
                  {filteredJobRecords.map((record) => {
                    const isCurrent = record.job_id === trackedJobId
                    const statusLabel = JOB_STATUS_LABELS[record.status] || record.status
                    const stageLabel = JOB_STAGE_LABELS[record.stage] || record.stage
                    const canDelete =
                      record.status === "completed" ||
                      record.status === "failed" ||
                      record.status === "cancelled"
                    const detailMessage =
                      (record.status === "failed" &&
                        typeof record.error?.message === "string" &&
                        record.error.message.trim()) ||
                      (typeof record.message === "string" && record.message.trim()) ||
                      null
                    return (
                      <div
                        key={record.job_id}
                        className={cn(
                          "border-b px-3 py-2 transition-colors duration-200 last:border-b-0",
                          isCurrent
                            ? "bg-secondary/80 shadow-[inset_4px_0_0_0_#111111]"
                            : "hover:bg-muted/40"
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="font-mono text-[11px] uppercase tracking-[0.14em]">
                            {record.job_id.slice(0, 8)}
                          </div>
                          <Badge className={cn("border-0", getStatusBadgeClass(record.status))}>
                            {statusLabel}
                          </Badge>
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {stageLabel} · {record.progress}%
                        </div>
                        <div className="mt-1 font-mono text-[11px] text-muted-foreground">
                          阶段代码：{record.stage}
                        </div>
                        {record.queue_state === "queued" &&
                        typeof record.queue_position === "number" ? (
                          <div className="mt-1 font-mono text-[11px] text-muted-foreground">
                            排队位置：第 {record.queue_position} 位
                          </div>
                        ) : record.queue_state ? (
                          <div className="mt-1 font-mono text-[11px] text-muted-foreground">
                            队列状态：{QUEUE_STATE_LABELS[record.queue_state] || record.queue_state}
                          </div>
                        ) : null}
                        {detailMessage ? (
                          <div className="mt-1 text-xs text-muted-foreground line-clamp-2">
                            {detailMessage}
                          </div>
                        ) : null}
                        <div className="mt-1 font-mono text-[11px] text-muted-foreground">
                          {formatDateTime(record.created_at)}
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Button
                            type="button"
                            size="xs"
                            variant="outline"
                            onClick={() => void fetchJobArtifacts(record.job_id)}
                          >
                            跟踪
                          </Button>
                          {record.status === "completed" ? (
                            <Button
                              type="button"
                              size="xs"
                              variant="outline"
                              onClick={() => {
                                void handleDownloadByJobId(record.job_id).catch((e) => {
                                  toast.error(normalizeFetchError(e, "下载失败"))
                                })
                              }}
                            >
                              下载
                            </Button>
                          ) : null}
                          {canDelete ? (
                            <Button
                              type="button"
                              size="xs"
                              variant="outline"
                              disabled={deletingJobId === record.job_id}
                              onClick={() => void handleDeleteJobById(record.job_id)}
                            >
                              {deletingJobId === record.job_id ? "删除中..." : "删除"}
                            </Button>
                          ) : null}
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="border border-dashed border-border bg-muted/10 px-3 py-5 text-sm text-muted-foreground">
                  没有匹配的任务记录，请调整筛选条件。
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="page-enter page-enter-delay-2 py-0 hard-shadow-hover">
            <CardHeader className="border-b border-border pt-5 md:pt-6">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <CardTitle>结果预览</CardTitle>
                  <CardDescription>支持逐页查看，也可切换前后对比。</CardDescription>
                </div>
                <Badge variant="outline">
                  {trackedJobId ? trackedJobId.slice(0, 8) : "未选择任务"}
                </Badge>
              </div>
              {hasTrackedVisualArtifacts ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    type="button"
                    size="xs"
                    variant={trackingMenu === "frames" ? "default" : "outline"}
                    onClick={() => setTrackingMenu("frames")}
                  >
                    逐页预览
                  </Button>
                  <Button
                    type="button"
                    size="xs"
                    variant={trackingMenu === "compare" ? "default" : "outline"}
                    onClick={() => setTrackingMenu("compare")}
                  >
                    前后对比
                  </Button>
                </div>
              ) : null}
              {trackedJobDetail ? (
                <div className="mt-3 grid gap-1 border border-border bg-muted/40 px-3 py-2">
                  <div className="text-xs text-muted-foreground">
                    {JOB_STAGE_LABELS[trackedJobDetail.stage] || trackedJobDetail.stage} ·{" "}
                    {trackedJobDetail.progress}%
                  </div>
                  {(trackedJobDetail.status === "failed" &&
                    typeof trackedJobDetail.error?.message === "string" &&
                    trackedJobDetail.error.message.trim()) ||
                  (trackedJobDetail.message && trackedJobDetail.message.trim()) ? (
                    <div className="text-xs text-muted-foreground">
                      {(trackedJobDetail.status === "failed" &&
                        typeof trackedJobDetail.error?.message === "string" &&
                        trackedJobDetail.error.message.trim()) ||
                        trackedJobDetail.message}
                    </div>
                  ) : null}
                  <div className="font-mono text-[11px] text-muted-foreground">
                    阶段代码：{trackedJobDetail.stage}
                  </div>
                  {trackedJobDetail.queue_state === "queued" &&
                  typeof trackedJobDetail.queue_position === "number" ? (
                    <div className="font-mono text-[11px] text-muted-foreground">
                      排队位置：第 {trackedJobDetail.queue_position} 位
                    </div>
                  ) : trackedJobDetail.queue_state ? (
                    <div className="font-mono text-[11px] text-muted-foreground">
                      队列状态：
                      {QUEUE_STATE_LABELS[trackedJobDetail.queue_state] || trackedJobDetail.queue_state}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </CardHeader>
            <CardContent className="grid gap-4 py-5">
              {trackedJobId ? (
                <JobDebugPanel
                  events={trackedJobStatus?.debug_events || []}
                  title="任务调试日志"
                  emptyLabel="选中任务后会在这里显示后端逐行调试信息"
                  compact
                />
              ) : null}
              {trackedJobStatusError ? (
                <div className="border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  {trackedJobStatusError}
                </div>
              ) : null}
              {trackedArtifactsError ? (
                <div className="border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  {trackedArtifactsError}
                </div>
              ) : null}
              {trackedArtifactsLoading ? (
                <div className="font-mono text-xs text-muted-foreground">正在加载任务产物...</div>
              ) : null}
              {!trackedArtifactsLoading && !trackedArtifacts?.job_id ? (
                <div className="border bg-muted px-3 py-3 text-sm text-muted-foreground">
                  在左侧列表选择任务后，右侧展示该任务的页面对比。
                </div>
              ) : null}

              {trackedArtifacts?.job_id ? (
                <>
                  {trackedPages.length || sourcePdfAbsoluteUrl ? (
                    <div className="flex flex-wrap items-center gap-2">
                      {trackedPages.length ? (
                        <>
                          <div className="font-mono text-xs text-muted-foreground">
                            第 {activeTrackedPageLabel} 页 / 共 {trackedPages.length} 页
                          </div>
                          <Button
                            type="button"
                            size="xs"
                            variant="outline"
                            onClick={() => {
                              if (activeTrackedPageIndex > 0) {
                                setTrackedArtifactsPage(trackedPages[activeTrackedPageIndex - 1] ?? null)
                              }
                            }}
                            disabled={activeTrackedPageIndex <= 0}
                          >
                            上一页
                          </Button>
                          <Button
                            type="button"
                            size="xs"
                            variant="outline"
                            onClick={() => {
                              if (
                                activeTrackedPageIndex >= 0 &&
                                activeTrackedPageIndex < trackedPages.length - 1
                              ) {
                                setTrackedArtifactsPage(
                                  trackedPages[activeTrackedPageIndex + 1] ?? null
                                )
                              }
                            }}
                            disabled={
                              activeTrackedPageIndex < 0 ||
                              activeTrackedPageIndex >= trackedPages.length - 1
                            }
                          >
                            下一页
                          </Button>
                        </>
                      ) : (
                        <div className="font-mono text-xs text-muted-foreground">
                          当前任务未保留逐页过程图
                        </div>
                      )}
                      {sourcePdfAbsoluteUrl ? (
                        <>
                          <Button
                            type="button"
                            size="xs"
                            variant={showInlinePdf ? "default" : "outline"}
                            onClick={() => setShowInlinePdf((prev) => !prev)}
                          >
                            {showInlinePdf ? "收起 PDF 预览" : "内嵌预览 PDF"}
                          </Button>
                          <Button asChild type="button" size="xs" variant="outline">
                            <a href={sourcePdfAbsoluteUrl} target="_blank" rel="noreferrer">
                              新窗口查看 PDF
                            </a>
                          </Button>
                        </>
                      ) : null}
                    </div>
                  ) : (
                    <div className="text-sm text-muted-foreground">当前任务尚未产出可视化图片。</div>
                  )}

                  {!hasTrackedVisualArtifacts ? (
                    <div className="border border-border bg-muted/20 px-3 py-3 text-sm text-muted-foreground">
                      {trackedArtifacts.artifacts_retained
                        ? "当前任务没有可展示的逐页过程图。"
                        : "当前任务未保留过程图。需要对比渲染前后效果时，请在首页提交任务时开启“保留过程对比图（调试）”。"}
                    </div>
                  ) : null}

                  {showInlinePdf && sourcePdfAbsoluteUrl ? (
                    <div className="overflow-hidden border bg-muted/10">
                      <iframe
                        src={sourcePdfAbsoluteUrl}
                        title={`原始 PDF - ${trackedArtifacts.job_id}`}
                        className="h-[68vh] w-full bg-white"
                        loading="lazy"
                      />
                    </div>
                  ) : null}

                  {hasTrackedVisualArtifacts && trackingMenu === "frames" ? (
                    <div className="grid gap-4 lg:grid-cols-2">
                      <div className="grid gap-2">
                        <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                          原始 PDF（悬停显示识别框）
                        </div>
                        <div className="panel-contrast group relative min-h-[22rem] overflow-hidden border sm:min-h-[28rem]">
                          {trackedOriginal ? (
                            <TrackingArtifactImage
                              src={`${apiOrigin}${trackedOriginal.url}`}
                              alt={`原始第 ${activeTrackedPageLabel} 页`}
                              className="object-contain"
                              priority
                            />
                          ) : (
                            <div className="grid h-52 place-items-center text-xs text-white/80">
                              暂无原始页图
                            </div>
                          )}
                          {trackedBeforeOverlay ? (
                            <TrackingArtifactImage
                              src={`${apiOrigin}${trackedBeforeOverlay.url}`}
                              alt={`第 ${activeTrackedPageLabel} 页识别框`}
                              className="pointer-events-none object-contain opacity-0 transition-opacity duration-200 group-hover:opacity-100"
                            />
                          ) : null}
                        </div>
                      </div>

                      <div className="grid gap-2">
                        <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                          转换完成图（悬停显示后处理框）
                        </div>
                        <div className="panel-contrast group relative min-h-[22rem] overflow-hidden border sm:min-h-[28rem]">
                          {trackedAfterOverlay ? (
                            <TrackingArtifactImage
                              src={`${apiOrigin}${trackedAfterOverlay.url}`}
                              alt={`第 ${activeTrackedPageLabel} 页转换对比`}
                              className="object-contain"
                              priority
                            />
                          ) : trackedOriginal ? (
                            <TrackingArtifactImage
                              src={`${apiOrigin}${trackedOriginal.url}`}
                              alt={`第 ${activeTrackedPageLabel} 页原图`}
                              className="object-contain"
                              priority
                            />
                          ) : (
                            <div className="grid h-52 place-items-center text-xs text-white/80">
                              暂无转换对比图
                            </div>
                          )}
                          {trackedLayoutAfter ? (
                            <TrackingArtifactImage
                              src={`${apiOrigin}${trackedLayoutAfter.url}`}
                              alt={`第 ${activeTrackedPageLabel} 页后处理框`}
                              className="pointer-events-none object-contain opacity-0 transition-opacity duration-200 group-hover:opacity-100"
                            />
                          ) : null}
                        </div>
                      </div>
                    </div>
                  ) : null}

                  {hasTrackedVisualArtifacts && trackingMenu === "compare" ? (
                    <div className="grid gap-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                          悬停对比（左：转换前 · 右：转换后）
                        </div>
                        <Badge variant="outline">分割线 {compareSplitPercent}%</Badge>
                      </div>
                      <div
                        className="panel-contrast-strong group relative min-h-[24rem] overflow-hidden border sm:min-h-[30rem]"
                        onMouseMove={handleComparePointerMove}
                        onTouchStart={handleCompareTouchMove}
                        onTouchMove={handleCompareTouchMove}
                      >
                        {trackedCompareBase ? (
                          <TrackingArtifactImage
                            src={`${apiOrigin}${trackedCompareBase.url}`}
                            alt={`第 ${activeTrackedPageLabel} 页对比底图`}
                            className="object-contain"
                            priority
                          />
                        ) : (
                          <div className="grid h-64 place-items-center text-sm text-white/80">
                            暂无可用于对比的图片
                          </div>
                        )}

                        {trackedCompareAfter ? (
                          <div
                            className="pointer-events-none absolute inset-0 transition-[clip-path] duration-150 ease-out"
                            style={{ clipPath: `inset(0 0 0 ${compareSplitPercent}%)` }}
                          >
                            <TrackingArtifactImage
                              src={`${apiOrigin}${trackedCompareAfter.url}`}
                              alt={`第 ${activeTrackedPageLabel} 页转换后`}
                              className="object-contain"
                            />
                          </div>
                        ) : null}

                        {trackedBeforeOverlay ? (
                          <div
                            className="pointer-events-none absolute inset-0"
                            style={{ clipPath: `inset(0 ${100 - compareSplitPercent}% 0 0)` }}
                          >
                            <TrackingArtifactImage
                              src={`${apiOrigin}${trackedBeforeOverlay.url}`}
                              alt={`第 ${activeTrackedPageLabel} 页转换前高亮`}
                              className="object-contain opacity-45"
                            />
                          </div>
                        ) : null}

                        {trackedLayoutAfter && trackedCompareAfter?.path !== trackedLayoutAfter.path ? (
                          <div
                            className="pointer-events-none absolute inset-0"
                            style={{ clipPath: `inset(0 0 0 ${compareSplitPercent}%)` }}
                          >
                            <TrackingArtifactImage
                              src={`${apiOrigin}${trackedLayoutAfter.url}`}
                              alt={`第 ${activeTrackedPageLabel} 页转换后高亮`}
                              className="object-contain opacity-60"
                            />
                          </div>
                        ) : null}

                        <div
                          className="compare-divider pointer-events-none absolute inset-y-0 z-20 w-0.5"
                          style={{ left: `${compareSplitPercent}%` }}
                        />
                        <div className="pointer-events-none absolute left-2 top-2 z-20 border bg-black/50 px-2 py-1 font-mono text-[11px] text-white">
                          转换前
                        </div>
                        <div className="pointer-events-none absolute right-2 top-2 z-20 border bg-black/50 px-2 py-1 font-mono text-[11px] text-white">
                          转换后
                        </div>
                      </div>
                      <div className="grid gap-2 border border-border bg-muted/20 p-3 sm:grid-cols-[1fr_auto] sm:items-center">
                        <label
                          htmlFor="compare-split"
                          className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground"
                        >
                          分割线位置
                        </label>
                        <Badge variant="outline">{compareSplitPercent}%</Badge>
                        <input
                          id="compare-split"
                          type="range"
                          min={0}
                          max={100}
                          step={1}
                          value={compareSplitPercent}
                          onChange={(e) => {
                            const next = Number(e.target.value)
                            if (Number.isFinite(next)) {
                              setCompareSplitRatio(Math.max(0, Math.min(1, next / 100)))
                            }
                          }}
                          className="col-span-full h-2 w-full accent-[#111111]"
                          aria-label="调整前后对比滑杆位置"
                        />
                      </div>
                      <div className="text-xs text-muted-foreground">
                        桌面端可移动鼠标调整分割线，移动端可拖动图片或使用滑杆精确控制对比位置。
                      </div>
                    </div>
                  ) : null}
                </>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default function TrackingPage() {
  return (
    <React.Suspense
      fallback={
        <div className="min-h-dvh bg-background">
          <div className="mx-auto w-full max-w-screen-xl px-4 py-6 md:py-10">
            <div className="border border-border bg-background px-4 py-6 text-sm text-muted-foreground">
              正在加载跟踪页面...
            </div>
          </div>
        </div>
      }
    >
      <TrackingPageContent />
    </React.Suspense>
  )
}
