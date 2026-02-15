"use client"

import * as React from "react"
import Link from "next/link"
import { ArrowLeftIcon } from "lucide-react"
import { useSearchParams } from "next/navigation"
import { toast } from "sonner"

import { cn } from "@/lib/utils"
import { apiFetch, normalizeFetchError, resolveApiOrigin } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/select"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

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

type JobArtifactImage = {
  page_index: number
  path: string
  url: string
}

type JobArtifactsResponse = {
  job_id: string
  status?: JobStatusValue | null
  source_pdf_url?: string | null
  original_images: JobArtifactImage[]
  cleaned_images: JobArtifactImage[]
  final_preview_images: JobArtifactImage[]
  ocr_overlay_images: JobArtifactImage[]
  layout_before_images: JobArtifactImage[]
  layout_after_images: JobArtifactImage[]
  available_pages: number[]
}

const jobStatusLabels: Record<string, string> = {
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

const queueStateLabels: Record<string, string> = {
  queued: "排队中",
  running: "执行中",
  waiting: "等待调度",
  done: "完成",
}

const jobStatusFilterOptions: Array<{ value: "all" | JobStatusValue; label: string }> = [
  { value: "all", label: "全部状态" },
  { value: "pending", label: "排队中" },
  { value: "processing", label: "处理中" },
  { value: "completed", label: "已完成" },
  { value: "failed", label: "失败" },
  { value: "cancelled", label: "已取消" },
]

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

function TrackingPageContent() {
  const [apiOrigin, setApiOrigin] = React.useState("")
  const searchParams = useSearchParams()
  const requestedJobId = (searchParams.get("job") || "").trim()

  const [jobRecords, setJobRecords] = React.useState<JobListItem[]>([])
  const [queueSize, setQueueSize] = React.useState(0)
  const [isJobsLoading, setIsJobsLoading] = React.useState(false)
  const [jobsError, setJobsError] = React.useState<string | null>(null)

  const [trackedArtifacts, setTrackedArtifacts] = React.useState<JobArtifactsResponse | null>(
    null
  )
  const [trackedArtifactsLoading, setTrackedArtifactsLoading] = React.useState(false)
  const [trackedArtifactsError, setTrackedArtifactsError] = React.useState<string | null>(null)
  const [trackedArtifactsPage, setTrackedArtifactsPage] = React.useState(1)
  const [trackingMenu, setTrackingMenu] = React.useState<"frames" | "compare">("compare")
  const [compareSplitRatio, setCompareSplitRatio] = React.useState(0.5)
  const [showInlinePdf, setShowInlinePdf] = React.useState(false)
  const [jobKeyword, setJobKeyword] = React.useState("")
  const [statusFilter, setStatusFilter] = React.useState<"all" | JobStatusValue>("all")

  const currentTrackedJob = React.useMemo(() => {
    if (!trackedArtifacts?.job_id) return null
    return jobRecords.find((r) => r.job_id === trackedArtifacts.job_id) || null
  }, [jobRecords, trackedArtifacts?.job_id])

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
      const rows = Array.isArray(body?.jobs) ? body.jobs : []
      const normalized = rows
        .map((row) => {
          if (!row || typeof row !== "object") return null
          if (typeof row.job_id !== "string" || !row.job_id) return null
          return {
            job_id: row.job_id,
            status: (row.status || "pending") as JobStatusValue,
            stage: typeof row.stage === "string" ? row.stage : "queued",
            progress:
              typeof row.progress === "number"
                ? Math.max(0, Math.min(100, row.progress))
                : 0,
            created_at:
              typeof row.created_at === "string"
                ? row.created_at
                : new Date().toISOString(),
            expires_at:
              typeof row.expires_at === "string"
                ? row.expires_at
                : new Date().toISOString(),
            message: typeof row.message === "string" ? row.message : null,
            error: row.error && typeof row.error === "object" ? row.error : null,
            queue_position:
              typeof row.queue_position === "number" ? row.queue_position : null,
            queue_state: typeof row.queue_state === "string" ? row.queue_state : null,
          } as JobListItem
        })
        .filter((row): row is JobListItem => row !== null)

      setJobRecords(normalized)
      setQueueSize(
        typeof body?.queue_size === "number" ? Math.max(0, body.queue_size) : 0
      )
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
        available_pages: Array.isArray(body.available_pages)
          ? body.available_pages
              .map((v) => Number(v))
              .filter((v) => Number.isFinite(v) && v > 0)
          : [],
      })
    } catch (e) {
      setTrackedArtifacts(null)
      setTrackedArtifactsError(normalizeFetchError(e, "加载任务产物失败"))
    } finally {
      setTrackedArtifactsLoading(false)
    }
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
    void fetchJobArtifacts(requestedJobId)
  }, [fetchJobArtifacts, requestedJobId])

  React.useEffect(() => {
    const pages = trackedArtifacts?.available_pages || []
    if (!pages.length) {
      setTrackedArtifactsPage(1)
      return
    }
    if (!pages.includes(trackedArtifactsPage)) {
      setTrackedArtifactsPage(pages[0])
    }
  }, [trackedArtifacts, trackedArtifactsPage])

  const findArtifactByPage = React.useCallback(
    (images: JobArtifactImage[] | undefined, page: number) => {
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
  const activeTrackedPage = trackedPages.includes(trackedArtifactsPage)
    ? trackedArtifactsPage
    : trackedPages[0] || 1
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

  return (
    <div className="min-h-dvh bg-background">
      <div className="mx-auto w-full max-w-screen-xl px-4 py-6 md:py-10">
        <header className="border border-border bg-background p-5 md:p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="font-mono text-xs uppercase tracking-[0.16em] text-muted-foreground">
                跟踪中心
              </div>
              <h1 className="mt-2 font-serif text-4xl leading-[0.95] md:text-5xl">
                转换追踪与前后对比
              </h1>
              <p className="mt-3 text-sm text-muted-foreground">
                单独查看任务产物，按页对比原始图与转换完成图。
              </p>
            </div>
            <Button asChild variant="outline">
              <Link href="/">
                <ArrowLeftIcon className="size-4" />
                返回主页
              </Link>
            </Button>
          </div>
        </header>

        <div className="mt-4 grid gap-4 lg:grid-cols-12">
          <Card className="py-0 lg:col-span-4">
            <CardHeader className="border-b border-border">
              <div className="flex items-center justify-between gap-2">
                <CardTitle>队列 / 历史记录</CardTitle>
                <Badge variant="outline">排队 {queueSize}</Badge>
              </div>
              <CardDescription>支持按状态和任务号筛选，点击后在右侧查看产物追踪。</CardDescription>
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
                <div className="max-h-[70vh] overflow-y-auto border">
                  {filteredJobRecords.map((record) => {
                    const isCurrent = record.job_id === trackedArtifacts?.job_id
                    const statusLabel = jobStatusLabels[record.status] || record.status
                    const stageLabel = jobStageLabels[record.stage] || record.stage
                    const detailMessage =
                      (record.status === "failed" &&
                        typeof record.error?.message === "string" &&
                        record.error.message.trim()) ||
                      (typeof record.message === "string" && record.message.trim()) ||
                      null
                    const badgeTone =
                      record.status === "completed"
                        ? "bg-[#d9fbe2] text-[#0f5132]"
                        : record.status === "failed"
                          ? "bg-[#ffe9e9] text-[#7d1111]"
                          : record.status === "cancelled"
                            ? "bg-[#e5e5e0] text-[#222222]"
                            : "bg-[#fff3cd] text-[#6b4f00]"
                    return (
                      <div
                        key={record.job_id}
                        className={cn(
                          "border-b px-3 py-2 last:border-b-0",
                          isCurrent && "bg-muted"
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="font-mono text-[11px] uppercase tracking-[0.14em]">
                            {record.job_id.slice(0, 8)}
                          </div>
                          <Badge className={cn("border-0", badgeTone)}>{statusLabel}</Badge>
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
                            队列状态：{queueStateLabels[record.queue_state] || record.queue_state}
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

          <Card className="py-0 lg:col-span-8">
            <CardHeader className="border-b border-border">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <CardTitle>跟踪画面</CardTitle>
                  <CardDescription>支持按页查看与悬停分割对比</CardDescription>
                </div>
                <Badge variant="outline">
                  {trackedArtifacts?.job_id ? trackedArtifacts.job_id.slice(0, 8) : "未选择任务"}
                </Badge>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  type="button"
                  size="xs"
                  variant={trackingMenu === "frames" ? "default" : "outline"}
                  onClick={() => setTrackingMenu("frames")}
                >
                  跟踪画面
                </Button>
                <Button
                  type="button"
                  size="xs"
                  variant={trackingMenu === "compare" ? "default" : "outline"}
                  onClick={() => setTrackingMenu("compare")}
                >
                  前后对比（悬停高亮）
                </Button>
              </div>
              {currentTrackedJob ? (
                <div className="mt-3 grid gap-1 border border-border bg-muted/40 px-3 py-2">
                  <div className="text-xs text-muted-foreground">
                    {jobStageLabels[currentTrackedJob.stage] || currentTrackedJob.stage} ·{" "}
                    {currentTrackedJob.progress}%
                  </div>
                  {(currentTrackedJob.status === "failed" &&
                    typeof currentTrackedJob.error?.message === "string" &&
                    currentTrackedJob.error.message.trim()) ||
                  (currentTrackedJob.message && currentTrackedJob.message.trim()) ? (
                    <div className="text-xs text-muted-foreground">
                      {(currentTrackedJob.status === "failed" &&
                        typeof currentTrackedJob.error?.message === "string" &&
                        currentTrackedJob.error.message.trim()) ||
                        currentTrackedJob.message}
                    </div>
                  ) : null}
                  <div className="font-mono text-[11px] text-muted-foreground">
                    阶段代码：{currentTrackedJob.stage}
                  </div>
                  {currentTrackedJob.queue_state === "queued" &&
                  typeof currentTrackedJob.queue_position === "number" ? (
                    <div className="font-mono text-[11px] text-muted-foreground">
                      排队位置：第 {currentTrackedJob.queue_position} 位
                    </div>
                  ) : currentTrackedJob.queue_state ? (
                    <div className="font-mono text-[11px] text-muted-foreground">
                      队列状态：{queueStateLabels[currentTrackedJob.queue_state] || currentTrackedJob.queue_state}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </CardHeader>
            <CardContent className="grid gap-4 py-5">
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
                  {trackedPages.length ? (
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="font-mono text-xs text-muted-foreground">
                        第 {activeTrackedPage} 页 / 共 {trackedPages.length} 页
                      </div>
                      <Button
                        type="button"
                        size="xs"
                        variant="outline"
                        onClick={() => {
                          const idx = trackedPages.indexOf(activeTrackedPage)
                          if (idx > 0) setTrackedArtifactsPage(trackedPages[idx - 1])
                        }}
                        disabled={activeTrackedPage <= (trackedPages[0] || 1)}
                      >
                        上一页
                      </Button>
                      <Button
                        type="button"
                        size="xs"
                        variant="outline"
                        onClick={() => {
                          const idx = trackedPages.indexOf(activeTrackedPage)
                          if (idx >= 0 && idx < trackedPages.length - 1) {
                            setTrackedArtifactsPage(trackedPages[idx + 1])
                          }
                        }}
                        disabled={activeTrackedPage >= (trackedPages[trackedPages.length - 1] || 1)}
                      >
                        下一页
                      </Button>
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

                  {trackedPages.length && trackingMenu === "frames" ? (
                    <div className="grid gap-4 lg:grid-cols-2">
                      <div className="grid gap-2">
                        <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                          原始 PDF（悬停显示识别框）
                        </div>
                        <div className="group relative overflow-hidden border bg-[#1e1e1e]">
                          {trackedOriginal ? (
                            <img
                              src={`${apiOrigin}${trackedOriginal.url}`}
                              alt={`原始第 ${activeTrackedPage} 页`}
                              className="block h-auto w-full"
                            />
                          ) : (
                            <div className="grid h-52 place-items-center text-xs text-[#e5e5e0]">
                              暂无原始页图
                            </div>
                          )}
                          {trackedBeforeOverlay ? (
                            <img
                              src={`${apiOrigin}${trackedBeforeOverlay.url}`}
                              alt={`第 ${activeTrackedPage} 页识别框`}
                              className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-0 transition-opacity duration-200 group-hover:opacity-100"
                            />
                          ) : null}
                        </div>
                      </div>

                      <div className="grid gap-2">
                        <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                          转换完成图（悬停显示后处理框）
                        </div>
                        <div className="group relative overflow-hidden border bg-[#1e1e1e]">
                          {trackedAfterOverlay ? (
                            <img
                              src={`${apiOrigin}${trackedAfterOverlay.url}`}
                              alt={`第 ${activeTrackedPage} 页转换对比`}
                              className="block h-auto w-full"
                            />
                          ) : trackedOriginal ? (
                            <img
                              src={`${apiOrigin}${trackedOriginal.url}`}
                              alt={`第 ${activeTrackedPage} 页原图`}
                              className="block h-auto w-full"
                            />
                          ) : (
                            <div className="grid h-52 place-items-center text-xs text-[#e5e5e0]">
                              暂无转换对比图
                            </div>
                          )}
                          {trackedLayoutAfter ? (
                            <img
                              src={`${apiOrigin}${trackedLayoutAfter.url}`}
                              alt={`第 ${activeTrackedPage} 页后处理框`}
                              className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-0 transition-opacity duration-200 group-hover:opacity-100"
                            />
                          ) : null}
                        </div>
                      </div>
                    </div>
                  ) : null}

                  {trackedPages.length && trackingMenu === "compare" ? (
                    <div className="grid gap-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                          悬停对比（左：转换前 · 右：转换后）
                        </div>
                        <Badge variant="outline">分割线 {compareSplitPercent}%</Badge>
                      </div>
                      <div
                        className="group relative overflow-hidden border bg-[#101010]"
                        onMouseMove={handleComparePointerMove}
                        onTouchStart={handleCompareTouchMove}
                        onTouchMove={handleCompareTouchMove}
                      >
                        {trackedCompareBase ? (
                          <img
                            src={`${apiOrigin}${trackedCompareBase.url}`}
                            alt={`第 ${activeTrackedPage} 页对比底图`}
                            className="block h-auto w-full"
                          />
                        ) : (
                          <div className="grid h-64 place-items-center text-sm text-[#e5e5e0]">
                            暂无可用于对比的图片
                          </div>
                        )}

                        {trackedCompareAfter ? (
                          <img
                            src={`${apiOrigin}${trackedCompareAfter.url}`}
                            alt={`第 ${activeTrackedPage} 页转换后`}
                            className="pointer-events-none absolute inset-0 h-full w-full object-contain transition-[clip-path] duration-75"
                            style={{ clipPath: `inset(0 0 0 ${compareSplitPercent}%)` }}
                          />
                        ) : null}

                        {trackedBeforeOverlay ? (
                          <img
                            src={`${apiOrigin}${trackedBeforeOverlay.url}`}
                            alt={`第 ${activeTrackedPage} 页转换前高亮`}
                            className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-45"
                            style={{ clipPath: `inset(0 ${100 - compareSplitPercent}% 0 0)` }}
                          />
                        ) : null}

                        {trackedLayoutAfter && trackedCompareAfter?.path !== trackedLayoutAfter.path ? (
                          <img
                            src={`${apiOrigin}${trackedLayoutAfter.url}`}
                            alt={`第 ${activeTrackedPage} 页转换后高亮`}
                            className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-60"
                            style={{ clipPath: `inset(0 0 0 ${compareSplitPercent}%)` }}
                          />
                        ) : null}

                        <div
                          className="pointer-events-none absolute inset-y-0 z-20 w-0.5 bg-[#ffe082]"
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
