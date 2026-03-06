export type JobStatusValue = "pending" | "processing" | "completed" | "failed" | "cancelled"
export type JobQueueState = "queued" | "running" | "waiting" | "done"

export type JobListItem = {
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

export type JobListResponse = {
  jobs: JobListItem[]
  queue_size: number
  returned: number
}

export type JobStatusResponse = {
  job_id: string
  status: JobStatusValue
  stage: string
  progress: number
  created_at: string
  expires_at: string
  message?: string | null
  error?: { code?: string; message?: string } | null
}

const JOB_STATUS_VALUES = new Set<JobStatusValue>([
  "pending",
  "processing",
  "completed",
  "failed",
  "cancelled",
])

export const TERMINAL_JOB_STATUSES = new Set<JobStatusValue>([
  "completed",
  "failed",
  "cancelled",
])

export const JOB_STATUS_LABELS: Record<JobStatusValue, string> = {
  pending: "排队中",
  processing: "处理中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
}

export const JOB_STAGE_LABELS: Record<string, string> = {
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

export const JOB_STAGE_COMPACT_LABELS: Record<string, string> = {
  queued: "队列",
  parsing: "解析",
  ocr: "OCR",
  layout_assist: "版式",
  pptx_generating: "PPTX",
  packaging: "打包",
  cleanup: "清理",
  done: "完成",
}

export const QUEUE_STATE_LABELS: Record<string, string> = {
  queued: "排队中",
  running: "执行中",
  waiting: "等待调度",
  done: "完成",
}

export const JOB_STAGE_FLOW = [
  "queued",
  "parsing",
  "ocr",
  "layout_assist",
  "pptx_generating",
  "packaging",
  "cleanup",
  "done",
] as const

const STAGE_FLOW_ALIASES: Record<string, string> = {
  upload_received: "queued",
}

function clampProgress(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return 0
  }
  return Math.max(0, Math.min(100, value))
}

function normalizeJobStatus(value: unknown): JobStatusValue {
  if (typeof value === "string" && JOB_STATUS_VALUES.has(value as JobStatusValue)) {
    return value as JobStatusValue
  }
  return "pending"
}

function normalizeQueuePosition(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null
  }
  return Math.max(1, Math.floor(value))
}

function normalizeIsoOrNow(value: unknown): string {
  if (typeof value === "string" && value.trim()) {
    return value
  }
  return new Date().toISOString()
}

export function normalizeJobListItem(row: unknown): JobListItem | null {
  if (!row || typeof row !== "object") return null

  const jobId = typeof (row as { job_id?: unknown }).job_id === "string" ? (row as { job_id: string }).job_id : ""
  if (!jobId) return null

  const errorValue = (row as { error?: unknown }).error
  return {
    job_id: jobId,
    status: normalizeJobStatus((row as { status?: unknown }).status),
    stage:
      typeof (row as { stage?: unknown }).stage === "string"
        ? (row as { stage: string }).stage
        : "queued",
    progress: clampProgress((row as { progress?: unknown }).progress),
    created_at: normalizeIsoOrNow((row as { created_at?: unknown }).created_at),
    expires_at: normalizeIsoOrNow((row as { expires_at?: unknown }).expires_at),
    message:
      typeof (row as { message?: unknown }).message === "string"
        ? (row as { message: string }).message
        : null,
    error: errorValue && typeof errorValue === "object" ? (errorValue as JobListItem["error"]) : null,
    queue_position: normalizeQueuePosition((row as { queue_position?: unknown }).queue_position),
    queue_state:
      typeof (row as { queue_state?: unknown }).queue_state === "string"
        ? (row as { queue_state: string }).queue_state
        : null,
  }
}

export function normalizeJobListResponse(body: unknown): {
  jobs: JobListItem[]
  queueSize: number
} {
  const rows =
    body && typeof body === "object" && Array.isArray((body as { jobs?: unknown }).jobs)
      ? (body as { jobs: unknown[] }).jobs
      : []

  return {
    jobs: rows.map((row) => normalizeJobListItem(row)).filter((row): row is JobListItem => row !== null),
    queueSize:
      body && typeof body === "object" && typeof (body as { queue_size?: unknown }).queue_size === "number"
        ? Math.max(0, Math.floor((body as { queue_size: number }).queue_size))
        : 0,
  }
}

export function normalizeJobStatusResponse(body: unknown): JobStatusResponse {
  const normalized = normalizeJobListItem(body)
  if (!normalized) {
    throw new Error("任务状态响应异常")
  }
  return {
    job_id: normalized.job_id,
    status: normalized.status,
    stage: normalized.stage,
    progress: normalized.progress,
    created_at: normalized.created_at,
    expires_at: normalized.expires_at,
    message: normalized.message,
    error: normalized.error,
  }
}

export function getJobStageFlowStage(stage: string | null | undefined): string | null {
  if (!stage) return null
  return STAGE_FLOW_ALIASES[stage] || stage
}

export function getJobStageFlowIndex(stage: string | null | undefined): number {
  const normalizedStage = getJobStageFlowStage(stage)
  if (!normalizedStage) return -1
  return JOB_STAGE_FLOW.findIndex((item) => item === normalizedStage)
}
