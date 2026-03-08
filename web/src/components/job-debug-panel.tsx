"use client"

import * as React from "react"

import { JOB_STAGE_LABELS, type JobDebugEvent } from "@/lib/job-status"
import { cn } from "@/lib/utils"

function formatDebugTime(iso: string) {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "Asia/Shanghai",
  }).format(date)
}

function compactSourceLabel(source: string | null | undefined) {
  if (!source) return null
  const segments = source.split(".").filter(Boolean)
  return segments[segments.length - 1] || source
}

function getLevelClass(level: string) {
  if (level === "error" || level === "critical") {
    return "border-destructive/30 bg-destructive/10 text-destructive"
  }
  if (level === "warning") {
    return "border-amber-300/40 bg-amber-500/10 text-amber-700"
  }
  return "border-border bg-background text-foreground"
}

export function JobDebugPanel({
  events,
  title = "处理日志",
  emptyLabel = "暂无处理记录",
  compact = false,
  className,
}: {
  events: JobDebugEvent[]
  title?: string
  emptyLabel?: string
  compact?: boolean
  className?: string
}) {
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const lastSeq = events[events.length - 1]?.seq ?? 0

  React.useEffect(() => {
    const node = containerRef.current
    if (!node) return
    node.scrollTop = node.scrollHeight
  }, [lastSeq])

  return (
    <section
      className={cn(
        "grid gap-2",
        compact && "w-full md:max-w-[18.5rem] justify-self-start",
        className
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div
          className={cn(
            "font-mono uppercase tracking-[0.14em] text-muted-foreground",
            compact ? "text-[10px]" : "text-[11px]"
          )}
        >
          {title}
        </div>
        <div
          className={cn(
            "font-mono text-muted-foreground",
            compact ? "text-[10px] leading-4" : "text-[11px]"
          )}
        >
          {events.length ? `${events.length} 条` : emptyLabel}
        </div>
      </div>

      <div
        ref={containerRef}
        className={cn(
          "overflow-y-auto border border-border bg-muted/20",
          compact ? "max-h-44" : "max-h-72"
        )}
      >
        {events.length ? (
          <div className="divide-y divide-border/70">
            {events.map((event) => {
              const sourceLabel = compactSourceLabel(event.source)
              const stageLabel = event.stage ? JOB_STAGE_LABELS[event.stage] || event.stage : null
              return (
                <article
                  key={event.seq}
                  className={cn("grid px-3 py-2", compact ? "gap-1 px-2.5 py-1.5" : "gap-1.5")}
                >
                  <div
                    className={cn(
                      "flex flex-wrap items-center gap-2 font-mono text-muted-foreground",
                      compact ? "text-[10px] leading-4" : "text-[11px]"
                    )}
                  >
                    <span>{formatDebugTime(event.timestamp)}</span>
                    <span
                      className={cn(
                        "rounded border px-1.5 py-0.5 uppercase tracking-[0.12em]",
                        compact ? "text-[9px]" : "text-[10px]",
                        getLevelClass(event.level)
                      )}
                    >
                      {event.level}
                    </span>
                    {typeof event.progress === "number" ? <span>{event.progress}%</span> : null}
                    {stageLabel ? <span>{stageLabel}</span> : null}
                    {sourceLabel ? <span>{sourceLabel}</span> : null}
                  </div>
                  <div className={cn("text-foreground", compact ? "text-xs leading-5" : "text-sm leading-6")}>
                    {event.message}
                  </div>
                </article>
              )
            })}
          </div>
        ) : (
          <div
            className={cn(
              "px-3 py-4 font-mono text-muted-foreground",
              compact ? "px-2.5 py-3 text-[10px] leading-4" : "text-xs"
            )}
          >
            {emptyLabel}
          </div>
        )}
      </div>
    </section>
  )
}
