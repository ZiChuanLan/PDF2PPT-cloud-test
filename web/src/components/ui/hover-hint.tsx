"use client"

import * as React from "react"
import { CircleHelpIcon } from "lucide-react"

import { cn } from "@/lib/utils"

type HoverHintProps = {
  text: string
  label?: string
  className?: string
}

function HoverHint({
  text,
  label = "说明",
  className,
}: HoverHintProps) {
  return (
    <button
      type="button"
      title={text}
      aria-label={`${label}: ${text}`}
      className={cn(
        "inline-flex size-4 shrink-0 cursor-help items-center justify-center text-muted-foreground/70 transition-colors hover:text-foreground focus-visible:text-foreground focus-visible:outline-none",
        className
      )}
    >
      <CircleHelpIcon className="size-3.5" />
      <span className="sr-only">{label}</span>
    </button>
  )
}

export { HoverHint }
