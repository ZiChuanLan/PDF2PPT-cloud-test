"use client"

import * as React from "react"
import { Loader2Icon } from "lucide-react"
import type {
  PDFDocumentLoadingTask,
  PDFDocumentProxy,
  RenderTask,
} from "pdfjs-dist/types/src/display/api"

import { cn } from "@/lib/utils"

type PdfCanvasPreviewProps = {
  fileUrl: string
  page: number
  onPageCountChange?: (count: number) => void
  className?: string
}

const PDF_WORKER_SRC = new URL("pdfjs-dist/legacy/build/pdf.worker.min.mjs", import.meta.url).toString()
type PdfJsRuntimeModule = typeof import("pdfjs-dist/legacy/build/pdf.mjs")
const BYTE_TO_HEX_TABLE = Array.from(
  { length: 256 },
  (_, index) => index.toString(16).padStart(2, "0")
)

function ensureUint8ArrayToHexPolyfill() {
  const proto = Uint8Array.prototype as Uint8Array & { toHex?: unknown }
  if (typeof proto.toHex === "function") return

  const toHex = function (this: Uint8Array) {
    let result = ""
    for (let index = 0; index < this.length; index += 1) {
      result += BYTE_TO_HEX_TABLE[this[index]]
    }
    return result
  }

  try {
    Object.defineProperty(Uint8Array.prototype, "toHex", {
      value: toHex,
      configurable: true,
      writable: true,
    })
  } catch {
    try {
      ;(Uint8Array.prototype as Uint8Array & { toHex?: (this: Uint8Array) => string }).toHex = toHex
    } catch {
      // Ignore assignment failures and let pdfjs surface the original error.
    }
  }
}

function clampPage(page: number, pageCount: number) {
  const normalized = Number.isFinite(page) ? Math.max(1, Math.floor(page)) : 1
  if (pageCount <= 0) return normalized
  return Math.min(normalized, pageCount)
}

export function PdfCanvasPreview({
  fileUrl,
  page,
  onPageCountChange,
  className,
}: PdfCanvasPreviewProps) {
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
  const pdfjsRuntimeRef = React.useRef<PdfJsRuntimeModule | null>(null)
  const [documentProxy, setDocumentProxy] = React.useState<PDFDocumentProxy | null>(null)
  const [containerWidth, setContainerWidth] = React.useState(0)
  const [isPdfRuntimeReady, setIsPdfRuntimeReady] = React.useState(false)
  const [isLoading, setIsLoading] = React.useState(true)
  const [loadingError, setLoadingError] = React.useState<string | null>(null)
  const [isRendering, setIsRendering] = React.useState(false)
  const [renderError, setRenderError] = React.useState<string | null>(null)

  React.useEffect(() => {
    let cancelled = false

    const loadRuntime = async () => {
      try {
        ensureUint8ArrayToHexPolyfill()
        const runtime = await import("pdfjs-dist/legacy/build/pdf.mjs")
        if (cancelled) return

        if (runtime.GlobalWorkerOptions.workerSrc !== PDF_WORKER_SRC) {
          runtime.GlobalWorkerOptions.workerSrc = PDF_WORKER_SRC
        }
        pdfjsRuntimeRef.current = runtime
        setIsPdfRuntimeReady(true)
      } catch (error) {
        if (cancelled) return
        const message = error instanceof Error ? error.message : "PDF 预览器初始化失败"
        setLoadingError(message)
        setIsPdfRuntimeReady(false)
        setIsLoading(false)
      }
    }

    void loadRuntime()

    return () => {
      cancelled = true
    }
  }, [])

  React.useEffect(() => {
    const element = containerRef.current
    if (!element) return

    const updateWidth = () => {
      setContainerWidth(Math.max(0, element.clientWidth))
    }

    updateWidth()
    const observer = new ResizeObserver(updateWidth)
    observer.observe(element)

    return () => {
      observer.disconnect()
    }
  }, [fileUrl])

  React.useEffect(() => {
    if (!isPdfRuntimeReady || !pdfjsRuntimeRef.current) return

    let cancelled = false
    let loadingTask: PDFDocumentLoadingTask | null = null
    let nextDocument: PDFDocumentProxy | null = null

    setIsLoading(true)
    setLoadingError(null)
    setRenderError(null)
    setDocumentProxy(null)

    const load = async () => {
      try {
        loadingTask = pdfjsRuntimeRef.current!.getDocument(fileUrl)
        nextDocument = await loadingTask.promise

        if (cancelled) {
          await nextDocument.destroy()
          return
        }

        setDocumentProxy(nextDocument)
        const nextPageCount = Math.max(1, nextDocument.numPages || 1)
        onPageCountChange?.(nextPageCount)
        setIsLoading(false)
      } catch (error) {
        if (cancelled) return
        const message = error instanceof Error ? error.message : "PDF 加载失败"
        setLoadingError(message)
        setIsLoading(false)
      }
    }

    void load()

    return () => {
      cancelled = true
      if (loadingTask) {
        void loadingTask.destroy()
      }
      if (nextDocument) {
        void nextDocument.destroy()
      }
    }
  }, [fileUrl, isPdfRuntimeReady, onPageCountChange])

  React.useEffect(() => {
    if (!documentProxy) return
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    let cancelled = false
    let task: RenderTask | null = null

    const render = async () => {
      try {
        setIsRendering(true)
        setRenderError(null)

        const safePage = clampPage(page, documentProxy.numPages)
        const pdfPage = await documentProxy.getPage(safePage)
        if (cancelled) return

        const baseViewport = pdfPage.getViewport({ scale: 1 })
        const safeWidth = containerWidth > 0 ? containerWidth : container.clientWidth
        const availableWidth = Math.max(280, safeWidth - 32)
        const maxCanvasWidth = Math.min(availableWidth, 760)
        const scale = Math.max(0.1, maxCanvasWidth / baseViewport.width)
        const viewport = pdfPage.getViewport({ scale })

        const context = canvas.getContext("2d", { alpha: false })
        if (!context) {
          throw new Error("无法创建画布上下文")
        }

        const ratio = Math.min(2, window.devicePixelRatio || 1)
        canvas.width = Math.floor(viewport.width * ratio)
        canvas.height = Math.floor(viewport.height * ratio)
        canvas.style.width = `${Math.floor(viewport.width)}px`
        canvas.style.height = `${Math.floor(viewport.height)}px`

        context.setTransform(ratio, 0, 0, ratio, 0, 0)
        context.clearRect(0, 0, viewport.width, viewport.height)

        task = pdfPage.render({
          canvas,
          canvasContext: context,
          viewport,
        })
        await task.promise
        if (cancelled) return

        setIsRendering(false)
      } catch (error) {
        if (cancelled) return
        if (
          typeof error === "object" &&
          error !== null &&
          "name" in error &&
          error.name === "RenderingCancelledException"
        ) {
          return
        }
        const message = error instanceof Error ? error.message : "页面渲染失败"
        setRenderError(message)
        setIsRendering(false)
      }
    }

    void render()

    return () => {
      cancelled = true
      if (task) {
        task.cancel()
      }
    }
  }, [containerWidth, documentProxy, page])

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative border border-border bg-[#ecebe7] min-h-[300px]",
        className
      )}
    >
      {!loadingError ? (
        <div className="flex min-h-[300px] items-start justify-center overflow-auto p-3 md:p-4">
          <canvas ref={canvasRef} className="block border border-[#c8c8c8] bg-white shadow-sm max-w-full" />
        </div>
      ) : null}

      {isLoading ? (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 text-sm">
          <Loader2Icon className="mr-2 size-4 animate-spin" />
          正在加载 PDF...
        </div>
      ) : null}

      {isRendering && !isLoading ? (
        <div className="absolute right-3 top-3 border border-border bg-background px-2 py-1 text-[11px]">
          渲染中...
        </div>
      ) : null}

      {loadingError ? (
        <div className="flex min-h-[300px] items-center justify-center p-4 text-sm text-destructive">
          {loadingError}
        </div>
      ) : null}

      {renderError ? (
        <div className="absolute bottom-3 left-3 border border-destructive bg-destructive/10 px-2 py-1 text-[11px] text-destructive">
          {renderError}
        </div>
      ) : null}
    </div>
  )
}
