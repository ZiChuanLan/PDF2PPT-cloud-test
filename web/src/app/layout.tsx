import type { Metadata } from "next"
import "./globals.css"

import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/sonner"
import { UploadSessionProvider } from "@/components/upload-session-provider"
import { WorkbenchNav } from "@/components/workbench-nav"

export const metadata: Metadata = {
  title: "PDF 转 PPT",
  description: "上传 PDF，自动生成可编辑 PPT",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body
        className="font-body antialiased"
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem={false}
          forcedTheme="light"
          disableTransitionOnChange
        >
          <UploadSessionProvider>
            <WorkbenchNav />
            {children}
            <Toaster />
          </UploadSessionProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
