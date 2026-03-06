"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { LayoutDashboardIcon, ListChecksIcon, Settings2Icon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

const workbenchNavItems = [
  {
    href: "/",
    label: "首页",
    description: "上传与执行",
    icon: LayoutDashboardIcon,
  },
  {
    href: "/tracking",
    label: "跟踪",
    description: "任务排障与对比",
    icon: ListChecksIcon,
  },
  {
    href: "/settings",
    label: "设置",
    description: "策略与接口配置",
    icon: Settings2Icon,
  },
] as const

function matchesRoute(pathname: string, href: string) {
  if (href === "/") return pathname === "/"
  return pathname === href || pathname.startsWith(`${href}/`)
}

export function WorkbenchNav() {
  const pathname = usePathname()

  if (!pathname) return null

  const activeItem = workbenchNavItems.find((item) => matchesRoute(pathname, item.href))

  if (!activeItem) return null

  return (
    <div className="sticky top-0 z-40 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/90">
      <div className="mx-auto flex w-full max-w-screen-xl flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              PDF 编排台
            </div>
            <Badge variant="outline" className="font-sans text-[11px] uppercase tracking-[0.12em]">
              Unified Workbench
            </Badge>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <div className="font-serif text-lg leading-none md:text-xl">{activeItem.label}</div>
            <div className="font-sans text-sm text-muted-foreground">{activeItem.description}</div>
          </div>
        </div>

        <nav aria-label="工作台导航" className="flex flex-wrap items-center gap-2">
          {workbenchNavItems.map((item) => {
            const active = matchesRoute(pathname, item.href)
            const Icon = item.icon
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "nav-highlight flex items-center gap-2 border px-3 py-2 font-sans text-sm",
                  active
                    ? "nav-highlight-active border-border bg-secondary text-foreground"
                    : "nav-highlight-inactive border-border/70 bg-background text-muted-foreground hover:text-foreground"
                )}
                aria-current={active ? "page" : undefined}
              >
                <Icon className="size-4" />
                <span>{item.label}</span>
              </Link>
            )
          })}
        </nav>
      </div>
    </div>
  )
}
