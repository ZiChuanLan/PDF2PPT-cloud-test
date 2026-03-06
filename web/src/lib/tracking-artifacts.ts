export function normalizeArtifactPages(values: unknown): number[] {
  if (!Array.isArray(values)) return []
  return Array.from(
    new Set(
      values
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value) && value >= 0)
        .sort((a, b) => a - b)
    )
  )
}

export function getFirstArtifactPage(pages: number[]): number | null {
  return pages.length ? pages[0] ?? null : null
}

export function resolveActiveArtifactPage(
  pages: number[],
  requestedPage: number | null
): number | null {
  if (!pages.length) return null
  if (requestedPage !== null && pages.includes(requestedPage)) {
    return requestedPage
  }
  return getFirstArtifactPage(pages)
}

export function getArtifactPageIndex(
  pages: number[],
  activePage: number | null
): number {
  if (activePage === null) return -1
  return pages.indexOf(activePage)
}

export function formatArtifactPageLabel(page: number | null): string {
  if (page === null) return "-"
  return String(page + 1)
}
