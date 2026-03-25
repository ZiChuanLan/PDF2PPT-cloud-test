import { defineConfig } from "vitepress"

const repo = "https://github.com/ZiChuanLan/PDF2PPT"
const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1] || "PDF2PPT"
const base =
  process.env.DOCS_BASE ||
  (process.env.GITHUB_ACTIONS ? `/${repoName}/` : "/")

const zhGuideSidebar = [
      {
        text: "开始使用",
        items: [
          { text: "部署指南", link: "/guide/deployment" },
          { text: "架构说明", link: "/guide/architecture" },
          { text: "MCP 集成", link: "/guide/mcp-integration" },
          { text: "OCR 与解析链路", link: "/guide/ocr-pipelines" },
          { text: "FAQ 与排障", link: "/guide/faq" },
        ],
      },
  {
    text: "扩展文档",
    items: [{ text: "MCP Server PRD", link: "/mcp-server-prd" }],
  },
]

const enGuideSidebar = [
      {
        text: "Getting Started",
        items: [
          { text: "Deployment Guide", link: "/en/guide/deployment" },
          { text: "Architecture", link: "/en/guide/architecture" },
          { text: "MCP Integration", link: "/en/guide/mcp-integration" },
          { text: "OCR and Parsing Pipelines", link: "/en/guide/ocr-pipelines" },
          { text: "FAQ and Troubleshooting", link: "/en/guide/faq" },
        ],
      },
  {
    text: "Extended Docs",
    items: [{ text: "MCP Server PRD", link: "/mcp-server-prd" }],
  },
]

function zhThemeConfig() {
  return {
    nav: [
      { text: "首页", link: "/" },
      { text: "部署", link: "/guide/deployment" },
      { text: "架构", link: "/guide/architecture" },
      { text: "MCP", link: "/guide/mcp-integration" },
      { text: "OCR", link: "/guide/ocr-pipelines" },
      { text: "FAQ", link: "/guide/faq" },
    ],
    sidebar: {
      "/guide/": zhGuideSidebar,
      "/mcp-server-prd": zhGuideSidebar,
    },
    socialLinks: [{ icon: "github", link: repo }],
    search: {
      provider: "local",
      options: {
        translations: {
          button: {
            buttonText: "搜索文档",
            buttonAriaLabel: "搜索文档",
          },
          modal: {
            noResultsText: "没有找到结果",
            resetButtonTitle: "清除条件",
            footer: {
              selectText: "选择",
              navigateText: "切换",
              closeText: "关闭",
            },
          },
        },
      },
    },
    editLink: {
      pattern: `${repo}/edit/main/docs/:path`,
      text: "在 GitHub 上编辑此页",
    },
    docFooter: {
      prev: "上一页",
      next: "下一页",
    },
    outline: {
      label: "页面导航",
    },
    footer: {
      message: "MIT Licensed",
      copyright: "Copyright © 2026 PDF2PPT",
    },
  }
}

function enThemeConfig() {
  return {
    nav: [
      { text: "Home", link: "/en/" },
      { text: "Deployment", link: "/en/guide/deployment" },
      { text: "Architecture", link: "/en/guide/architecture" },
      { text: "MCP", link: "/en/guide/mcp-integration" },
      { text: "OCR", link: "/en/guide/ocr-pipelines" },
      { text: "FAQ", link: "/en/guide/faq" },
    ],
    sidebar: {
      "/en/guide/": enGuideSidebar,
      "/mcp-server-prd": enGuideSidebar,
    },
    socialLinks: [{ icon: "github", link: repo }],
    search: {
      provider: "local",
    },
    editLink: {
      pattern: `${repo}/edit/main/docs/:path`,
      text: "Edit this page on GitHub",
    },
    footer: {
      message: "MIT Licensed",
      copyright: "Copyright © 2026 PDF2PPT",
    },
  }
}

export default defineConfig({
  title: "PDF2PPT",
  description: "High-fidelity PDF-to-PPTX conversion docs",
  base,
  cleanUrls: true,
  themeConfig: zhThemeConfig(),
  locales: {
    root: {
      label: "简体中文",
      lang: "zh-CN",
      title: "PDF2PPT",
      description: "高保真 PDF 转 PPTX 文档站",
      themeConfig: zhThemeConfig(),
    },
    en: {
      label: "English",
      lang: "en-US",
      title: "PDF2PPT",
      description: "High-fidelity PDF-to-PPTX documentation",
      themeConfig: enThemeConfig(),
    },
  },
  head: [
    ["meta", { name: "theme-color", content: "#0f766e" }],
    ["meta", { property: "og:title", content: "PDF2PPT Docs" }],
    [
      "meta",
      {
        property: "og:description",
        content: "Documentation for PDF2PPT deployment, architecture, and OCR pipelines.",
      },
    ],
  ],
})
