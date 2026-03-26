# PDF2PPT

<p align="center">
  <img src="https://i.postimg.cc/44W75HTp/shou-ye.png" alt="PDF2PPT 首页" width="100%" />
</p>

> 将扫描版 PDF、课件截图和图片型文档转换为尽量高保真、尽量可编辑的 PPTX。

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ZiChuanLan/PDF2PPT)
[![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/templates/UKLIVV)

[中文](./README.md) | [English](./README_EN.md)

[文档站](https://zichuanlan.github.io/PDF2PPT/) · [快速开始](#快速开始) · [部署选项](#部署选项) · [License](#license)

演示站点：<https://ppt.015201314.xyz/>  
访问密码：`lanPDF2PPT2026!`

`PDF2PPT` 是一个面向实际使用和部署的开源服务。  
它不是简单地把 PDF 每页导成一张图，而是尽量把页面重建为可编辑文本、独立图片区域和清理后的页面底图，再导出为 PowerPoint。

## 为什么用它

- 不只是截图式导出，而是尽量把页面重建为文本层、图片块和页面底图
- 不只适合单机试用，也适合部署成 Web、API、Worker 的完整服务
- 不绑定单一路线，可以在本地 OCR、远程 OCR 和文档解析链路之间切换

## 适合谁

- 需要处理扫描版 PDF、课件截图、图片型报告的个人用户
- 想把 PDF 转 PPT 能力做成内部服务的团队或开发者
- 希望把转换流程接入自动化、脚本或 MCP 客户端的集成方

## 核心能力

- 高保真页面重建，优先兼顾原稿观感与可编辑性
- 支持本地 OCR、远程 OCR、百度文档解析、MinerU 等多种链路
- 提供 Web 上传、任务跟踪、结果下载和 API 接入
- 支持标准 Docker 部署、Hosted 单服务模式、Render 和 Zeabur

## 界面预览

<p align="center">
  <img src="https://i.postimg.cc/CMcBNnVq/she-zhi-aiocr.png" alt="PDF2PPT AI OCR 设置页" width="49%" />
  <img src="https://i.postimg.cc/pVshZ5t5/gen-zong-ye-mian.png" alt="PDF2PPT 跟踪页面" width="49%" />
</p>

## 快速开始

### 本地体验

```bash
make dev-local
```

默认会启动：

- Web: `http://localhost:3000`
- API: `http://127.0.0.1:8000`

### 完整部署

适合完整部署 `web + api + worker + redis`：

```bash
cp .env.example .env
docker compose up -d --build
```

常用检查命令：

```bash
docker compose ps
docker compose logs -f
curl http://127.0.0.1:8000/health
```

### 轻量托管后端

适合先在云平台验证后端：

```bash
cp .env.example .env
docker compose -f docker-compose.hosted.yml up -d --build
```

推荐两种方式：

- 最低成本验证：`REDIS_URL=memory://`
- 接入托管 Redis：`REDIS_URL=<your-redis-url>` 且 `EMBEDDED_WORKER_CONCURRENCY=1`

## 基本使用流程

1. 上传 PDF
2. 提交转换参数并创建任务
3. 等待任务完成后下载 `output.pptx`

首次运行如果想优先求稳，建议从 `remote_ocr + aiocr + fullpage` 开始。

## 部署选项

| 模式 | 适合场景 | 入口 |
| --- | --- | --- |
| 本地体验 | 本机快速试用与调试 | `make dev-local` |
| 完整部署 | 长期运行或生产部署 | [`docker-compose.yml`](docker-compose.yml) |
| 轻量托管后端 | 云平台快速验证后端 | [`docker-compose.hosted.yml`](docker-compose.hosted.yml) |
| Render | 一键部署 | [`render.yaml`](render.yaml) |
| Zeabur | 模板化部署 | [`zeabur.template.yaml`](zeabur.template.yaml) |

## 文档

更详细的架构、OCR 链路、MCP 集成、部署细节和 FAQ 已迁移到文档站：

- [文档站首页](https://zichuanlan.github.io/PDF2PPT/)
- [部署指南](https://zichuanlan.github.io/PDF2PPT/guide/deployment)
- [MCP 集成](https://zichuanlan.github.io/PDF2PPT/guide/mcp-integration)
- [FAQ 与排障](https://zichuanlan.github.io/PDF2PPT/guide/faq)

## License

MIT. See [LICENSE](LICENSE).
