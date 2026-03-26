
<p align="center">
  <img src="./assets/branding/banner.svg" alt="PDF2PPT banner" width="100%" />
</p>

> Convert scanned PDFs, slide screenshots, and image-heavy documents into high-fidelity, as-editable-as-possible PPTX files.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ZiChuanLan/PDF2PPT)
[![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/templates/UKLIVV)

[中文](./README.md) | [English](./README_EN.md)

[Docs Site](https://zichuanlan.github.io/PDF2PPT/) · [Live Demo](#live-demo) · [Quick Start](#quick-start) · [Deployment Options](#deployment-options) · [License](#license)

`PDF2PPT` is an open-source service built for real usage and deployment.  
Instead of flattening every PDF page into a single image, it tries to rebuild pages into editable text, separated image regions, and cleaned backgrounds before exporting to PowerPoint.

## Live Demo

| Item | Address / Info |
| --- | --- |
| Demo site | <https://ppt.015201314.xyz/> |
| Access password | `lanPDF2PPT2026!` |



## Interface Preview

| Tracking Page | Job History |
| --- | --- |
| ![PDF2PPT tracking page](https://i.postimg.cc/pVshZ5t5/gen-zong-ye-mian.png) | ![PDF2PPT job history](https://i.postimg.cc/3rt4B0Hy/ren-wu-ji-lu.png) |

| AIOCR Settings | MinerU Settings |
| --- | --- |
| ![PDF2PPT AIOCR settings](https://i.postimg.cc/CMcBNnVq/she-zhi-aiocr.png) | ![PDF2PPT MinerU settings](https://i.postimg.cc/gcgXDLPR/she-zhi-yun-duanmineru.png) |
## Overview

When handling scanned PDFs, slide screenshots, and image-heavy reports, the common outcome is often just a full-page image with almost no editability.  
`PDF2PPT` is not trying to simply paste PDFs into PowerPoint. Its goal is to rebuild pages into editable text, separated image regions, and page backgrounds so the final PPTX stays closer to the source while remaining more usable.

| Common problem | How PDF2PPT addresses it |
| --- | --- |
| Each page becomes one big image | It tries to separate text, image blocks, and page background |
| OCR is treated as secondary | OCR and page reconstruction are core parts of the workflow |
| Only suitable for local experimentation | It supports Web, API, Worker, and standard deployment paths |
| One fixed route for every document | It can switch across local OCR, remote OCR, and parsing pipelines |

## Why It Exists

- It aims for page reconstruction, not just screenshot-style export
- It is designed for real deployment, not only one-off local conversion
- It supports multiple OCR and parsing paths instead of one fixed route

## Who It Is For

- Individual users working with scanned PDFs, slide screenshots, and image-heavy reports
- Teams that want to turn PDF-to-PPT into an internal service
- Integrators who want to expose the workflow through APIs, automation, or MCP clients

## Core Capabilities

- High-fidelity page reconstruction with editability as a first-class goal
- Support for local OCR, remote OCR, Baidu document parsing, MinerU, and related routes
- Web upload, job tracking, result download, and API-based integration
- Standard Docker deployment, hosted single-service mode, Render, and Zeabur support
## Quick Start

### Local Tryout

```bash
make dev-local
```

By default it starts:

- Web: `http://localhost:3000`
- API: `http://127.0.0.1:8000`

### Full Deployment

For the full `web + api + worker + redis` stack:

```bash
cp .env.example .env
docker compose up -d --build
```

Useful checks:

```bash
docker compose ps
docker compose logs -f
curl http://127.0.0.1:8000/health
```

### Lightweight Hosted Backend

For quick backend validation on hosted platforms:

```bash
cp .env.example .env
docker compose -f docker-compose.hosted.yml up -d --build
```

Recommended variants:

- Lowest-friction validation: `REDIS_URL=memory://`
- Hosted Redis: `REDIS_URL=<your-redis-url>` and `EMBEDDED_WORKER_CONCURRENCY=1`

## Basic Flow

1. Upload a PDF
2. Submit conversion parameters and create a job
3. Download `output.pptx` after the job finishes

For a conservative first run, start with `remote_ocr + aiocr + fullpage`.

## Deployment Options

| Mode | Best for | Entry |
| --- | --- | --- |
| Local tryout | Fast local testing and iteration | `make dev-local` |
| Full deployment | Long-running or production setups | [`docker-compose.yml`](docker-compose.yml) |
| Lightweight hosted backend | Fast hosted validation | [`docker-compose.hosted.yml`](docker-compose.hosted.yml) |
| Render | One-click deployment | [`render.yaml`](render.yaml) |
| Zeabur | Template-based deployment | [`zeabur.template.yaml`](zeabur.template.yaml) |

## Docs

Detailed architecture, OCR pipelines, MCP integration, deployment details, and FAQ are available in the docs site:

- [Docs Home](https://zichuanlan.github.io/PDF2PPT/en/)
- [Deployment Guide](https://zichuanlan.github.io/PDF2PPT/en/guide/deployment)
- [MCP Integration](https://zichuanlan.github.io/PDF2PPT/en/guide/mcp-integration)
- [FAQ and Troubleshooting](https://zichuanlan.github.io/PDF2PPT/en/guide/faq)

## License

MIT. See [LICENSE](LICENSE).
