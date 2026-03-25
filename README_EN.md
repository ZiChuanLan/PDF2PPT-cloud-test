# PDF2PPT

> Convert scanned PDFs, slide screenshots, and image-heavy documents into high-fidelity, as-editable-as-possible PPTX files.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ZiChuanLan/PDF2PPT)
[![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/templates/UKLIVV)

[中文](./README.md) | [English](./README_EN.md)

[Docs Site](https://zichuanlan.github.io/PDF2PPT/) · [Quick Start](#quick-start) · [Deployment Modes](#deployment-modes) · [License](#license)

`PDF2PPT` is an open-source service built for real usage and deployment.  
Instead of flattening every PDF page into a single image, it tries to rebuild pages into editable text, separated image regions, and cleaned backgrounds before exporting to PowerPoint.

## Best Fit

- Scanned PDFs where you want to preserve the original look while recovering editable text
- Screenshot-based slide decks and image-heavy reports that still need editing in PPT
- Teams that want a Web UI instead of a one-off conversion script
- Deployments that need a reusable service for internal tools, automation, or API integration

## Core Highlights

- High-fidelity page reconstruction instead of pure screenshot export
- Support for local OCR, remote OCR, and multiple document parsing pipelines
- Full service architecture with Web, API, and Worker
- Standard Docker deployment, hosted single-service mode, Render, and Zeabur support

## Quick Start

### Local Development

```bash
make dev-local
```

By default it starts:

- Web: `http://localhost:3000`
- API: `http://127.0.0.1:8000`

### Standard Docker Deployment

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

### Hosted Single-Service Backend

For quick backend validation on hosted platforms:

```bash
cp .env.example .env
docker compose -f docker-compose.hosted.yml up -d --build
```

Recommended variants:

- Lowest-friction validation: `REDIS_URL=memory://`
- Hosted Redis: `REDIS_URL=<your-redis-url>` and `EMBEDDED_WORKER_CONCURRENCY=1`

## Deployment Modes

| Mode | Best for | Entry |
| --- | --- | --- |
| Local development | Local testing and iteration | `make dev-local` |
| Standard deployment | Long-running or production setups | [`docker-compose.yml`](docker-compose.yml) |
| Hosted single-service backend | Fast hosted validation | [`docker-compose.hosted.yml`](docker-compose.hosted.yml) |
| Render | One-click deployment | [`render.yaml`](render.yaml) |
| Zeabur | Template-based deployment | [`zeabur.template.yaml`](zeabur.template.yaml) |

## How To Use

1. Upload a PDF from the Web UI or through the API
2. Submit conversion parameters and create a job
3. Download `output.pptx` after the job finishes

For a conservative first run, start with `remote_ocr + aiocr + fullpage`.

## Docs

Detailed architecture, OCR pipelines, MCP integration, deployment details, and FAQ are now available in the docs site:

- [Docs Home](https://zichuanlan.github.io/PDF2PPT/en/)
- [Deployment Guide](https://zichuanlan.github.io/PDF2PPT/en/guide/deployment)
- [Architecture](https://zichuanlan.github.io/PDF2PPT/en/guide/architecture)
- [MCP Integration](https://zichuanlan.github.io/PDF2PPT/en/guide/mcp-integration)
- [OCR and Parsing Pipelines](https://zichuanlan.github.io/PDF2PPT/en/guide/ocr-pipelines)
- [FAQ and Troubleshooting](https://zichuanlan.github.io/PDF2PPT/en/guide/faq)

## License

MIT. See [LICENSE](LICENSE).
