# PDF2PPT

> 将扫描版 PDF、课件截图和图片型文档转换为尽量高保真、尽量可编辑的 PPTX。

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ZiChuanLan/PDF2PPT-cloud-test)
[![Deploy on Zeabur](https://zeabur.com/button.svg)](https://zeabur.com/templates/UKLIVV)

[中文](./README.md) | [English](./README_EN.md)

[文档站](https://zichuanlan.github.io/PDF2PPT-cloud-test/) · [快速开始](#快速开始) · [部署方式](#部署方式) · [License](#license)

`PDF2PPT` 是一个面向实际使用和部署的开源服务。  
它不是简单地把 PDF 每页导成一张图，而是尽量把页面重建为可编辑文本、独立图片区域和清理后的页面底图，再导出为 PowerPoint。

## 适合什么场景

- 扫描版 PDF，希望尽量保留原始视觉效果，同时恢复可编辑文字
- 课件截图、讲义截图、图片型报告，希望导出后还能继续改 PPT
- 想用 Web 界面直接上传转换，而不是只用一次性脚本
- 想部署成可复用的服务，供团队、自动化流程或其他系统调用

## 核心特点

- 高保真页面重建，而不只是截图式导出
- 支持本地 OCR、远程 OCR 和多种文档解析链路
- 提供 Web、API、Worker 的完整服务化结构
- 支持标准 Docker 部署、Hosted 单服务模式、Render 和 Zeabur

## 快速开始

### 本地开发

```bash
make dev-local
```

默认会启动：

- Web: `http://localhost:3000`
- API: `http://127.0.0.1:8000`

### 标准 Docker 部署

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

### Hosted 单服务后端

适合先在云平台验证后端：

```bash
cp .env.example .env
docker compose -f docker-compose.hosted.yml up -d --build
```

推荐两种方式：

- 最低成本验证：`REDIS_URL=memory://`
- 接入托管 Redis：`REDIS_URL=<your-redis-url>` 且 `EMBEDDED_WORKER_CONCURRENCY=1`

## 部署方式

| 模式 | 适合场景 | 入口 |
| --- | --- | --- |
| 本地开发 | 本机体验与调试 | `make dev-local` |
| 标准部署 | 长期运行或生产部署 | [`docker-compose.yml`](docker-compose.yml) |
| Hosted 单服务后端 | 云平台快速验证 | [`docker-compose.hosted.yml`](docker-compose.hosted.yml) |
| Render | 一键部署 | [`render.yaml`](render.yaml) |
| Zeabur | 模板化部署 | [`zeabur.template.yaml`](zeabur.template.yaml) |

## 怎么使用

1. 打开 Web 页面或调用 API 上传 PDF
2. 提交转换参数并创建任务
3. 等待任务完成后下载 `output.pptx`

首次运行如果想优先求稳，建议从 `remote_ocr + aiocr + fullpage` 开始。

## 文档

详细架构、OCR 链路、MCP 集成、部署细节和 FAQ 已迁移到文档站：

- [文档站首页](https://zichuanlan.github.io/PDF2PPT-cloud-test/)
- [部署指南](https://zichuanlan.github.io/PDF2PPT-cloud-test/guide/deployment)
- [架构说明](https://zichuanlan.github.io/PDF2PPT-cloud-test/guide/architecture)
- [MCP 集成](https://zichuanlan.github.io/PDF2PPT-cloud-test/guide/mcp-integration)
- [OCR 与解析链路](https://zichuanlan.github.io/PDF2PPT-cloud-test/guide/ocr-pipelines)
- [FAQ 与排障](https://zichuanlan.github.io/PDF2PPT-cloud-test/guide/faq)

## License

MIT. See [LICENSE](LICENSE).
