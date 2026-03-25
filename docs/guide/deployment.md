# 部署指南

## 部署模式

| 模式 | 适合场景 | 入口文件 |
| --- | --- | --- |
| 本地开发 | 本机联调、快速改动 | `make dev-local` |
| 标准部署 | 长期运行或生产部署 | [`docker-compose.yml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/docker-compose.yml) |
| Hosted 单服务后端 | 云平台快速验证后端 | [`docker-compose.hosted.yml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/docker-compose.hosted.yml) |
| Docs 文档站 | 独立运行文档站 | [`docker-compose.docs.yml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/docker-compose.docs.yml) |
| Render | 一键部署 | [`render.yaml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/render.yaml) |
| Zeabur | 模板化部署 | [`zeabur.template.yaml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/zeabur.template.yaml) |

## 本地开发

```bash
make dev-local
```

默认会启动：

- Web: `http://localhost:3000`
- API: `http://127.0.0.1:8000`

## 标准 Docker 部署

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

默认行为：

- `web` 对外暴露 `${WEB_PORT}`，默认 `3000`
- `api` 默认只绑定 `${API_BIND_HOST:-127.0.0.1}:${API_PORT:-8000}`
- `redis` 不直接对外暴露
- 浏览器默认通过 Web 同源访问 `/api/*`

## Hosted 单服务后端

```bash
cp .env.example .env
docker compose -f docker-compose.hosted.yml up -d --build
```

推荐两种方式：

- 最低成本验证：`REDIS_URL=memory://`
- 接入托管 Redis：`REDIS_URL=<your-redis-url>` 且 `EMBEDDED_WORKER_CONCURRENCY=1`

容器启动命令使用的是 [`api/scripts/run_hosted.sh`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/api/scripts/run_hosted.sh)。

## Docs 文档站 Docker 运行

```bash
docker compose -f docker-compose.docs.yml up -d --build
```

默认访问地址：

- Docs: `http://localhost:4173`

如果需要修改端口，可设置：

```bash
DOCS_PORT=8080 docker compose -f docker-compose.docs.yml up -d --build
```

## GitHub Pages 自动部署

仓库已经包含 GitHub Pages 工作流：

- [`.github/workflows/deploy-docs.yml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/.github/workflows/deploy-docs.yml)

启用方式：

1. 把代码推到 GitHub 仓库
2. 进入仓库 `Settings -> Pages`
3. 在 `Build and deployment` 中把 `Source` 设为 `GitHub Actions`
4. 之后每次 push 到 `main`，文档站都会自动构建并发布

默认发布地址通常是：

```text
https://<your-github-username>.github.io/PDF2PPT/
```

当前 VitePress 配置已经自动适配 GitHub Pages 的仓库子路径。

如果将来你改成独立域名，可以通过环境变量覆盖：

```bash
DOCS_BASE=/ npm run docs:build
```

## 云平台部署

### Render

- Blueprint 文件：[`render.yaml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/render.yaml)
- 默认会创建 `pdf2ppt-api`、`pdf2ppt-web` 和 `pdf2ppt-redis`

### Zeabur

- 模板文件：[`zeabur.template.yaml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/zeabur.template.yaml)
- 模板页：<https://zeabur.com/templates/UKLIVV>

## 访问控制

项目默认有两层访问边界：

- `WEB_ACCESS_PASSWORD`
  保护 Web 页面和同源 `/api/*`
- `API_BEARER_TOKEN`
  保护直接访问 `/api/v1/*` 的客户端

上线前至少应处理这两件事：

- 把默认的 `WEB_ACCESS_PASSWORD=123456` 改成强密码
- 如果需要对外直连 API，开启 `API_BEARER_TOKEN`

## 关键环境变量

完整变量说明见 [`.env.example`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/.env.example)。

| 变量 | 用途 |
| --- | --- |
| `WEB_PORT` | Web 暴露端口，默认 `3000` |
| `API_BIND_HOST` | API 绑定地址，默认 `127.0.0.1` |
| `API_PORT` | API 暴露端口，默认 `8000` |
| `REDIS_URL` | Redis 连接地址；Hosted 模式可用 `memory://` |
| `WEB_ACCESS_PASSWORD` | Web 访问密码 |
| `API_BEARER_TOKEN` | 直连 API 的 Bearer Token |
| `SILICONFLOW_API_KEY` | 默认远程 OCR API Key |
| `SILICONFLOW_BASE_URL` | 默认远程 OCR 网关 |
| `SILICONFLOW_MODEL` | 默认远程 OCR 模型 |
| `JOB_TTL_MINUTES` | 任务与产物保留时长 |
| `OCR_PADDLE_LAYOUT_PREWARM` | 启动时预热版面模型 |
| `OCR_PADDLE_VL_PREWARM` | 启动时预热 PaddleOCR-VL |
