# Deployment Guide

## Deployment Modes

| Mode | Best for | Entry |
| --- | --- | --- |
| Local development | Fast local iteration | `make dev-local` |
| Standard deployment | Long-running or production setups | [`docker-compose.yml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/docker-compose.yml) |
| Hosted single-service backend | Fast backend validation on hosted platforms | [`docker-compose.hosted.yml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/docker-compose.hosted.yml) |
| Docs site | Standalone documentation site | [`docker-compose.docs.yml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/docker-compose.docs.yml) |
| Render | One-click deployment | [`render.yaml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/render.yaml) |
| Zeabur | Template-based deployment | [`zeabur.template.yaml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/zeabur.template.yaml) |

## Local Development

```bash
make dev-local
```

By default it starts:

- Web: `http://localhost:3000`
- API: `http://127.0.0.1:8000`

## Standard Docker Deployment

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

Default behavior:

- `web` is exposed on `${WEB_PORT}`, default `3000`
- `api` binds to `${API_BIND_HOST:-127.0.0.1}:${API_PORT:-8000}` by default
- `redis` is not publicly exposed
- Browsers access `/api/*` through the Web app by default

## Hosted Single-Service Backend

```bash
cp .env.example .env
docker compose -f docker-compose.hosted.yml up -d --build
```

Recommended variants:

- Lowest-friction validation: `REDIS_URL=memory://`
- Hosted Redis: `REDIS_URL=<your-redis-url>` and `EMBEDDED_WORKER_CONCURRENCY=1`

The container entrypoint is [`api/scripts/run_hosted.sh`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/api/scripts/run_hosted.sh).

## Running the Docs Site with Docker

```bash
docker compose -f docker-compose.docs.yml up -d --build
```

Default URL:

- Docs: `http://localhost:4173`

To change the port:

```bash
DOCS_PORT=8080 docker compose -f docker-compose.docs.yml up -d --build
```

## GitHub Pages Auto Deployment

The repository already includes a GitHub Pages workflow:

- [`.github/workflows/deploy-docs.yml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/.github/workflows/deploy-docs.yml)

To enable it:

1. Push the code to GitHub
2. Open `Settings -> Pages` in the repository
3. Set `Build and deployment -> Source` to `GitHub Actions`
4. After that, every push to `main` will automatically build and publish the docs site

The default URL is typically:

```text
https://<your-github-username>.github.io/PDF2PPT/
```

The current VitePress config already auto-adapts to the repository subpath used by GitHub Pages.

If you later switch to a custom domain, you can override the base with:

```bash
DOCS_BASE=/ npm run docs:build
```

## Cloud Deployment

### Render

- Blueprint: [`render.yaml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/render.yaml)
- Creates `pdf2ppt-api`, `pdf2ppt-web`, and `pdf2ppt-redis` by default

### Zeabur

- Template file: [`zeabur.template.yaml`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/zeabur.template.yaml)
- Template page: <https://zeabur.com/templates/UKLIVV>

## Access Control

The project uses two access layers by default:

- `WEB_ACCESS_PASSWORD`
  Protects the Web UI and same-origin `/api/*`
- `API_BEARER_TOKEN`
  Protects direct `/api/v1/*` API access

Before exposing it beyond local testing:

- Change the default `WEB_ACCESS_PASSWORD=123456`
- Enable `API_BEARER_TOKEN` if direct external API access is needed

## Key Environment Variables

See [`.env.example`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/.env.example) for the full list.

| Variable | Purpose |
| --- | --- |
| `WEB_PORT` | Web port, default `3000` |
| `API_BIND_HOST` | API bind address, default `127.0.0.1` |
| `API_PORT` | API port, default `8000` |
| `REDIS_URL` | Redis connection URL; hosted mode can use `memory://` |
| `WEB_ACCESS_PASSWORD` | Web access password |
| `API_BEARER_TOKEN` | Bearer token for direct API access |
| `SILICONFLOW_API_KEY` | Default remote OCR API key |
| `SILICONFLOW_BASE_URL` | Default remote OCR gateway |
| `SILICONFLOW_MODEL` | Default remote OCR model |
| `JOB_TTL_MINUTES` | Retention time for jobs and artifacts |
| `OCR_PADDLE_LAYOUT_PREWARM` | Prewarm the layout model at startup |
| `OCR_PADDLE_VL_PREWARM` | Prewarm PaddleOCR-VL at startup |
