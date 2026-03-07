# PDF2PPT

`PDF2PPT` 用来把 PDF，尤其是扫描版、图片版和课件截图类文档，转换成**尽量高保真、尽量可编辑**的 PPTX。

项目重点：
- 尽量保留原稿的文字位置、字号、换行和图片区块
- OCR 可切换远程或本地引擎，但统一走同一条合成管线
- 部署配置尽量收敛，默认值尽量放在代码里而不是堆在 compose 里

## 快速启动（本地开发）

```bash
make dev-local
```

会自动启动：
- API: `http://127.0.0.1:8000`（若端口占用会自动换 8001）
- Web: `http://localhost:3000`

也可以直接运行：

```bash
bash scripts/dev/local_dev.sh
```

## VPS / Docker 最小配置

常规部署只需要关心下面几项：

```env
SILICONFLOW_API_KEY=你的key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=PaddlePaddle/PaddleOCR-VL-1.5
OCR_PADDLE_VL_PREWARM=1
OCR_PADDLE_VL_PREWARM_TARGET=worker
OCR_PADDLE_VL_DOCPARSER_MAX_SIDE_PX=2200
```

说明：
- 本地部署现在默认只会在容器启动时预拉 `PP-DocLayoutV3` 到缓存卷里：`OCR_PADDLE_LAYOUT_PREWARM=1`。这样首个 PaddleOCR-VL 相关任务不会再承担这个本地 layout 模型下载。
- `OCR_PADDLE_VL_PREWARM=1` 用于容器启动时预热 PaddleOCR-VL，避免第一个请求承担冷启动
- `OCR_PADDLE_VL_PREWARM` 仍然需要启动阶段已经有可用的远程 API key；如果 key 只在 Web 提交任务时才提供，这一项会跳过，但本地模型预热仍可执行
- `OCR_PADDLE_VL_DOCPARSER_MAX_SIDE_PX` 是目前仍建议保留的公开调节项
- 其他 PaddleOCR-VL 超时、重试、并发等细粒度参数默认走代码内置值，除非你在排障，否则不需要管

### 生产部署命令

默认的 `docker-compose.yml` 就是部署版，可以直接走标准命令。

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 至少补齐 `.env` 里的这些值：

```env
SILICONFLOW_API_KEY=你的key
SILICONFLOW_MODEL=PaddlePaddle/PaddleOCR-VL-1.5
OCR_PADDLE_VL_PREWARM=1
OCR_PADDLE_VL_DOCPARSER_MAX_SIDE_PX=2200
```

3. 直接启动：

```bash
docker compose up -d --build
```

生产编排默认行为：
- `web` 对外暴露 `${WEB_PORT}`，默认 `3000`
- `api` 只绑定到宿主机 `127.0.0.1:${API_PORT}`，默认 `8000`
- `redis` 不对外暴露
- 任务结果和模型缓存走 named volume：`api-data`、`paddlex-cache`、`paddle-cache`
- 前端默认走同源 `/health` 和 `/api/*`，再由 Next 反代到容器内 `api:8000`

如果你有自己的域名反代，直接把外部流量转到 `WEB_PORT` 即可，不需要单独公开 `api`。

### 同源与跨域配置

默认推荐：
- 保持 `NEXT_PUBLIC_API_URL=` 为空
- 保持 `INTERNAL_API_ORIGIN=http://api:8000`
- 让浏览器只访问 Web，同源 `/health` 和 `/api/v1/*` 由 Next 转发到后端

只有在你明确要把 API 暴露为另一个公网地址时，才需要：
- 设置 `NEXT_PUBLIC_API_URL=https://你的-api-域名`
- 更新 `CORS_ALLOW_ORIGINS`
- 重新构建 Web 镜像，因为 `NEXT_PUBLIC_*` 变量会进入前端构建产物

### 常用运维命令

```bash
docker compose ps
docker compose logs -f
docker compose down
curl http://127.0.0.1:8000/health
```

### 开发态 Compose

如果你还想保留源码挂载和 `next dev` 那套开发容器，改用：

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

## 远程 OCR（推荐）

远程 OCR 走 OpenAI-Compatible 接口（例如 SiliconFlow / PPIO / Novita / OpenAI / DeepSeek 网关）。

建议通过以下方式之一配置：
- 在前端“设置页”填写 OCR 的 `API Key / Base URL / Model`
- 或在后端环境变量 / `.env` 中设置（示例见 `.env.example`）

注意：不要把真实 Key 提交到仓库或发到公开渠道。

### 常用模型示例

- DeepSeek 专用 OCR：`Pro/deepseek-ai/deepseek-ocr`
- PaddleOCR-VL：`PaddlePaddle/PaddleOCR-VL-1.5`
- 通用 VL 也可尝试 OCR：`Qwen/Qwen2.5-VL-72B-Instruct`（效果取决于模型与 prompt）

## 扫描页图片处理方式（关键）

设置项：`scanned_page_mode`

- `segmented`（图片拆出来）：尽量把截图/图表等区域裁为独立图片对象，方便在 PPT 里单独编辑
- `fullpage`（留在整页背景里）：整页作为背景图，仅覆盖可编辑文字，通常最接近原图

## 项目结构

- `api/`：FastAPI 接口、任务队列、PDF 解析、OCR 和 PPTX 生成
- `web/`：Next.js 前端，负责上传、运行配置、结果跟踪和设置页
- `scripts/dev/`：本地开发辅助脚本

说明：
- 公开仓库默认不保留测试样本、截图对比产物和临时基准脚本
- OCR 的真实密钥、样本 PDF、运行缓存也不应提交
