# pdf2ppt

将 PDF（尤其是扫描版/图片版）转换为**高保真、可编辑**的 PPTX。

核心目标：
- 生成后的 PPT（文字位置/字号/换行/图片区域）尽量与原图一致
- OCR 可切换不同远程/本地引擎，但统一走同一条合成管线
- 前端默认保持简单，高级选项折叠

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

## Windows 可下载版（EXE + Release 包）

如果你希望用户在 GitHub 上下载后直接运行，请使用 Windows 打包流程：

- 文档：`packaging/windows/README.md`
- 构建脚本：`packaging/windows/build_exe.ps1` / `packaging/windows/build_exe.bat`
- 自动化工作流：`.github/workflows/windows-release.yml`

构建后会得到：

- `release/windows/PPT-OpenCode-Launcher.exe`
- `release/windows/ppt-opencode-win-x64.zip`

说明：
- `PPT-OpenCode-Launcher.exe` 是启动器
- 对最终用户分发，优先使用 `ppt-opencode-win-x64.zip`（包含 EXE + 运行所需目录）

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

## 扫描页合成模式（关键）

设置项：`scanned_page_mode`

- `segmented`（分块）：尽量把截图/图表等区域裁为可编辑图片块，文字仍可编辑
- `fullpage`（全页）：整页作为背景图，仅覆盖可编辑文字，通常最接近原图（图片不可单独编辑）

## 项目结构

- `api/`：FastAPI 接口、任务队列、PDF 解析、OCR 和 PPTX 生成
- `web/`：Next.js 前端，负责上传、运行配置、结果跟踪和设置页
- `scripts/dev/`：本地开发辅助脚本
- `packaging/windows/`：Windows 启动器与发布打包脚本

说明：
- 公开仓库默认不保留测试样本、截图对比产物和临时基准脚本
- OCR 的真实密钥、样本 PDF、运行缓存也不应提交
