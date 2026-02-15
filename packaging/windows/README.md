# Windows EXE 与 GitHub Release

本目录提供 Windows 发行流程，目标是产出可直接上传到 GitHub Releases 的两个资产：

- `release/windows/PPT-OpenCode-Launcher.exe`
- `release/windows/ppt-opencode-win-x64.zip`

zip 内包含：

- 启动器 EXE
- `docker-compose.yml`
- `api/` 与 `web/` 运行代码
- 基础说明文档

推荐用户下载 zip 解压后，双击其中 EXE 启动完整服务栈（web + api + worker + redis）。

## 前提

- Windows 10/11
- Docker Desktop 已安装并已启动
- 命令行可用 `docker compose`
- Python 3.10+（构建 EXE 时需要）

## 本地构建

PowerShell:

```powershell
.\packaging\windows\build_exe.ps1
```

CMD:

```bat
packaging\windows\build_exe.bat
```

## 使用 EXE

默认双击或不带参数是 `start`。

```bat
PPT-OpenCode-Launcher.exe
PPT-OpenCode-Launcher.exe start
PPT-OpenCode-Launcher.exe stop
PPT-OpenCode-Launcher.exe restart
PPT-OpenCode-Launcher.exe status
PPT-OpenCode-Launcher.exe logs --lines 200
```

可选参数：

- `--timeout 300` 启动等待秒数
- `--skip-build` 启动时跳过镜像构建
- `--no-browser` 启动后不自动打开浏览器

## GitHub Actions 自动构建

仓库包含工作流：`.github/workflows/windows-release.yml`

- `workflow_dispatch`: 手动触发
- `push tags(v*)`: 打版本 tag 时构建
- `release published`: 发布 Release 时构建并自动附加 exe + zip

## 注意事项

- 该 EXE 是启动器，不会把 Docker 本体打进单文件。
- 单独下载 EXE 时，EXE 附近仍需有 `docker-compose.yml`、`api/`、`web/` 才能运行。
- 面向最终用户分发时，优先发布并引导下载 `ppt-opencode-win-x64.zip`。
- 首次运行会构建容器镜像，耗时较长且需要联网。
- 不要把真实 OCR API Key 提交到仓库；建议在 UI 设置页或本机环境变量配置。
