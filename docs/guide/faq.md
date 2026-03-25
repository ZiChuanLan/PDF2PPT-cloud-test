# FAQ 与排障

## 常见问题

### 它和“每页导成一张图”的工具有什么区别？

核心区别是输出目标不同。  
这类工具主要追求“看起来像”，而 `PDF2PPT` 更强调在保留原稿视觉效果的同时，把文本和部分图片区域尽量重建成可继续编辑的对象。

### 是否保证导出的 PPT 完全可编辑？

不保证。  
扫描质量、页面复杂度、OCR 识别效果、图片区域检测结果都会影响最终可编辑程度。更稳的策略通常是优先保证视觉还原，再逐步提高可编辑性。

### 一定要配置远程 OCR API Key 吗？

不一定。  
仓库支持本地和远程多种 OCR/解析链路，但如果你希望直接使用远程 OCR 作为默认能力，就需要在 [`.env.example`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/.env.example) 对应的环境变量基础上补齐配置。

### 它更适合本地用，还是部署成服务？

两者都支持。  
如果你只是自己试用，本地开发或单机 Docker 已经足够；如果你想给团队或其他系统使用，标准 Docker 部署会更合适。

## 常用排障入口

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f worker
docker compose logs -f web
```

## 排查方向

- OCR 卡住：检查 worker 日志、任务状态和 OCR 调试产物
- 识别正确但导出成图片：检查 `image_regions` 与扫描页重建策略
- 首次启动慢：检查 Paddle 相关模型预热与缓存下载
- Web 能访问但 API 不通：检查 `NEXT_PUBLIC_API_URL`、`INTERNAL_API_ORIGIN` 与健康检查
