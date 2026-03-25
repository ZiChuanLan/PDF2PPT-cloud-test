# FAQ and Troubleshooting

## FAQ

### How is this different from tools that export each page as one image?

The output goal is different.  
Those tools mainly optimize for looking similar. `PDF2PPT` tries to preserve the original appearance while rebuilding text and some image regions into objects that remain editable in PowerPoint.

### Does it guarantee a fully editable PPT?

No.  
The final level of editability depends on scan quality, page complexity, OCR quality, and image-region detection. In practice, the safer strategy is to preserve visual fidelity first, then improve editability where possible.

### Do I have to configure a remote OCR API key?

Not necessarily.  
The project supports both local and remote OCR/parsing pipelines. But if you want remote OCR as the default path, you need to fill the corresponding variables from [`.env.example`](https://github.com/ZiChuanLan/PDF2PPT/blob/main/.env.example).

### Is it better for local use or service deployment?

Both are supported.  
For personal usage, local development or single-machine Docker is enough. For shared use across a team or other systems, the standard Docker deployment is the better fit.

## Useful Troubleshooting Commands

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f worker
docker compose logs -f web
```

## Troubleshooting Directions

- OCR gets stuck: inspect worker logs, job status, and OCR debug artifacts
- OCR is correct but part of the output becomes an image: inspect `image_regions` and scanned-page reconstruction strategy
- First startup is slow: check Paddle model prewarm and cache download behavior
- Web works but API does not: check `NEXT_PUBLIC_API_URL`, `INTERNAL_API_ORIGIN`, and health checks
