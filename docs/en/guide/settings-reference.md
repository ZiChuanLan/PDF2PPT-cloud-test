# Settings Reference

This page explains the settings page in detail, including which fields are used daily and which ones belong to advanced tuning.

## Page Structure

The settings page is organized into three sections:

- `API / connection settings`
- `processing strategy`
- `recognition settings`

The UI automatically hides fields that are irrelevant to the current parsing route.

## 1. API / Connection Settings

This section contains backend addresses and cloud-service credentials.

### Backend API Origin

Default behavior:

- the browser auto-detects the API origin
- in most cases users do not need to change it

Useful only when:

- doing local debugging
- running under a special deployment layout
- working behind a reverse proxy with non-default routing

Actions:

- `apply address`
- `auto-detect`

### MinerU Options

When the current route uses `MinerU`, the page shows:

- `MinerU token`
- `MinerU base URL`
- `MinerU model version`
- `MinerU language`
- `enable formula recognition`
- `enable table recognition`
- `enable MinerU OCR`

Useful for:

- structurally complex documents
- tables and formulas
- cloud parsing workflows

## 2. Processing Strategy

This section controls how the final PPT output is assembled.

### Page Image Handling Mode

This directly affects how images are kept in the final PPT.

Options:

- `extract images as separate editable elements`
- `keep images inside the full-page background`

Practical guidance:

- use full-page background when you want the safest visual fidelity
- use extracted image blocks when you want more editability

### Text Erase Mode

Typical options:

- `fill` (recommended)
- `smart`

Recommendation:

- use `fill` for most cases
- only try `smart` for special pages

### OCR Render DPI

This only affects the rendered input used for OCR.

Impact:

- higher values may help with small text
- but they increase runtime and memory usage

Most users should leave it unchanged at first.

### Remove NotebookLM Footer

Useful when source documents contain a specific footer mark.  
This is not a general cleanup toggle.

### Background Clearing and Image Region Thresholds

This is a typical advanced section.

It includes:

- minimum clearing expansion
- maximum clearing expansion
- clearing expansion ratio
- minimum image-region area ratio
- maximum image-region area ratio
- maximum image-region aspect ratio

Only tune these when:

- background cleanup leaves visible artifacts
- image blocks are clearly misdetected
- small icons or image blocks are frequently missed

The page also provides:

- `restore default thresholds`

## 3. Recognition Settings

This section controls how OCR or parsing is executed.

### OCR Provider

Depending on the active route, users may see providers such as:

- `AIOCR`
- `local OCR (PaddleOCR)`
- `local OCR (Tesseract)`
- `Baidu OCR`

Users only need to configure the provider actually used by the selected route.

### AIOCR API Parameters

When AIOCR is active, common fields include:

- `OCR API key`
- `OCR base URL`
- `AIOCR vendor adapter`
- `AIOCR recognition chain`
- `layout model`
- `PaddleOCR-VL max side`
- `OCR model`
- `check OCR configuration`

### AIOCR Recognition Chain

This is one of the most important groups of settings.

Typical chains:

- `local layout-block OCR`
- `model-direct text and boxes`
- `built-in document parsing (PaddleOCR-VL)`

How to think about them:

- layout-block OCR is the safest default
- model-direct output depends more on the model and prompt discipline
- built-in document parsing is the structured PaddleOCR-VL path

### OCR Model

The model field supports:

- manual input
- choosing from discovered candidates

Candidates are filtered according to the selected chain.

Examples:

- `doc_parser` only shows PaddleOCR-VL-compatible models
- `direct` filters out unsuitable options

### OCR Configuration Check

Button:

- `check OCR configuration`

This verifies:

- whether the current model is reachable
- whether the selected chain returns usable OCR output
- whether obvious route-level failures occur

It is useful before launching real jobs.

### Prompt Experiments

This is an advanced feature.

Applies to:

- direct model OCR
- layout-block OCR

Users can tune:

- prompt preset
- current-chain prompt override
- image-region detection prompt override

Recommendation:

- start with built-in presets
- only override prompts when the current model regularly misses text, scrambles order, or leaks labels into the output

### Concurrency and Rate Limits

This is also advanced tuning.

Includes:

- page concurrency
- block concurrency
- RPM limit
- TPM limit
- retry count

Useful when:

- cloud OCR quotas matter
- the model times out often
- balancing speed and stability matters

### Baidu Parsing / Baidu OCR

When a Baidu route is active, fields include:

- `document parsing type`
- `Baidu API key`
- `Baidu secret key`
- `Baidu app ID` (optional)

For Baidu document parsing, the focus is structured output.  
For Baidu OCR, the focus is text recognition itself.

### Tesseract Settings

Typical fields:

- `minimum confidence`
- `language`

Useful for:

- fully local environments
- deployments that avoid external OCR providers

### Local OCR Full Check

This is an important diagnostics area.

It checks both:

- `Tesseract`
- `PaddleOCR`

And separates:

- runtime readiness
- model file completeness

If local OCR does not work, this section should be the first place to inspect.

## Daily Settings vs Advanced Settings

### Daily Most-Used Settings

- parse engine
- page image handling mode
- OCR provider / OCR route
- OCR API key / base URL / model
- remove NotebookLM footer

### Only For Tuning or Diagnostics

- API origin override
- text erase mode
- OCR render DPI
- background and image-region thresholds
- prompt experiments
- concurrency and rate limits
- local OCR full check

## Recommended Usage Order

1. choose the parse engine
2. fill in the credentials and model required by that route
3. run with default strategy first
4. only expand advanced settings when the result needs tuning
