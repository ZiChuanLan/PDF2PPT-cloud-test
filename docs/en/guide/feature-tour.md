# Feature Tour

This page explains `PDF2PPT` by the actual screens users see, not by backend modules.

## Home Page

The home page is the main workbench. It is responsible for:

- uploading PDFs or single images
- previewing the current file
- choosing page ranges
- starting conversion tasks
- showing current task status and recent jobs

### Upload and Preview

The main panel on the home page is the upload and preview area.

You can:

- drag and drop a PDF or image
- click to choose a file
- preview the current PDF page or image
- clear the current file and start over

Supported upload types:

- `PDF`
- `PNG`
- `JPG / JPEG`
- `WEBP`

Image inputs are automatically wrapped into a single-page PDF before entering the existing pipeline, so page-range controls are disabled for image mode.

### Preview Area

The preview area supports:

- current page input
- previous / next page navigation
- reading total PDF page count
- direct single-image preview

It is designed to match the same page logic used by single-page trial runs.

### Page Range

For PDFs, the home page allows page-range selection.

Typical usage:

- process the full document
- manually set start and end page
- use `single-page trial run (current page)`

The frontend blocks invalid ranges before submission.

### Common Execution Options

The home page keeps only a few high-frequency execution options:

- `retain process images`
  useful for intermediate validation and troubleshooting
- `PPT generation mode`
  choose between speed and fidelity

### Current Configuration Card

The right side shows a summary of the current runtime configuration, including:

- active parse route
- active OCR route
- active PPT generation mode
- a shortcut into the settings page

This panel is intentionally compact. Detailed tuning remains in the settings page.

### Current Task Status

The right side also shows the current task summary:

- current status
- current stage
- simple progress bar
- queue total / running / completed / failed counts

### Recent Jobs

The bottom section of the home page shows recent jobs for quick reference:

- status
- stage
- progress
- creation time

For page-by-page validation, users should move to the tracking page.

## Tracking Page

The tracking page is for job progress, result preview, and before/after comparison.

### Job List

The left-side job list supports:

- searching by job ID or stage
- filtering by status
- viewing queue size
- selecting one job to inspect in the right panel

Each job row shows:

- short job ID
- status
- current stage
- progress
- queue position or queue state
- recent error / summary message

Common actions:

- `track`
- `download`
- `delete`

### Result Preview

The right-side result panel supports:

- per-page preview
- before/after comparison
- viewing job logs
- reviewing the source PDF

If visual artifacts are retained, available pages are shown automatically.

### Per-Page Preview

Useful when checking one page in detail. Typical visuals may include:

- original page render
- cleaned background
- final preview image
- OCR overlay
- layout assist before/after images

### Before/After Comparison

Useful when checking:

- whether cleanup removed too much
- whether text replacement shifted
- whether image splitting is reasonable
- whether final output still matches the source visually

## Settings Page

The settings page is not meant to dump all parameters at once.

Its design is:

- common settings are visible by default
- advanced settings are collapsed by default
- only relevant fields appear for the currently selected route

The main sections are:

- `API / connection settings`
- `processing strategy`
- `recognition settings`

It also provides sticky section navigation and current-section highlighting.

### Parse Engine

At the top of the settings page, users first choose the parse engine.

Current options include:

- Baidu parsing
- AIOCR
- classic OCR
- cloud MinerU

The visible settings below change according to this selection.

### Advanced Parameters and Diagnostics

There is one shared `advanced parameters and diagnostics` toggle near the top.

When expanded, users can access:

- API origin override
- deeper OCR tuning
- prompt experiments
- concurrency and rate limits
- local OCR checks
- threshold and expansion settings

## Recommended Reading Order

For first-time users:

1. use the home page to upload, preview, and start tasks
2. use the tracking page to validate results
3. only open the settings page when switching routes, adding credentials, or tuning quality
