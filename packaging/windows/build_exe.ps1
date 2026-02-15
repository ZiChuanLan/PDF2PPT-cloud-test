$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

Write-Host "[1/5] Installing build dependencies..."
python -m pip install --upgrade pip pyinstaller

Write-Host "[2/5] Building launcher exe..."
python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --name "PPT-OpenCode-Launcher" `
  packaging/windows/app_launcher.py

Write-Host "[3/5] Copying launcher to release/windows..."
New-Item -ItemType Directory -Path "release/windows" -Force | Out-Null
Copy-Item "dist/PPT-OpenCode-Launcher.exe" "release/windows/PPT-OpenCode-Launcher.exe" -Force

Write-Host "[4/5] Building release bundle zip..."
python packaging/windows/build_release_bundle.py `
  --repo-root "$repoRoot" `
  --launcher "release/windows/PPT-OpenCode-Launcher.exe"

Write-Host "[5/5] Done."
Write-Host "EXE: release/windows/PPT-OpenCode-Launcher.exe"
Write-Host "ZIP: release/windows/ppt-opencode-win-x64.zip"
