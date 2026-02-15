@echo off
setlocal

set REPO_ROOT=%~dp0\..\..
pushd "%REPO_ROOT%"

echo [1/5] Installing build dependencies...
python -m pip install --upgrade pip pyinstaller
if errorlevel 1 (
  echo Failed to install build dependencies.
  popd
  exit /b 1
)

echo [2/5] Building launcher exe...
python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --name PPT-OpenCode-Launcher ^
  packaging/windows/app_launcher.py
if errorlevel 1 (
  echo Build failed.
  popd
  exit /b 1
)

echo [3/5] Copying launcher to release\windows...
if not exist release\windows mkdir release\windows
copy /Y dist\PPT-OpenCode-Launcher.exe release\windows\PPT-OpenCode-Launcher.exe >nul

echo [4/5] Building release bundle zip...
python packaging/windows/build_release_bundle.py --repo-root "%REPO_ROOT%" --launcher "release/windows/PPT-OpenCode-Launcher.exe"
if errorlevel 1 (
  echo Failed to build release bundle.
  popd
  exit /b 1
)

echo [5/5] Done.
echo EXE: release\windows\PPT-OpenCode-Launcher.exe
echo ZIP: release\windows\ppt-opencode-win-x64.zip

popd
endlocal
