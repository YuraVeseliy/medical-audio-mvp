param(
    [string]$PythonExe = '',
    [string]$OutputRoot = '',
    [switch]$InstallBuildTool,
    [switch]$SkipRuntime
)

$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $projectPath '.venv\Scripts\python.exe'
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $projectPath '..\..\output\portable'
}
$outputPath = [IO.Path]::GetFullPath($OutputRoot)
$pythonPath = [IO.Path]::GetFullPath($PythonExe)
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python не найден: $pythonPath. Сначала выполните setup.cmd."
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stagePath = Join-Path $outputPath $stamp
$workPath = Join-Path $projectPath "build\portable-$stamp"
if (Test-Path -LiteralPath $stagePath) {
    throw "Папка сборки уже существует: $stagePath. Повторите запуск позже."
}
New-Item -ItemType Directory -Path $stagePath -Force | Out-Null

Push-Location -LiteralPath $projectPath
try {
    if ($InstallBuildTool) {
        & $pythonPath '-m' 'pip' 'install' '-r' 'requirements-build.txt' '-c' 'requirements-lock.txt'
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить PyInstaller 6.16.0.' }
    }
    & $pythonPath '-c' 'import PyInstaller; assert PyInstaller.__version__ == "6.16.0", PyInstaller.__version__'
    if ($LASTEXITCODE -ne 0) {
        throw 'Нужен PyInstaller 6.16.0. Используйте -InstallBuildTool или установите его в окружение.'
    }
    & $pythonPath '-m' 'PyInstaller' '--noconfirm' '--distpath' $stagePath '--workpath' $workPath 'MedicalAudio.spec'
    if ($LASTEXITCODE -ne 0) { throw 'Сборка PyInstaller завершилась с ошибкой.' }
    $bundlePath = Join-Path $stagePath 'MedicalAudio'
    $exePath = Join-Path $bundlePath 'MedicalAudio.exe'
    if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) { throw 'В сборке нет MedicalAudio.exe.' }
    if (-not $SkipRuntime) {
        & $pythonPath 'scripts\package_runtime.py' '--destination' (Join-Path $bundlePath '.runtime')
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось проверить и скопировать runtime.' }
    }
    # Папка сборки новая. Никакие старые результаты не удаляются и не перемещаются.
    Copy-Item -LiteralPath (Join-Path $projectPath 'README.md') -Destination $bundlePath
    Copy-Item -LiteralPath (Join-Path $projectPath 'data') -Destination (Join-Path $bundlePath 'data') -Recurse
    New-Item -ItemType Directory -Path (Join-Path $bundlePath 'docs') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectPath 'docs\manual-tests.md') -Destination (Join-Path $bundlePath 'docs')
    $launcher = @'
@echo off
setlocal
cd /d "%~dp0"
start "" /wait "MedicalAudio.exe"
if errorlevel 1 (
    echo Application failed. Keep the full error message and package version.
    pause
)
endlocal
'@
    [IO.File]::WriteAllText((Join-Path $bundlePath 'launch.cmd'), $launcher.Replace("`n", "`r`n"), [Text.Encoding]::ASCII)
    $version = & $pythonPath '-c' 'import platform, sys; print(sys.version); print(platform.platform())'
    $summary = [ordered]@{
        created_at = (Get-Date).ToString('o')
        python_and_platform = @($version)
        pyinstaller = '6.16.0'
        runtime_included = (-not $SkipRuntime)
        executable_sha256 = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash.ToLowerInvariant()
        scope = 'Windows build on development PC; microphone and another PC require separate human checks'
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $bundlePath 'build-manifest.json') -Encoding UTF8
    Write-Host "Готовая папка: $bundlePath"
    Write-Host 'Запустите launch.cmd. Перед передачей проверьте весь комплект на другом ПК.'
    if ($SkipRuntime) { Write-Host 'Внимание: -SkipRuntime исключил модели; для обработки нужен подготовленный runtime.' }
}
finally {
    Pop-Location
}
