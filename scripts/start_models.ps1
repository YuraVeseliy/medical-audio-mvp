param(
    [int]$Port = 8081,
    [int]$ContextSize = 4096,
    [int]$Threads = 8
)

$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtimePath = Join-Path $projectPath '.runtime'
$serverFile = Join-Path $runtimePath 'llama\llama-server.exe'
$modelFile = Join-Path $runtimePath 'models\qwen2.5-3b-instruct-q4_k_m.gguf'

if (-not (Test-Path -LiteralPath $serverFile -PathType Leaf)) {
    $serverEntry = Get-ChildItem -LiteralPath (Join-Path $runtimePath 'llama') -Filter 'llama-server.exe' -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $serverEntry) { $serverFile = $serverEntry.FullName }
}
if (-not (Test-Path -LiteralPath $serverFile -PathType Leaf) -or -not (Test-Path -LiteralPath $modelFile -PathType Leaf)) {
    throw 'Модель или llama-server не найдены. Сначала выполните python scripts/setup_models.py.'
}

# В ручном терминале сервер работает до Ctrl+C. Приложение запускает его скрыто само.
Write-Host "Локальная модель: http://127.0.0.1:$Port; остановка Ctrl+C"
& $serverFile '-m' $modelFile '--host' '127.0.0.1' '--port' $Port '-c' $ContextSize '-ngl' '0' '-t' $Threads '--alias' 'medical-local' '--no-webui' '--no-context-shift'
exit $LASTEXITCODE
