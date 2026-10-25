param(
    [string]$Manifest = "data/manifest.json",
    [string]$VoiceName = "Microsoft Irina Desktop - Russian"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = if ([System.IO.Path]::IsPathRooted($Manifest)) { $Manifest } else { Join-Path $projectRoot $Manifest }
$dataset = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$voice = New-Object -ComObject SAPI.SpVoice
$tokens = $voice.GetVoices()
$token = $null
for ($index = 0; $index -lt $tokens.Count; $index++) {
    $candidate = $tokens.Item($index)
    if ($candidate.GetDescription() -eq $VoiceName) { $token = $candidate; break }
}
if ($null -eq $token) { throw "SAPI voice not installed: $VoiceName" }
$voice.Voice = $token
$voice.Rate = 0
$voice.Volume = 100
$utf8 = New-Object System.Text.UTF8Encoding($false)
$results = @()
foreach ($item in $dataset.items) {
    $audioPath = Join-Path $projectRoot $item.audio
    $referencePath = Join-Path $projectRoot $item.reference
    [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $audioPath))
    [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $referencePath))
    [System.IO.File]::WriteAllText($referencePath, [string]$item.text, $utf8)
    $stream = New-Object -ComObject SAPI.SpFileStream
    # SpeechAudioFormatType.SAFT16kHz16BitMono = 18.
    $stream.Format.Type = 18
    try {
        $stream.Open($audioPath, 3, $false)
        $voice.AudioOutputStream = $stream
        [void]$voice.Speak([string]$item.text, 0)
    } finally {
        $stream.Close()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($stream)
    }
    $results += [ordered]@{ id = $item.id; split = $item.split; audio = $item.audio; sha256 = (Get-FileHash -LiteralPath $audioPath -Algorithm SHA256).Hash.ToLower(); referenceSha256 = (Get-FileHash -LiteralPath $referencePath -Algorithm SHA256).Hash.ToLower() }
}
foreach ($control in $dataset.controls) {
    $controlPath = Join-Path $projectRoot $control.audio
    [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $controlPath))
    $sampleCount = [int](16000 * $control.duration_seconds)
    $pcmLength = $sampleCount * 2
    $file = [System.IO.File]::Create($controlPath)
    $writer = New-Object System.IO.BinaryWriter($file)
    try {
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("RIFF")); $writer.Write([int](36 + $pcmLength))
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("WAVEfmt ")); $writer.Write([int]16)
        $writer.Write([int16]1); $writer.Write([int16]1); $writer.Write([int]16000); $writer.Write([int]32000)
        $writer.Write([int16]2); $writer.Write([int16]16); $writer.Write([System.Text.Encoding]::ASCII.GetBytes("data"))
        $writer.Write([int]$pcmLength); $writer.Write((New-Object byte[] $pcmLength))
    } finally { $writer.Dispose(); $file.Dispose() }
    $results += [ordered]@{ id = $control.id; split = "control"; audio = $control.audio; sha256 = (Get-FileHash -LiteralPath $controlPath -Algorithm SHA256).Hash.ToLower() }
}
$generation = [ordered]@{
    generatedAt = (Get-Date).ToString("o"); generator = "Windows SAPI.SpVoice"; voice = $VoiceName
    format = "PCM 16000 Hz 16 bit mono"; rate = 0; volume = 100
    manifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLower()
    validation = "Синтетическое аудио; эталон равен входу SAPI. Независимая сверка прослушиванием не выполнена."
    files = $results
}
[System.IO.File]::WriteAllText((Join-Path $projectRoot "data/generation.json"), ($generation | ConvertTo-Json -Depth 8), $utf8)
[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($voice)
Write-Output "Generated $($dataset.items.Count) fictional TTS dialogues and $($dataset.controls.Count) silence control."
