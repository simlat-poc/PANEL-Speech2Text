$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$exePath = "$root\dist\panel_stt\panel_stt.exe"
$ffmpegBin = "$root\ffmpeg-8.1-essentials_build\bin"

if (!(Test-Path $exePath) -and (Test-Path "$root\App\panel_stt.exe")) {
    $exePath = "$root\App\panel_stt.exe"
}
if (!(Test-Path $ffmpegBin) -and (Test-Path "$root\ffmpeg_bin")) {
    $ffmpegBin = "$root\ffmpeg_bin"
}

$env:PATH = "$ffmpegBin;$env:PATH"
$env:HF_HOME = "$root\offline_assets\hf_home"
$env:TRANSFORMERS_CACHE = "$root\offline_assets\hf_home\transformers"
$env:TORCH_HOME = "$root\offline_assets\torch_home"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

if (-not $env:HF_TOKEN -and -not $env:HUGGINGFACE_TOKEN -and -not $env:HUGGINGFACEHUB_API_TOKEN) {
    Write-Error "Diarization requires a Hugging Face token. Set HF_TOKEN in this shell before running."
}

& "$exePath" `
  --input "$root\Inputs\a3en-5speakers-short.mp3" `
  --output "$root\Outputs\exe-offline-largev3-diar.json" `
  --model large-v3 `
  --enable-diarization `
  --export-html `
  --device cuda
