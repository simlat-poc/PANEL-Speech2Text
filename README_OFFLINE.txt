PANEL Speech2Text Offline Delivery (Windows 10/11 x64, NVIDIA GPU)

What is included:
- EXE bundle in dist\panel_stt
- Pre-cached models in offline_assets
  - Systran/faster-whisper-small
  - Systran/faster-whisper-large-v3
  - pyannote/speaker-diarization-community-1
- Offline wheelhouse in offline_wheels
- Bundled FFmpeg binaries in ffmpeg-8.1-essentials_build\bin
- Launcher script: run_offline_gpu.ps1

Prerequisites on client machine:
1) NVIDIA driver installed and GPU visible in nvidia-smi.
2) Hugging Face token available for diarization access checks:
   - set HF_TOKEN before running.
3) No internet is required for model downloads if offline_assets is copied intact.

How to run:
1) Open PowerShell in this folder.
2) Set token:
   $env:HF_TOKEN = "hf_..."
3) Run:
   .\run_offline_gpu.ps1

Default launcher command:
- input: Inputs\a3en-5speakers-short.mp3
- output: Outputs\exe-offline-largev3-diar.json
- model: large-v3
- diarization: enabled
- html export: enabled
- device: cuda

How to change input/output:
Run the EXE directly with your own paths:
  .\dist\panel_stt\panel_stt.exe --input "Inputs\your.mp3" --output "Outputs\your.json" --model large-v3 --enable-diarization --export-html --device cuda

Troubleshooting:
- "ffmpeg not found":
  Ensure ffmpeg-8.1-essentials_build\bin exists and run via run_offline_gpu.ps1.

- "no Hugging Face token provided":
  Set HF_TOKEN in the shell before running.

- "cuda unavailable":
  Update NVIDIA driver and verify with:
  python -c "import torch; print(torch.cuda.is_available())"

- Torchcodec warnings:
  Pyannote may warn about torchcodec DLL loading. In this pipeline, diarization still runs successfully as long as final transcription/diarization outputs are produced.

Validation target:
- JSON output should be created.
- HTML timeline should be created next to JSON.
