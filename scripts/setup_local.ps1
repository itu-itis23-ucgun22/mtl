# Yerel kurulum (Windows / PowerShell). Repo kökünden çalıştır:
#     powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1
# Yaptığı: Python kontrolü -> .venv -> GPU'ya göre doğru torch -> editable kurulum -> doğrulama.
# Veri adımı için SETUP.md'ye bak.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")    # repo köküne geç
$Venv = ".venv"

Write-Host "==> 1/5  Python kontrolu"
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { throw "HATA: python bulunamadi (python.org'dan 3.10+ kur, PATH'e ekle)" }
$ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]"3.10") { throw "HATA: Python >=3.10 gerekli, bulunan $ver" }
Write-Host "    Python $ver OK"

Write-Host "==> 2/5  Sanal ortam ($Venv)"
if (-not (Test-Path $Venv)) { & python -m venv $Venv }
$Pip = Join-Path $Venv "Scripts\pip.exe"
$PyV = Join-Path $Venv "Scripts\python.exe"
& $Pip install --quiet --upgrade pip setuptools wheel

Write-Host "==> 3/5  Bagimliliklar"
$hasGpu = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasGpu) {
    Write-Host "    NVIDIA GPU bulundu -> requirements-gpu.txt (CUDA torch)"
    & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    $Req = "requirements-gpu.txt"
} else {
    Write-Host "    GPU yok -> requirements-local.txt (CPU torch; egitim cok yavas olur)"
    $Req = "requirements-local.txt"
}
& $Pip install -r $Req

Write-Host "==> 4/5  Paketi editable kur (src-layout: 'import mtl' icin sart)"
& $Pip install -e .

Write-Host "==> 5/5  Dogrulama"
& $PyV -c @"
import torch, timm, mtl
print('torch      :', torch.__version__)
print('CUDA var mi:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU        :', torch.cuda.get_device_name(0))
print('timm       :', timm.__version__)
print('mtl paketi : OK')
"@

Write-Host ""
Write-Host "======================================================================="
Write-Host "Kurulum tamam.  Ortami etkinlestir:   .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Hizli test (veri gerekmez):"
Write-Host "    pytest tests\ -q"
Write-Host ""
Write-Host "Sonraki adim: VERI (SETUP.md 'Veri' bolumu)"
Write-Host "    data\coco_subset\{images\{train,val},annotations\*.json}"
Write-Host "======================================================================="
