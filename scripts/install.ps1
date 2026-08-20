# IL Optimus — Windows install script
# Requires: Python 3.11+, NVIDIA CUDA GPU, git
# Usage: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  IL Optimus — Windows Installer" -ForegroundColor Cyan
Write-Host "  Intuition Learning Pipeline Studio" -ForegroundColor DarkGray
Write-Host ""

# ---- Check Python ----
$python = if (Get-Command python -ErrorAction SilentlyContinue) { "python" }
          elseif (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
          else { $null }
if (-not $python) {
    Write-Host "  Error: Python 3.11+ is required. Install from https://python.org" -ForegroundColor Red
    Write-Host "  Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    exit 1
}

$pyVersion = & $python --version 2>&1
Write-Host "  Found: $pyVersion" -ForegroundColor Green

# ---- Check CUDA ----
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpuInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    Write-Host "  GPU: $gpuInfo" -ForegroundColor Green
} else {
    Write-Host "  Warning: nvidia-smi not found. IL Optimus needs an NVIDIA CUDA GPU." -ForegroundColor Yellow
    Write-Host "  CPU-only mode will work for chat but not for training." -ForegroundColor Yellow
}

# ---- Install uv (fast Python package manager) ----
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host ""
    Write-Host "  Installing uv (fast Python package manager)..." -ForegroundColor Cyan
    & $python -m pip install uv --quiet
    $uv = "uv"
} else {
    Write-Host "  uv: already installed" -ForegroundColor Green
}

# ---- Clone and install IL Optimus ----
$installDir = "iloptimus"
if (-not (Test-Path $installDir)) {
    Write-Host ""
    Write-Host "  Cloning IL Optimus..." -ForegroundColor Cyan
    git clone https://github.com/Vaskrokodile/iloptimus.git $installDir
}
Set-Location $installDir

Write-Host ""
Write-Host "  Installing Python dependencies..." -ForegroundColor Cyan
Write-Host "  (This downloads torch + transformers + peft — may take a few minutes)" -ForegroundColor DarkGray
& $uv pip install -e ".[cuda]" --quiet 2>&1 | Out-Null

# ---- Build the web frontend ----
Write-Host ""
Write-Host "  Building web frontend..." -ForegroundColor Cyan
if (Get-Command npm -ErrorAction SilentlyContinue) {
    npm install --silent 2>&1 | Out-Null
    npm run build 2>&1 | Out-Null
    Write-Host "  Frontend built." -ForegroundColor Green
} else {
    Write-Host "  Warning: npm not found. The web UI won't be available." -ForegroundColor Yellow
    Write-Host "  Install Node.js from https://nodejs.org to get the full UI." -ForegroundColor Yellow
    Write-Host "  The API will still work at http://127.0.0.1:7860" -ForegroundColor Yellow
}

# ---- Done ----
Write-Host ""
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host "  IL Optimus is installed." -ForegroundColor Green
Write-Host "  ============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Quick start:" -ForegroundColor White
Write-Host "    cd $installDir" -ForegroundColor White
Write-Host "    iloptimus serve" -ForegroundColor White
Write-Host ""
Write-Host "  Then open http://127.0.0.1:7860 in your browser." -ForegroundColor White
Write-Host ""
Write-Host "  First run will download a model (~1-2 GB for the 1.5B model)." -ForegroundColor DarkGray
Write-Host "  The boosted-v1-small adapter (HumanEval 24% -> 70.88%) downloads automatically." -ForegroundColor DarkGray
Write-Host ""

# ---- Optionally start the server ----
$start = Read-Host "  Start IL Optimus now? (y/N)"
if ($start -match "^[yY]") {
    Write-Host ""
    Write-Host "  Starting server..." -ForegroundColor Cyan
    iloptimus serve
}
