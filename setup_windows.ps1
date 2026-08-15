<#
.SYNOPSIS
    One-shot Windows setup for this project (Kin no Mimi to Me). Creates the venv,
    installs PyTorch with the right CUDA build for your GPU, installs the rest of the
    dependencies, and verifies FFmpeg.

.DESCRIPTION
    Run from the repo folder:

        powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1

    Re-running is safe: it reuses an existing .venv and skips anything already installed.

    NOTE: this file is deliberately pure ASCII. Windows PowerShell 5.1 reads .ps1 files
    using the legacy ANSI codepage unless they carry a UTF-8 BOM, which corrupts non-ASCII
    characters -- and a corrupted byte can decode to a quote character and break parsing.

.PARAMETER CudaIndex
    Force a specific PyTorch wheel index: cu130, cu128, or cpu. Default: auto-detected
    from your driver. NVIDIA 50-series (Blackwell, sm_120) REQUIRES cu128 or newer; the
    plain PyPI wheel may lack sm_120 kernels and will fail at inference time.

.PARAMETER SkipFFmpeg
    Do not try to install FFmpeg via winget (just warn if it is missing).
#>
param(
    [ValidateSet('auto', 'cu130', 'cu128', 'cpu')]
    [string]$CudaIndex = 'auto',
    [switch]$SkipFFmpeg
)

# 'Continue', not 'Stop': this script shells out constantly (winget, pip, ffmpeg,
# nvidia-smi) and Windows PowerShell 5.1 turns any stderr line from a native command into
# an ErrorRecord, which under 'Stop' aborts the whole script over harmless chatter. Every
# native call below checks $LASTEXITCODE explicitly instead.
$ErrorActionPreference = 'Continue'
$repo = $PSScriptRoot
Write-Host "=== Japanese immersion subtitles + condensed audio: Windows setup ===" -ForegroundColor Cyan
Write-Host "repo: $repo"
Write-Host ""

function Test-Cmd([string]$name) {
    $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

# Pick up PATH changes made by winget in THIS session. winget tells you to restart the
# shell; this saves you from having to.
function Update-Path {
    $m = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
    $u = [Environment]::GetEnvironmentVariable('PATH', 'User')
    $env:PATH = ($m + ';' + $u)
}

# Start from the registry PATH so tools installed by an earlier run of this script (or by
# winget in another window) are visible even in a shell that started before them.
Update-Path

# ---------------------------------------------------------------- 1. Python
Write-Host "[1/5] Python" -ForegroundColor Yellow
$py = $null
foreach ($c in @('python.exe', 'py.exe')) {
    if (-not (Test-Cmd $c)) { continue }
    # Windows ships an "App Execution Alias" stub for python.exe that only advertises the
    # Microsoft Store. It exits non-zero and prints to stderr, so discard stderr and trust
    # the exit code rather than the presence of the command.
    $v = (& $c --version 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $v -notmatch 'Python 3\.(\d+)') {
        Write-Host "      $c is not a usable Python (Microsoft Store stub?) - skipping"
        continue
    }
    if ([int]$Matches[1] -lt 9) {
        Write-Host "      found $v - too old, need 3.9+"
        continue
    }
    $py = (Get-Command $c).Source
    break
}
if (-not $py) {
    if (-not (Test-Cmd 'winget')) {
        throw "No suitable Python and no winget. Install Python 3.12 from python.org, then re-run."
    }
    Write-Host "      installing Python 3.12 via winget..."
    winget install --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements --disable-interactivity --silent
    Update-Path
    $py = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    if (-not $py) {
        throw "Python installed but not on PATH yet. Open a NEW terminal and re-run this script."
    }
}
Write-Host "      using $py  ($(& $py --version 2>&1))" -ForegroundColor Green

# ---------------------------------------------------------------- 2. FFmpeg
Write-Host "[2/5] FFmpeg" -ForegroundColor Yellow
if (-not (Test-Cmd 'ffmpeg') -or -not (Test-Cmd 'ffprobe')) {
    if ($SkipFFmpeg) {
        Write-Host "      MISSING (skipped by request) - install it before running." -ForegroundColor Red
    }
    elseif (Test-Cmd 'winget') {
        Write-Host "      installing FFmpeg via winget..."
        winget install --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements --disable-interactivity
        Update-Path
    }
    else {
        Write-Host "      MISSING and no winget. Get a build from https://ffmpeg.org/download.html" -ForegroundColor Red
    }
}
if (Test-Cmd 'ffmpeg') {
    $fv = (& ffmpeg -version 2>&1 | Select-Object -First 1)
    Write-Host "      $fv" -ForegroundColor Green
}

# ---------------------------------------------------------------- 3. GPU / wheel index
Write-Host "[3/5] GPU" -ForegroundColor Yellow
$index = $CudaIndex
if ($index -eq 'auto') {
    if (Test-Cmd 'nvidia-smi') {
        $smi = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $smi) {
            $parts = ("$smi" -split ',')
            $gpuName = $parts[0].Trim()
            $driver = $parts[1].Trim()
            Write-Host "      $gpuName (driver $driver)"
            $major = 0
            if ($driver -match '^(\d+)') { $major = [int]$Matches[1] }
            # CUDA 13 wheels want a 580+ driver; CUDA 12.8 wheels want 525+.
            if ($major -ge 580) { $index = 'cu130' } else { $index = 'cu128' }
        }
        else { $index = 'cpu' }
    }
    else {
        Write-Host "      no NVIDIA GPU detected - CPU build (transcription will be slow)"
        $index = 'cpu'
    }
}
Write-Host "      PyTorch wheel index: $index" -ForegroundColor Green

# ---------------------------------------------------------------- 4. venv + deps
Write-Host "[4/5] Virtualenv and Python packages" -ForegroundColor Yellow
$venvPy = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    & $py -m venv (Join-Path $repo '.venv')
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
& $venvPy -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

$torchVer = & $venvPy -c "import torch;print(torch.__version__)" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "      torch $torchVer already installed - skipping"
}
else {
    $url = "https://download.pytorch.org/whl/$index"
    Write-Host "      installing torch from $url (about a 2 GB download)"
    & $venvPy -m pip install torch --index-url $url
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
}
Write-Host "      installing transformers / accelerate / numpy"
& $venvPy -m pip install "transformers>=4.40" "accelerate>=0.26" "numpy>=1.24" --quiet
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

# ---------------------------------------------------------------- 5. verify
Write-Host "[5/5] Verifying" -ForegroundColor Yellow
$env:PYTHONUTF8 = '1'
# Run check_env.py as a FILE, never as `python -c "<multi-line>"`: PowerShell does not
# escape inner double quotes when it builds a native command line, so the code would be
# truncated at the first one.
& $venvPy (Join-Path $repo 'check_env.py')
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "Setup complete." -ForegroundColor Green
    Write-Host '  Try it:   .\jpsubs.cmd --dry-run --batch "D:\path\to\Library"'
    Write-Host '  Then:     .\jpsubs.cmd --batch "D:\path\to\Library"'
    Write-Host "  The anime-whisper model (about 3 GB) downloads on the first real run."
}
else {
    Write-Host "Setup incomplete - see the problems listed above." -ForegroundColor Red
    Write-Host "  If FFmpeg was just installed, open a NEW terminal and re-run:"
    Write-Host "     .\.venv\Scripts\python.exe check_env.py"
}
exit $code
