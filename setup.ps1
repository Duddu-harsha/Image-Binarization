# =============================================================================
# image-binarization - Windows setup (PowerShell)
# =============================================================================
# Usage (from the project folder):
#   .\setup.bat                      (easiest - bypasses execution policy)
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1 [-Recreate]
#
# Creates .\venv with Python 3.10+ and installs requirements.txt.
# =============================================================================

param([switch]$Recreate)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvDir = Join-Path $ProjectRoot "venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"

function Ok($m)   { Write-Host "[OK]    $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[FAIL]  $m" -ForegroundColor Red; exit 1 }

function Run {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

# Find Python 3.10+
$PythonCmd = $null
foreach ($v in @("3.12", "3.11", "3.10", "3.13")) {
    try {
        & py "-$v" --version *> $null
        if ($LASTEXITCODE -eq 0) { $PythonCmd = @("py", "-$v"); break }
    } catch { }
}
if (-not $PythonCmd) {
    try {
        $parts = ((& python --version 2>&1) -replace "Python ", "").Split(".")
        if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10) { $PythonCmd = @("python") }
    } catch { }
}
if (-not $PythonCmd) { Fail "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH')." }
$PyExe  = $PythonCmd[0]
$PyArgs = @()
if ($PythonCmd.Count -gt 1) { $PyArgs = $PythonCmd[1..($PythonCmd.Count - 1)] }
Ok ("Using " + (& $PyExe @PyArgs --version))

if ((Test-Path $VenvDir) -and $Recreate) {
    Warn "Removing existing venv (-Recreate)..."
    Remove-Item -Recurse -Force $VenvDir
}
if (-not (Test-Path $VenvPy)) {
    Run $PyExe ($PyArgs + @("-m", "venv", $VenvDir))
    Ok "Virtual environment created"
}

Run $VenvPy @("-m", "pip", "install", "--upgrade", "pip")
Run $VenvPy @("-m", "pip", "install", "-r", (Join-Path $ProjectRoot "requirements.txt"))
Run $VenvPy @("-m", "pytest", "-q", (Join-Path $ProjectRoot "tests"))
Ok "Setup complete - tests pass"

Write-Host ""
Write-Host "  Activate:  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "  Convert:   python main.py --input path\to\image.jpg" -ForegroundColor Cyan
Write-Host "  Web demo:  python app.py" -ForegroundColor Cyan
