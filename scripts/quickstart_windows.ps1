<#
.SYNOPSIS
  Set up and verify the project on Windows without the dataset: environment, setup check, core tests, synthetic demonstration.

.DESCRIPTION
  Run from anywhere inside a clone:
      powershell -ExecutionPolicy Bypass -File scripts/quickstart_windows.ps1            (full run)
      powershell -ExecutionPolicy Bypass -File scripts/quickstart_windows.ps1 -SkipTests (setup and demonstration only)
      powershell -ExecutionPolicy Bypass -File scripts/quickstart_windows.ps1 -Python C:\Path\To\Python312\python.exe

  Steps: (1) create .venv with Python 3.12 if it does not exist (an existing .venv is reused, never deleted); (2) pip install -r requirements.txt;
  (3) scripts/check_setup.py; (4) scripts/run_core_tests.py (on a fresh clone 21 tests are NOT RUN because data and model files are not published,
  and five PyTorch modules are SKIPPED: expected, never counted as passed); (5) scripts/demo_end_to_end.py into demo_output/.
  It never deletes files, never downloads data, never reads or scores the held-out period and never creates an authorization file.
  Exit code 0 when every step succeeded; otherwise 1 with the failing step named.
#>
param(
    [switch]$SkipTests,
    [string]$Python = ""
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Fail([string]$message) {
    Write-Host ""
    Write-Host "QUICKSTART FAILED: $message" -ForegroundColor Red
    exit 1
}
function Step([string]$message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

if (-not (Test-Path (Join-Path $root "requirements.txt")) -or -not (Test-Path (Join-Path $root "scripts\check_setup.py"))) {
    Fail "this script must stay in the scripts folder of a clone of the repository (requirements.txt was not found in $root)"
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
Step "1/5 Python environment (.venv)"
if (Test-Path $venvPython) {
    Write-Host "Reusing the existing .venv"
} else {
    $base = $null
    $baseArgs = @()
    if ($Python) {
        if (-not (Test-Path $Python)) { Fail "-Python '$Python' does not exist" }
        $base = $Python
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $base = "py"; $baseArgs = @("-3.12") }
    }
    if (-not $base -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $v = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($v -eq "3.12") { $base = "python" }
    }
    if (-not $base) {
        Fail "Python 3.12 was not found. Install it from https://www.python.org/downloads/ and run this script again, or pass -Python <path to python.exe>."
    }
    Write-Host "Creating .venv with: $base $($baseArgs -join ' ')"
    & $base @baseArgs -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) { Fail "could not create .venv" }
}
$version = & $venvPython -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
Write-Host "Python in .venv: $version"
if (-not $version.StartsWith("3.12.")) {
    Write-Host "WARNING: the project is tested with Python 3.12; this .venv uses $version" -ForegroundColor Yellow
}

Step "2/5 Installing requirements.txt"
& $venvPython -m pip install --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "pip install -r requirements.txt failed (network access to PyPI is needed for the first installation)" }

Step "3/5 Setup check"
& $venvPython -B scripts/check_setup.py
if ($LASTEXITCODE -ne 0) { Fail "scripts/check_setup.py reported a problem (see the messages above)" }

if ($SkipTests) {
    Step "4/5 Core tests skipped (-SkipTests)"
} else {
    Step "4/5 Core tests (a few minutes)"
    & $venvPython -B scripts/run_core_tests.py
    if ($LASTEXITCODE -ne 0) { Fail "a core test FAILED (NOT RUN and SKIPPED lines are expected on a fresh clone; a FAILED line is not)" }
}

Step "5/5 Synthetic end-to-end demonstration"
& $venvPython -B scripts/demo_end_to_end.py --out-dir demo_output
if ($LASTEXITCODE -ne 0) { Fail "scripts/demo_end_to_end.py failed" }

Write-Host ""
Write-Host "QUICKSTART OK" -ForegroundColor Green
Write-Host "Start the dashboard:  .venv\Scripts\python.exe -B scripts\run_dashboard.py --demo"
Write-Host "Then open http://127.0.0.1:8765/ in a browser and stop the server with Ctrl+C."
exit 0
