#!/usr/bin/env bash
# Set up and verify the project on macOS or Linux without the dataset: environment, setup check, core tests, synthetic demonstration.
#
#   bash scripts/quickstart_unix.sh                          (full run)
#   bash scripts/quickstart_unix.sh --skip-tests             (setup and demonstration only)
#   bash scripts/quickstart_unix.sh --python /path/to/python3.12
#
# Steps: (1) create .venv with Python 3.12 if it does not exist (an existing .venv is reused, never deleted); (2) install requirements.txt;
# (3) scripts/check_setup.py; (4) scripts/run_core_tests.py (on a fresh clone some tests are NOT RUN because data and model files are not published,
# and five PyTorch modules are SKIPPED: expected, never counted as passed); (5) scripts/demo_end_to_end.py into demo_output/.
# It never deletes files, never downloads data, never reads or scores the held-out period and never creates an authorization file.
# Exit code 0 when every step succeeded; otherwise 1 with the failing step named.
# Tested on macOS 27 (Apple Silicon, Python 3.12.13) on 2026-10-23. Linux has not been tested.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

fail() { printf '\nQUICKSTART FAILED: %s\n' "$1"; exit 1; }
step() { printf '\n==> %s\n' "$1"; }

SKIP_TESTS=0
PYTHON=""
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-tests) SKIP_TESTS=1; shift ;;
    --python)
      [ $# -ge 2 ] || fail "--python needs the path to a Python 3.12 interpreter"
      PYTHON="$2"; shift 2
      { [ -f "$PYTHON" ] && [ -x "$PYTHON" ]; } || fail "--python '$PYTHON' is not an executable file" ;;
    -h|--help) sed -n 2,13p "$0"; exit 0 ;;
    *) fail "unknown option: $1 (see --help)" ;;
  esac
done

[ -f requirements.txt ] && [ -f scripts/check_setup.py ] || fail "this script must stay in the scripts folder of a clone of the repository (requirements.txt was not found in $ROOT)"

VENV_PY="$ROOT/.venv/bin/python"
step "1/5 Python environment (.venv)"
if [ -x "$VENV_PY" ]; then
  echo "Reusing the existing .venv"
else
  BASE=""
  if [ -n "$PYTHON" ]; then
    BASE="$PYTHON"
  elif command -v python3.12 >/dev/null 2>&1; then
    BASE="$(command -v python3.12)"
  elif command -v python3 >/dev/null 2>&1 && [ "$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = "3.12" ]; then
    BASE="$(command -v python3)"
  fi
  [ -n "$BASE" ] || fail "Python 3.12 was not found. Install it (macOS: 'brew install python@3.12' or https://www.python.org/downloads/; Linux: your package manager or 'uv python install 3.12'), then run this script again, or pass --python <path to python3.12>."
  echo "Creating .venv with: $BASE"
  "$BASE" -m venv .venv || fail "could not create .venv"
  [ -x "$VENV_PY" ] || fail "could not create .venv"
fi
VERSION="$("$VENV_PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
echo "Python in .venv: $VERSION"
case "$VERSION" in 3.12.*) ;; *) echo "WARNING: the project is tested with Python 3.12; this .venv uses $VERSION" ;; esac

if [ "$(uname -s)" = "Darwin" ] && [ ! -e /opt/homebrew/opt/libomp/lib/libomp.dylib ] && [ ! -e /usr/local/opt/libomp/lib/libomp.dylib ]; then
  echo "NOTE: XGBoost on macOS needs the OpenMP runtime, which was not found. Install it with:  brew install libomp"
  echo "      (the setup check in step 3 confirms whether XGBoost loads)"
fi

step "2/5 Installing requirements.txt"
if "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  "$VENV_PY" -m pip install --disable-pip-version-check -r requirements.txt || fail "pip install -r requirements.txt failed (network access to PyPI is needed for the first installation)"
elif command -v uv >/dev/null 2>&1; then
  echo "This .venv has no pip (it was probably created by uv); installing with: uv pip install"
  uv pip install --python "$VENV_PY" -r requirements.txt || fail "uv pip install -r requirements.txt failed (network access to PyPI is needed for the first installation)"
else
  "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || fail "this .venv has no pip and ensurepip failed; recreate the environment with python3.12 -m venv in a new folder"
  "$VENV_PY" -m pip install --disable-pip-version-check -r requirements.txt || fail "pip install -r requirements.txt failed (network access to PyPI is needed for the first installation)"
fi

step "3/5 Setup check"
"$VENV_PY" -B scripts/check_setup.py || fail "scripts/check_setup.py reported a problem (see the messages above)"

if [ "$SKIP_TESTS" -eq 1 ]; then
  step "4/5 Core tests skipped (--skip-tests)"
else
  step "4/5 Core tests (about a minute)"
  "$VENV_PY" -B scripts/run_core_tests.py || fail "a core test FAILED (NOT RUN and SKIPPED lines are expected on a fresh clone; a FAILED line is not)"
fi

step "5/5 Synthetic end-to-end demonstration"
"$VENV_PY" -B scripts/demo_end_to_end.py --out-dir demo_output || fail "scripts/demo_end_to_end.py failed"

printf '\nQUICKSTART OK\n'
echo "Start the dashboard:  .venv/bin/python -B scripts/run_dashboard.py --demo"
echo "Then open http://127.0.0.1:8765/ in a browser and stop the server with Ctrl+C."
exit 0
