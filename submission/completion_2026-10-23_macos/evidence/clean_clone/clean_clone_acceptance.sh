#!/usr/bin/env bash
# Clean-clone acceptance sequence (2026-10-23). Runs inside a fresh clone whose .venv was created by scripts/quickstart_unix.sh.
# Usage: bash clean_clone_acceptance.sh <clone dir> <evidence dir>
set -u -o pipefail
set -m    # job control: without it bash starts background jobs with SIGINT ignored, so 'kill -INT' could not stand in for Ctrl+C (attempt 1, 2026-10-23)
CC="$1"; EV="$2"; PORT=8791
cd "$CC" || exit 1
PY="$CC/.venv/bin/python"
mkdir -p "$EV"
pass=0; fail=0
check() { if [ "$2" = "0" ]; then echo "[PASS] $1"; pass=$((pass+1)); else echo "[FAIL] $1 ($3)"; fail=$((fail+1)); fi; }

echo "clone HEAD: $(git log --oneline -1)"; echo "python: $($PY --version)"
"$PY" -B scripts/check_setup.py > "$EV/check_setup.txt" 2>&1; rc=$?
check "setup check passes and imports every package" $([ $rc -eq 0 ] && grep -q "SETUP OK" "$EV/check_setup.txt" && echo 0 || echo 1) "exit $rc"

"$PY" -B scripts/run_core_tests.py --json "$EV/run_core_tests.json" > "$EV/run_core_tests.txt" 2>&1; rc=$?
check "core tests: exit 0, no FAILED module" $([ $rc -eq 0 ] && ! grep -q "^\[FAILED" "$EV/run_core_tests.txt" && echo 0 || echo 1) "exit $rc"
tail -1 "$EV/run_core_tests.txt"

"$PY" -B scripts/demo_end_to_end.py --out-dir "$EV/demo_run1" > "$EV/demo_run1.txt" 2>&1; r1=$?
"$PY" -B scripts/demo_end_to_end.py --out-dir "$EV/demo_run2" > "$EV/demo_run2.txt" 2>&1; r2=$?
h1=$(cd "$EV/demo_run1" && find . -type f | sort | xargs shasum -a 256 | shasum -a 256); h2=$(cd "$EV/demo_run2" && find . -type f | sort | xargs shasum -a 256 | shasum -a 256)
check "demonstration runs twice and both output folders are byte-identical" $([ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ "$h1" = "$h2" ] && echo 0 || echo 1) "exit $r1/$r2"

"$PY" -B scripts/verify_dashboard_e2e.py --out "$EV/e2e_verification.json" > "$EV/e2e.txt" 2>&1; rc=$?
check "dashboard end to end (documented server command, every route and error, shutdown)" $rc "$(tail -1 "$EV/e2e.txt")"
tail -1 "$EV/e2e.txt"

"$PY" -B scripts/verify_research_serving_e2e.py --out "$EV/research_e2e.json" > "$EV/research_e2e.txt" 2>&1; rc=$?
check "research-model end to end reports NOT RUN (exit 3) because the artifacts are withheld" $([ $rc -eq 3 ] && grep -q "NOT RUN" "$EV/research_e2e.txt" && echo 0 || echo 1) "exit $rc"

"$PY" -B scripts/run_dashboard.py --research-model --port $PORT > "$EV/research_mode_start.txt" 2>&1; rc=$?
check "--research-model without the model file refuses to start with an explanation (exit 2)" $([ $rc -eq 2 ] && grep -q "does not exist" "$EV/research_mode_start.txt" && echo 0 || echo 1) "exit $rc"

start_server() { "$PY" -u -B scripts/run_dashboard.py --demo --port $PORT > "$EV/$1" 2>&1 & echo $!; }
wait_up() { for _ in $(seq 1 120); do curl -s -o /dev/null "http://127.0.0.1:$PORT/health" && return 0; sleep 0.5; done; return 1; }
SRV=$(start_server server_run1.txt); wait_up; up=$?
check "demo server starts on 127.0.0.1:$PORT and answers /health" $up "pid $SRV"
lsof -nP -iTCP:$PORT -sTCP:LISTEN > "$EV/listening.txt" 2>&1
check "it listens on the loopback address only" $(grep -q "127.0.0.1:$PORT" "$EV/listening.txt" && ! grep -q "\*:$PORT" "$EV/listening.txt" && echo 0 || echo 1) "$(tail -1 "$EV/listening.txt")"
curl -s "http://127.0.0.1:$PORT/api/status" > "$EV/api_status.json"
check "/api/status reports validation-only, held-out not performed, demo mode, research model and data absent" $("$PY" -c "
import json,sys; s=json.load(open('$EV/api_status.json'))
ok = s['validation_period_only'] and not s['held_out_evaluation_performed'] and s['synthetic_demo'] and not s['research_model_loaded'] and not s['research_data_present'] and not s['research_model_file_present']
sys.stdout.write('0' if ok else '1')") "see api_status.json"
curl -s "http://127.0.0.1:$PORT/api/sample" | "$PY" -c "import json,sys; s=json.load(sys.stdin); print(json.dumps({'history': s['history'], 'new': s['new']}))" > "$EV/score_request.json"
curl -s -X POST -H 'Content-Type: application/json' --data @"$EV/score_request.json" "http://127.0.0.1:$PORT/api/score" > "$EV/score_response.json"
check "synthetic scoring: one finite score in [0,1] per posted row, in order, labelled synthetic" $("$PY" -c "
import json,math,sys; q=json.load(open('$EV/score_request.json')); r=json.load(open('$EV/score_response.json'))
ok = r['synthetic_demo'] is True and [x['TransactionID'] for x in r['results']] == [x['TransactionID'] for x in q['new']] and all(math.isfinite(x['score']) and 0 <= x['score'] <= 1 for x in r['results']) and r['n_alerts'] == sum(x['score'] >= r['threshold'] for x in r['results'])
sys.stdout.write('0' if ok else '1')") "see score_response.json"
code=$(curl -s -o "$EV/invalid_empty.json" -w "%{http_code}" -X POST -H 'Content-Type: application/json' --data '{"new": []}' "http://127.0.0.1:$PORT/api/score")
code2=$(curl -s -o "$EV/invalid_amount.json" -w "%{http_code}" -X POST -H 'Content-Type: application/json' --data '{"new": [{"TransactionID": 1, "TransactionDT": 100, "TransactionAmt": "abc"}]}' "http://127.0.0.1:$PORT/api/score")
code3=$(curl -s -o "$EV/invalid_host.json" -w "%{http_code}" -H 'Host: attacker.example' "http://127.0.0.1:$PORT/api/status")
check "invalid input gets 400 with an explanation; a foreign Host header gets 403" $([ "$code" = 400 ] && [ "$code2" = 400 ] && [ "$code3" = 403 ] && echo 0 || echo 1) "$code/$code2/$code3"
curl -s "http://127.0.0.1:$PORT/api/communities" > "$EV/api_communities.json"
check "candidate communities: aggregate cards with the candidate statement" $("$PY" -c "
import json,sys; c=json.load(open('$EV/api_communities.json')); sys.stdout.write('0' if c['n_cards'] == len(c['cards']) > 0 and 'not' in c['standing_statement'].lower() + c['not_established'].lower() else '1')") "see api_communities.json"
code=$(curl -s -o "$EV/research_model_404.json" -w "%{http_code}" "http://127.0.0.1:$PORT/api/research/model")
check "research endpoint answers 404 'not loaded' (never a substitute score)" $([ "$code" = 404 ] && grep -q "not loaded" "$EV/research_model_404.json" && echo 0 || echo 1) "$code"
"$PY" -B scripts/run_dashboard.py --demo --port $PORT > "$EV/port_conflict.txt" 2>&1; rc=$?
check "a second server on the busy port is refused with an error (exit 2)" $([ $rc -eq 2 ] && grep -qi "address already in use" "$EV/port_conflict.txt" && echo 0 || echo 1) "exit $rc: $(tail -1 "$EV/port_conflict.txt")"
kill -INT "$SRV"; for _ in $(seq 1 40); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.25; done
check "Ctrl+C (SIGINT) stops the server and it prints 'stopped'" $(! kill -0 "$SRV" 2>/dev/null && grep -q "stopped" "$EV/server_run1.txt" && echo 0 || echo 1) "see server_run1.txt"
check "after the stop the port is free" $(! lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && echo 0 || echo 1) "port $PORT still listening"
SRV2=$(start_server server_run2.txt); wait_up; up=$?
code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health")
owner=$(lsof -t -nP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | head -1)
check "restart on the same port works: the NEW process listens and /health answers 200" $([ $up -eq 0 ] && [ "$code" = 200 ] && [ "$owner" = "$SRV2" ] && grep -q "dashboard on" "$EV/server_run2.txt" && echo 0 || echo 1) "code $code owner $owner new $SRV2"
kill -INT "$SRV2"; for _ in $(seq 1 40); do kill -0 "$SRV2" 2>/dev/null || break; sleep 0.25; done
check "the restarted server stops on SIGINT too" $(! kill -0 "$SRV2" 2>/dev/null && grep -q "stopped" "$EV/server_run2.txt" && echo 0 || echo 1) "see server_run2.txt"
kill "$SRV" "$SRV2" 2>/dev/null    # safety net: never leave a test server behind
git status --porcelain --untracked-files=no > "$EV/git_status_tracked_after.txt"
check "no tracked experiments/ record was changed by any of these runs" $(! grep -q "experiments/" "$EV/git_status_tracked_after.txt" && echo 0 || echo 1) "see git_status_tracked_after.txt"
echo; echo "acceptance: $pass passed, $fail failed"
