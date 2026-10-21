"""Start the local API and investigation dashboard (Phase 13; standard library only).

  synthetic demonstration:   .venv/Scripts/python.exe -B scripts/run_dashboard.py --demo
  your own scoring bundle:   .venv/Scripts/python.exe -B scripts/run_dashboard.py --bundle path/to/bundle

Then open http://127.0.0.1:8765/ . The server binds to 127.0.0.1 only (use --allow-remote to override; it has no authentication). Without --demo or
--bundle the evidence and community tabs work and scoring answers 404. The dashboard shows recorded validation-period aggregates only; the demo
bundle is trained on synthetic data and reproduces no research result. Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.inference import scorer as S  # noqa: E402
from src.serving import app  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="train the synthetic-fixture bundle in memory and enable the sample button")
    ap.add_argument("--bundle", default=None, help="directory written by scripts/score_transactions.py train")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--allow-remote", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.demo and a.bundle:
            raise ValueError("choose --demo or --bundle, not both")
        if a.demo:
            from src.serving.demo import build_demo_services
            print("training the synthetic demonstration bundle ...", flush=True)
            services = build_demo_services()
        elif a.bundle:
            services = app.Services(bundle=S.load_bundle(a.bundle), synthetic_demo=False)
        else:
            services = app.Services()
        server = app.make_server(services, a.host, a.port, a.allow_remote)
    except (ValueError, S.ScoringError, S.SealedDataError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"dashboard on http://{a.host}:{server.server_address[1]}/  (validation-period results only; Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
