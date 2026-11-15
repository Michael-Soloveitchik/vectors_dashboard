"""Boot the dashboard outside Jupyter (blocks until Ctrl-C).

    python _runserver.py            # synthetic demo data
    python _runserver.py --real     # real embeddings + audio inspector
    python _runserver.py --real --model redimnet_b3 --port 8071
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]

ap = argparse.ArgumentParser()
ap.add_argument("--real", action="store_true")
ap.add_argument("--model", default="idrnd_redimnet_b3")
ap.add_argument("--port", type=int, default=8070)
ap.add_argument("--labels-dir", default=None, help="where own labels are saved (default: dumps/)")
args = ap.parse_args()

from frontend import Gui  # noqa: E402

kw = {"labels_dir": args.labels_dir} if args.labels_dir else {}
gui = Gui.real(args.model, **kw) if args.real else Gui.demo(**kw)
gui.run(port=args.port, jupyter_mode="external")
print("READY", flush=True)
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    gui.stop()
