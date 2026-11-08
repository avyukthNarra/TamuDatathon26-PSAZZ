"""Stage the combo entry (predict.py + models + kit modules) and zip it.

Usage: python3 mainFiles/combo/build_combo.py --out <zip outside the repo>
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
KIT_MODULES = ("__init__.py", "baseline.py", "schema.py", "io.py", "pair_mixer.py")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
with tempfile.TemporaryDirectory() as tmp:
    entry = Path(tmp) / "entry"
    (entry / "slot_forecasting").mkdir(parents=True)
    for name in KIT_MODULES:
        shutil.copyfile(MAIN / "slot_forecasting" / name, entry / "slot_forecasting" / name)
    for name in ("predict.py", "reproduce.json", "README.md"):
        shutil.copyfile(HERE / name, entry / name)
    shutil.copyfile(MAIN / "rule_model.py", entry / "rule_model.py")
    subprocess.run([sys.executable, str(MAIN / "build_entry.py"), "--source", str(entry), "--out", str(args.out)],
                   cwd=MAIN, check=True)
