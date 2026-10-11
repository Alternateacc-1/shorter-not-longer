"""Paths, the commit guard and the manifest writer shared by the scripts in analyses/.

Each analysis refuses to run on uncommitted code (outputs under results/ are
exempt), so the commit it records is the code that produced its numbers.
"""
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
PIPELINE_OUT = RESULTS / "pipeline"              # outputs of steps 1-8, read only here
sys.path.insert(0, str(REPO))

MANIFEST_FIELDS = ["quantity", "value", "units", "script", "function", "seeds", "commit"]
MANIFESTS = {"pipeline": PIPELINE_OUT / "manifest.csv",
             "alignment": RESULTS / "alignment" / "manifest.csv",
             "supplementary": RESULTS / "supplementary" / "manifest.csv"}


def out_dir(name):
    """results/<name>/, created if needed."""
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.rstrip()


def require_committed():
    """Refuse to run if any code or data file outside results/ differs from HEAD."""
    bad = []
    for l in git("status", "--porcelain", "--untracked-files=all").splitlines():
        path = l[3:].replace("\\", "/")
        if path.startswith("results/"):
            continue
        if path.endswith((".py", ".txt", ".md", ".csv", ".json", ".cff")):
            bad.append(l)
    if bad:
        raise SystemExit("refusing to run: uncommitted changes:\n  " + "\n  ".join(bad))
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD")}


def write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
