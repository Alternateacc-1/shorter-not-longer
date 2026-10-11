"""Run bookkeeping shared by the pipeline steps.

Every step calls require_committed_code() before it computes anything, so the
commit hash it records is the code that actually ran. The only working-tree
changes allowed are outputs under results/ and deleted derived files.

Set DEV_RUN=1 to run on uncommitted code while testing; the outputs then go to
results/pipeline/dev/ and never replace the published ones.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_MAIN = REPO / "results" / "pipeline"
DEV = bool(os.environ.get("DEV_RUN"))
OUT = OUT_MAIN / "dev" if DEV else OUT_MAIN
OUT.mkdir(parents=True, exist_ok=True)
EMB_NPZ = OUT_MAIN / "embeddings_cache.npz"     # written by step 1 from the raw images
sys.path.insert(0, str(REPO))

DERIVED_EXT = (".npz", ".csv", ".json", ".txt", ".log")


def _git(*args):
    # rstrip only: porcelain lines start with a significant space (" D path")
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.rstrip()


def git_state():
    lines = [l for l in _git("status", "--porcelain", "--untracked-files=all").splitlines() if l]
    deleted = [l[3:] for l in lines if l.startswith(" D")]
    other = [l for l in lines if not l.startswith(" D")]
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "deleted_derived": deleted,
        "other_changes": other,
    }


def require_committed_code():
    """Refuse to run unless every code file is exactly as committed."""
    st = git_state()
    bad = [l for l in st["other_changes"]
           if not l[3:].replace("\\", "/").startswith("results/")]
    bad += [f for f in st["deleted_derived"] if not f.endswith(DERIVED_EXT)]
    st["dev"] = DEV
    if bad and DEV:
        st["uncommitted"] = bad
        print("DEV_RUN: running on uncommitted code; outputs go to", OUT, flush=True)
        return st
    if bad:
        raise SystemExit("refusing to run: working tree differs from HEAD in code:\n  "
                         + "\n  ".join(bad))
    return st


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_name():
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return platform.processor()


def versions():
    import numpy, scipy, sklearn, onnxruntime, PIL
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "onnxruntime": onnxruntime.__version__,
        "onnxruntime_providers": onnxruntime.get_available_providers(),
        "Pillow": PIL.__version__,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "cpu_id": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }


class Log:
    """Tee to stdout and a log file, with elapsed time."""
    def __init__(self, name):
        self.t0 = time.time()
        self.f = open(OUT / f"{name}.log", "w", encoding="utf-8")

    def __call__(self, *a):
        msg = f"[{time.time() - self.t0:7.1f}s] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n"); self.f.flush()


def _local_paths():
    """Machine-specific prefixes that must not end up in a published JSON file,
    longest first, with the placeholder that replaces each."""
    from fuzzycommit.common import LFW_DIR
    pairs = {str(LFW_DIR): "<LFW_DIR>", sys.prefix: "<VENV>", str(REPO): "<REPO>",
             str(Path.home()): "<HOME>"}
    return sorted(((k, v) for k, v in pairs.items() if k and len(k) > 3), key=lambda kv: -len(kv[0]))


def redact(text):
    """Replace local paths in a string with placeholders (both slash styles)."""
    for prefix, tag in _local_paths():
        for p in {prefix, prefix.replace("\\", "/"), prefix.replace("/", "\\")}:
            text = text.replace(p, tag)
    return text


def dump(obj, name):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, str):
            return redact(o)
        if hasattr(o, "item") and not hasattr(o, "__len__"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        return o
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(conv(obj), f, indent=1)
