"""Compare the manifests in results/ with the committed versions, value by value.

A full run rewrites the four manifest files in place. This script reads each one
next to its committed version (git show HEAD:<path>) and reports every quantity
whose value changed. Numbers count as equal when they agree to 9 significant
digits, which absorbs last-digit floating-point noise between platforms; the
script, function and commit columns are ignored.

    python checks/compare_manifests.py            (from the repository root)
"""
import csv
import io
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFESTS = ["results/pipeline/manifest.csv", "results/alignment/manifest.csv",
             "results/supplementary/manifest.csv", "results/codes_lookalike/manifest.csv"]


def rows(text):
    return {r["quantity"]: r for r in csv.DictReader(io.StringIO(text))}


def same(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-15)


def main():
    changed = 0
    for path in MANIFESTS:
        committed = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{path}"],
                                   capture_output=True, text=True)
        if committed.returncode != 0 or not (REPO / path).exists():
            print(f"{path}: not found (committed or local), skipped")
            continue
        old = rows(committed.stdout)
        new = rows((REPO / path).read_text(encoding="utf-8"))
        diffs = [(q, old[q]["value"], new[q]["value"]) for q in old
                 if q in new and not same(old[q]["value"], new[q]["value"])]
        missing, extra = sorted(set(old) - set(new)), sorted(set(new) - set(old))
        print(f"{path}: {len(old)} rows, {len(diffs)} changed, {len(missing)} missing, {len(extra)} new")
        for q, a, b in diffs:
            print(f"   {q}: {a} -> {b}")
        for q in missing:
            print(f"   missing: {q}")
        changed += len(diffs) + len(missing) + len(extra)
    print("no differences" if changed == 0 else f"{changed} differences")
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()
