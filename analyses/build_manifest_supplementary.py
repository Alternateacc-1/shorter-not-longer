"""Merge the label-error, table-value and limitations part files into
results/supplementary/manifest.csv. Refuses a quantity that already exists in the
pipeline or alignment manifest (those are never modified) or appears twice.

    python -m analyses.build_manifest_supplementary
"""
import csv

from analyses.helpers import MANIFESTS, out_dir, require_committed, write_manifest

OUT = out_dir("supplementary")
PARTS = ["part_label_errors.csv", "part_label_errors_n20.csv", "part_table_values.csv", "part_limitations.csv"]


def keys(path):
    return {r["quantity"] for r in csv.DictReader(open(path, encoding="utf-8"))}


def main():
    require_committed()
    taken = keys(MANIFESTS["pipeline"]) | keys(MANIFESTS["alignment"])
    rows, seen = [], set()
    for p in PARTS:
        for r in csv.DictReader(open(OUT / p, encoding="utf-8")):
            q = r["quantity"]
            assert q not in taken, f"{q} already in the pipeline or alignment manifest"
            assert q not in seen, f"duplicate {q}"
            seen.add(q); rows.append(r)
    write_manifest(OUT / "manifest.csv", rows)
    print(f"supplementary/manifest.csv: {len(rows)} quantities from {len(PARTS)} parts")


if __name__ == "__main__":
    main()
