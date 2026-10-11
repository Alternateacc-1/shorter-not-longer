"""Merge the QR-code and lookalike part files into results/codes_lookalike/manifest.csv.

Refuses a quantity that already exists in the pipeline, alignment or supplementary
manifest (those are never modified) or that appears twice here.

    python -m analyses.build_manifest_codes
"""
import csv

from analyses.helpers import MANIFESTS, out_dir, require_committed, write_manifest

OUT = out_dir("codes_lookalike")
PARTS = ["part_qr_codes.csv", "part_lookalike.csv"]


def main():
    require_committed()
    taken = set()
    for p in MANIFESTS.values():
        taken |= {r["quantity"] for r in csv.DictReader(open(p, encoding="utf-8"))}
    rows, seen = [], set()
    for p in PARTS:
        for r in csv.DictReader(open(OUT / p, encoding="utf-8")):
            q = r["quantity"]
            assert q not in taken, f"{q} already in an earlier manifest"
            assert q not in seen, f"duplicate {q}"
            seen.add(q); rows.append(r)
    # derived rows: forms the text prints, and the definitions behind the lookalike counts
    V = {r["quantity"]: r for r in rows}
    st = require_committed()

    def der(q, v, u, f, src):
        assert q not in taken and q not in seen
        rows.append({"quantity": q, "value": repr(v) if isinstance(v, float) else v, "units": u,
                     "script": "analyses/build_manifest_codes.py", "function": f, "seeds": V[src]["seeds"],
                     "commit": st["commit"][:12]}); seen.add(q)
    der("FAR_sampled n=31 per thousand", 1000 * float(V["FAR_sampled n=31"]["value"]), "per 1000",
        "1000 x FAR_sampled n=31", "FAR_sampled n=31")
    der("w2 lookalike attacker identities per victim", int(V["w2 lookalike n=15 oracle_all_images victims"]["value"]) - 1,
        "identities", "victims - 1 (every other enrolled identity)", "w2 lookalike n=15 oracle_all_images victims")
    for p in (10, 50):
        der(f"w2 lookalike count threshold {p} pct", p, "%", f"definition: n_gt_{p}pct counts victims with rate > {p}%",
            "w2 lookalike n=15 oracle_all_images victims")
    der("w2 lookalike split fraction", 0.5, "fraction", "definition: u's images split into two halves (seed 3042)",
        "w2 lookalike n=15 realistic_half_B mean")
    der("w2 term is a name or unit, not a count", "True", "bool",
        "'zero-effort' and 'per thousand' are a name and a unit in the text, not counts", "w2 lookalike n=15 realistic_half_B mean")
    import json
    w0 = {r["quantity"]: r["value"] for r in csv.DictReader(open(MANIFESTS["supplementary"], encoding="utf-8"))}
    lengths = sorted(set(json.loads(w0["w0 studied lengths"])) | {31, 47})
    rows.append({"quantity": "w2 studied lengths", "value": json.dumps(lengths, separators=(",", ":")), "units": "list",
                 "script": "analyses/build_manifest_codes.py", "function": "w0 studied lengths + the QR lengths 31, 47",
                 "seeds": "-", "commit": st["commit"][:12]})
    write_manifest(OUT / "manifest.csv", rows)
    print(f"codes_lookalike/manifest.csv: {len(rows)} quantities from {len(PARTS)} parts")


if __name__ == "__main__":
    main()
