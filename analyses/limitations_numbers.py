"""Numbers the Limitations section cites: the label errors (all probes are kept and the
errors disclosed) and the stability of the recommendation (the longest n positive under
every weighting with stable closure; n=20 is reported as a knife-edge).

Nothing a manifest or an earlier analysis output already holds is recomputed:
  * base values are pipeline manifest rows; label-error variants at n = 9, 15, 24 come from
    analyses.label_errors, the n=20 failure and closure from analyses.label_errors_n20, and
    the n=20 t / parity / residual per weighting are copied from label_errors.json (its
    commit is recorded).
  * computed here, because no output has them: closure at n=15 and n=20 under every weighting
    for every variant. Same draws as Table 1 (subject bootstrap, 1,000 replicates, seed
    542+n); subjects are never removed, so every variant resamples the same subjects.
  * every value computed here that a manifest already holds is asserted equal to it.
Also: names, counts and LFW image numbers of the label errors, the FRR count threshold,
and the selection under both readings of "stable closure".

    python -m analyses.limitations_numbers   (after label_errors, label_errors_n20, table_values)
"""
import csv
import json

import numpy as np

from analyses.helpers import MANIFESTS, out_dir, require_committed, write_manifest
from analyses import crosscheck as V
from analyses.label_errors import DROP, variant
from analyses.build_manifest_supplementary import PARTS

OUT = out_dir("supplementary")
CONFIRMED = ["Michael_Schumacher_0008", "Mahmoud_Abbas_0012"]   # a different person (checked by eye)
UNCERTAIN = ["Naomi_Watts_0022", "Abdullah_Gul_0016"]           # lowest genuine cosine, identity not settled
VARIANTS = ("base", "drop2", "drop4", "drop2_fixedmean", "drop4_fixedmean")
WGT = ("probe", "capped", "subject")


def manifest(path):
    return {r["quantity"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}


def closures(c, W, n):
    S = V.Setup(c, W); L = V.Length(S, n)
    draws = V.bootstrap_draws(n, S.NS); nes = np.array([L.neff(x) for x in draws])
    out = {}
    for w in WGT:
        par = np.array([L.choose(w, x)[0] for x in draws], float)
        out[w] = 100 * float(((n - par > 0) & (nes - par > 0)).mean())
    return out


def main():
    st = require_committed()
    M = manifest(MANIFESTS["pipeline"])
    M0 = {}                                     # the other supplementary parts (not this script's own)
    for part in PARTS:
        if part != "part_limitations.csv":
            M0.update(manifest(OUT / part))
    j3 = json.load(open(OUT / "label_errors.json", encoding="utf-8"))
    j3b = json.load(open(OUT / "label_errors_n20.json", encoding="utf-8"))
    c3 = j3["git"]["commit"][:12]
    rows = []

    def add(q, v, u, f, seeds, script="analyses/limitations_numbers.py", commit=None):
        assert q not in M and q not in M0, q
        rows.append({"quantity": q, "value": json.dumps(v, separators=(",", ":")) if isinstance(v, list) else
                     (repr(v) if isinstance(v, float) else v), "units": u, "script": script, "function": f,
                     "seeds": seeds, "commit": commit or st["commit"][:12]})

    # label_errors' base values must equal the pipeline's (else the drop deltas are not deltas)
    b = j3["base"]["lengths"]
    pairs = [(b["15"]["subject"]["closure_pct"], "closure_pct n=15 subject"), (b["24"]["probe"]["closure_pct"], "closure_pct n=24"),
             (b["15"]["frr"]["mean"], "frr_mean n=15"), (b["15"]["frr"]["n_gt_5pct"], "frr_n_gt_5pct n=15"),
             (b["20"]["subject"]["residual"], "residual n=20 subject"), (j3b["base"]["closure_subject_pct"], "closure_pct n=20 subject")]
    for x, k in pairs:
        assert abs(float(x) - float(M[k]["value"])) < 1e-9, (k, x, M[k]["value"])

    # n=20 t / parity / residual per weighting, drop variants: copied from label_errors.json
    for v in VARIANTS[1:]:
        for w in WGT:
            r = j3[v]["lengths"]["20"][w]
            for k in ("t", "parity"):
                add(f"w1 label {v} n=20 {w} {k}", int(r[k]), "-", f"label_errors.json {v} lengths[20][{w}][{k}]",
                    "projection 43", script="analyses/label_errors.py", commit=c3)
            add(f"w1 label {v} n=20 {w} residual", float(r["residual"]), "bits",
                f"label_errors.json {v} lengths[20][{w}][residual]", "projection 43",
                script="analyses/label_errors.py", commit=c3)
        add(f"w1 label {v} probes", int(j3[v]["probes"]), "count", "label_errors.json probes", "-",
            script="analyses/label_errors.py", commit=c3)

    # closure at n=15 and n=20 under every weighting, every variant (computed: absent everywhere)
    emb, cen = V.load(); W, _ = V.W_of(43)
    data = {"base": cen}
    for k, d in DROP.items():
        data[k] = variant(emb, cen, d, fixed_mean=False)
        data[k + "_fixedmean"] = variant(emb, cen, d, fixed_mean=True)
    clo = {}
    for v in VARIANTS:
        for n in (15, 20):
            clo[(v, n)] = closures(data[v], W, n)
            print(v, n, clo[(v, n)], flush=True)
    for n in (15, 20):                                          # cross-checks against every value already held
        for w in WGT:
            key = f"closure_pct n={n}" + ("" if w == "probe" else f" {w}")
            assert abs(clo[("base", n)][w] - float(M[key]["value"])) < 1e-9, (key, clo[("base", n)][w])
    for v in VARIANTS[1:]:
        for w in ("probe", "subject"):
            assert abs(clo[(v, 15)][w] - j3[v]["lengths"]["15"][w]["closure_pct"]) < 1e-9, (v, w)
    for v in ("drop2", "drop4"):
        assert abs(clo[(v, 20)]["subject"] - j3b[v]["closure_subject_pct"]) < 1e-9, v
    for v in VARIANTS[1:]:
        for n in (15, 20):
            for w in WGT:
                have = (n == 15 and w != "capped") or (n == 20 and w == "subject" and v in ("drop2", "drop4"))
                if not have:
                    add(f"w1 label {v} n={n} {w} closure_pct", clo[(v, n)][w], "%",
                        "1,000 replicates, N_eff per replicate (crosscheck.bootstrap_draws)", f"projection 43; boot {542 + n}")

    # which length the rule picks, under both readings of 'stable closure'
    def allpos(v):
        L = j3[v]["lengths"]
        return sorted((int(n), min(L[n][w]["residual"] for w in WGT)) for n in L if all(L[n][w]["residual"] > 0 for w in WGT))
    lo = max(min(clo[(v, 20)].values()) for v in VARIANTS)       # n=20 must fail 'stable' in every variant
    hi = min(min(clo[(v, 15)].values()) for v in VARIANTS)       # n=15 must pass it in every variant
    for v in VARIANTS:
        ap = allpos(v)
        longer = [n for n, _ in ap if n > 15]
        assert set(longer) <= {20}, (v, longer)                   # only n=20 competes with n=15
        add(f"w1 rule {v} longest n positive under every weighting", ap[-1][0], "n", "max n with residual > 0 under all three weightings",
            "projection 43")
        add(f"w1 rule {v} n with the largest all-weighting residual", max(ap, key=lambda x: x[1])[0], "n",
            "literal reading 'largest residual': argmax over all-positive n of the minimum residual", "projection 43")
        for n in (15, 20):
            add(f"w1 rule {v} n={n} closure min over weightings", min(clo[(v, n)].values()), "%", "min over probe/capped/subject",
                f"projection 43; boot {542 + n}")
    add("w1 rule stable-closure threshold lower bound (exclusive)", lo, "%",
        "max over the 5 variants of the n=20 closure (min over weightings); a threshold must exceed it", "-")
    add("w1 rule stable-closure threshold upper bound (inclusive)", hi, "%",
        "min over the 5 variants of the n=15 closure (min over weightings); a threshold must not exceed it", "-")
    add("w1 rule stable-closure window, every weighting", [lo, hi], "% (lo, hi]",
        "closure = min over the three weightings; a threshold in (lo, hi] picks n=15 in every variant", "-")
    plo = max(clo[(v, 20)]["probe"] for v in VARIANTS); phi = min(clo[(v, 15)]["probe"] for v in VARIANTS)
    add("w1 rule stable-closure window, probe-weighted", [plo, phi], "% (lo, hi]",
        "closure = probe-weighted (Table 1's column); a threshold in (lo, hi] picks n=15 in every variant", "-")

    # label errors: names, counts, image indices; FRR count threshold
    add("w1 label confirmed probes", CONFIRMED, "list", "checked by eye: a different person in the image", "-")
    add("w1 label uncertain probes", UNCERTAIN, "list", "lowest genuine cosine; identity not settled", "-")
    add("w1 label confirmed count", len(CONFIRMED), "probes", "len(confirmed)", "-")
    add("w1 label uncertain count", len(UNCERTAIN), "probes", "len(uncertain)", "-")
    add("w1 label suspect count", len(CONFIRMED + UNCERTAIN), "probes", "confirmed + uncertain", "-")
    for f in CONFIRMED + UNCERTAIN:
        s, i = f.rsplit("_", 1)
        add(f"w1 label image index {s}", int(i), "LFW file number", f"{f}.jpg", "-")
    add("w1 label frr count threshold pct", 5, "%", "definition: frr_n_gt_5pct counts subjects with FRR > 5%", "-")
    # seed labels the text may print ('bootstrap seed 557'), read from the manifest's seeds column
    import re
    seeds = {"boot": set(), "far": set()}
    for k, r in M.items():
        if k.startswith("closure_pct n="):
            for kind in seeds:
                seeds[kind] |= {int(x) for x in re.findall(kind + r" (\d+)", r["seeds"])}
    add("w1 bootstrap seeds", sorted(seeds["boot"]), "list", "every 'boot N' in manifest.csv closure rows (542+n)", "-")
    add("w1 FAR seeds", sorted(seeds["far"]), "list", "every 'far N' in manifest.csv closure rows (342+n)", "-")
    write_manifest(OUT / "part_limitations.csv", rows)
    json.dump({"git": st, "closures": {f"{v} n={n}": c for (v, n), c in clo.items()},
               "all_positive": {v: allpos(v) for v in VARIANTS}, "threshold_window": [lo, hi], "threshold_window_probe": [plo, phi]},
              open(OUT / "limitations_numbers.json", "w"), indent=1)
    print(f"{len(rows)} rows; stable-closure threshold window ({lo}, {hi}]")


if __name__ == "__main__":
    main()
