"""Label-error sensitivity. Reported only: the main results keep every probe.

Variants:   base  all 5,425 images
            drop2 the two confirmed mislabelled probes removed
                  (Michael_Schumacher_0008, Mahmoud_Abbas_0012)
            drop4 all four suspects removed (+ Naomi_Watts_0022, Abdullah_Gul_0016)
Primary: the images are removed from the data and the population mean is recomputed
(what a re-run without them does). Secondary ('fixedmean'): the original centring is
kept, which isolates the probes themselves. The removed images also leave the impostor
set, so N_eff and FAR move as well.

At n = 9, 15, 24: beta, t, parity, decode failure, residual (probe / capped / subject),
closure (probe and subject, 1,000 replicates, seed 542+n), per-subject FRR summary.
Recommendation check: the subject-weighted band edge over the 17 measured lengths
(the paper's choice, n=15) and the n=9 bound (mean and minimum, both weightings).

    python -m analyses.label_errors
Writes results/supplementary/label_errors.json and part_label_errors.csv."""
import json

import numpy as np

from analyses.helpers import out_dir, require_committed, write_manifest
from analyses import crosscheck as V

OUT = out_dir("supplementary")

DROP = {"drop2": ["Michael_Schumacher_0008.jpg", "Mahmoud_Abbas_0012.jpg"],
        "drop4": ["Michael_Schumacher_0008.jpg", "Mahmoud_Abbas_0012.jpg", "Naomi_Watts_0022.jpg", "Abdullah_Gul_0016.jpg"]}
LFW = V._common.LFW_DIR   # the configured LFW tree (LFW_DIR, paths.conf, or the kagglehub default)
MEASURED = [9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 32, 40, 48, 56, 64]
DETAIL = (9, 15, 24)
WGT = ("probe", "capped", "subject")


def files_of(s):
    return sorted(p.name for p in (LFW / s).iterdir() if p.suffix.lower() == ".jpg")


def variant(emb, cen_full, drop, fixed_mean):
    idx = {}
    for f in drop:
        s = f.rsplit("_", 1)[0]; i = files_of(s).index(f)
        assert i >= 5, f"{f} is an enrollment image"
        idx.setdefault(s, []).append(i)
    if fixed_mean:
        return {s: (np.delete(v, idx[s], axis=0) if s in idx else v) for s, v in cen_full.items()}
    e = {s: (np.delete(v, idx[s], axis=0) if s in idx else v) for s, v in emb.items()}
    mu = np.concatenate([e[s] for s in sorted(e)]).mean(axis=0)
    return {s: v - mu for s, v in e.items()}


def evaluate(cen, W):
    S = V.Setup(cen, W); res = {"images": int(S.sizes.sum()), "probes": int(S.nprobe.sum()), "lengths": {}}
    for n in MEASURED:
        L = V.Length(S, n); ne = L.neff(); r = {"neff": ne}
        for w in WGT:
            p, t = L.choose(w)
            r[w] = {"beta": L.beta(w), "t": int(t), "parity": int(p), "decode_fail": L.fail(t, w), "residual": ne - p}
        if n in DETAIL:
            draws = V.bootstrap_draws(n, S.NS)
            nes = np.array([L.neff(c) for c in draws])
            for w in ("probe", "subject"):
                par = np.array([L.choose(w, c)[0] for c in draws], float)
                r[w]["closure_pct"] = 100 * float(((n - par > 0) & (nes - par > 0)).mean())
            t = r["probe"]["t"]; frr = np.array([(h > t).mean() for h in L.hd])
            r["frr"] = {"mean": float(frr.mean()), "median": float(np.median(frr)), "p90": float(np.percentile(frr, 90)),
                        "p99": float(np.percentile(frr, 99)), "max": float(frr.max()), "n_gt_5pct": int((frr > .05).sum()),
                        "n_gt_10pct": int((frr > .10).sum())}
        if n == 9:
            fl = L.mcv_floors()
            for w in ("probe", "subject"):
                r[w]["bound_mean"] = float(fl.mean() - r[w]["parity"]); r[w]["bound_min"] = float(fl.min() - r[w]["parity"])
        res["lengths"][n] = r
    sw_pos = [n for n in MEASURED if res["lengths"][n]["subject"]["residual"] > 0 and n - res["lengths"][n]["subject"]["parity"] > 0]
    res["band_edge"] = max(sw_pos); res["subject_positive"] = sw_pos
    res["n15_positive_all_weightings"] = all(res["lengths"][15][w]["residual"] > 0 for w in WGT)
    res["n9_bound_positive_both"] = all(res["lengths"][9][w]["bound_mean"] > 0 and res["lengths"][9][w]["bound_min"] > 0
                                        for w in ("probe", "subject"))
    return res


def main():
    st = require_committed()
    M = V.manifest(); emb, cen = V.load(); W, _ = V.W_of(43)
    out = {"git": st, "drop": DROP}
    out["base"] = evaluate(cen, W)
    for n in DETAIL:                                      # the baseline must be the manifest
        for w in WGT:
            sfx = "" if w == "probe" else f" {w}"
            assert out["base"]["lengths"][n][w]["parity"] == M[f"parity n={n}{sfx}"]
            assert abs(out["base"]["lengths"][n][w]["residual"] - M[f"residual n={n}{sfx}"]) < 1e-9
    for name, drop in DROP.items():
        out[name] = evaluate(variant(emb, cen, drop, fixed_mean=False), W)
        out[name + "_fixedmean"] = evaluate(variant(emb, cen, drop, fixed_mean=True), W)
        print(f"{name}: images {out[name]['images']}, band edge {out[name]['band_edge']}", flush=True)
    changes = []
    for v in ("drop2", "drop4", "drop2_fixedmean", "drop4_fixedmean"):
        for n in MEASURED:
            for w in WGT:
                a, b = out["base"]["lengths"][n][w], out[v]["lengths"][n][w]
                for k in ("t", "parity"):
                    if a[k] != b[k]:
                        changes.append(f"{v} n={n} {w} {k}: {a[k]} -> {b[k]}")
                if (a["residual"] > 0) != (b["residual"] > 0):
                    changes.append(f"{v} n={n} {w} residual sign: {a['residual']:+.3f} -> {b['residual']:+.3f}")
        for k in ("band_edge", "n15_positive_all_weightings", "n9_bound_positive_both"):
            if out["base"][k] != out[v][k]:
                changes.append(f"{v} {k}: {out['base'][k]} -> {out[v][k]}")
    out["changes"] = changes
    out["verdict"] = "no change" if not changes else "CHANGE"
    json.dump(out, open(OUT / "label_errors.json", "w"), indent=1, default=V._jsonable)
    rows = []
    def add(q, v, u, f):
        rows.append({"quantity": q, "value": repr(v) if isinstance(v, float) else v, "units": u,
                     "script": "analyses/label_errors.py", "function": f, "seeds": "projection 43; boot 542+n",
                     "commit": st["commit"][:12]})
    for v in ("drop2", "drop4", "drop2_fixedmean", "drop4_fixedmean"):
        add(f"w0 label {v} images", out[v]["images"], "count", "images after removal")
        add(f"w0 label {v} band edge", out[v]["band_edge"], "n", "longest measured n with subject residual > 0")
        add(f"w0 label {v} n=9 bound positive both weightings", str(out[v]["n9_bound_positive_both"]), "bool", "mean and min > 0")
        for n in DETAIL:
            r = out[v]["lengths"][n]
            add(f"w0 label {v} n={n} N_eff", r["neff"], "bits", "p(1-p)/var over impostor HDs")
            for w in WGT:
                for k in ("beta", "t", "parity", "decode_fail", "residual"):
                    add(f"w0 label {v} n={n} {w} {k}", r[w][k], "-", "crosscheck.Length")
            for w in ("probe", "subject"):
                add(f"w0 label {v} n={n} {w} closure_pct", r[w]["closure_pct"], "%", "1,000 replicates, N_eff per replicate")
            for k, x in r["frr"].items():
                add(f"w0 label {v} n={n} frr_{k}", x, "-", "per-subject decode failure at the probe-weighted t")
    add("w0 label verdict", out["verdict"], "-", "any t, parity, residual sign, band edge or n=9 bound change")
    write_manifest(OUT / "part_label_errors.csv", rows)
    print("verdict:", out["verdict"]); print("\n".join(changes) if changes else "(no t, parity, sign, band-edge or bound change)")


if __name__ == "__main__":
    main()
