"""Step 5: the distribution-free bound on the 4-bit plateau (n = 8..11) for all
five projection seeds, under probe and subject weighting: mean, minimum over the
311 index sets, and minimum after Bonferroni (alpha = 0.01/311).

This is what places the second operating point at n = 9: the longest length at
which the bound is positive for every index set, both weightings and every seed.

    python -m pipeline.step5_bound_seeds
"""
import time

from pipeline.provenance import EMB_NPZ, OUT_MAIN, Log, dump, require_committed_code, sha256_file

SEEDS = (43, 101, 202, 303, 404)
LENGTHS = (8, 9, 10, 11)


def main():
    st = require_committed_code()
    log = Log("step5_bound_seeds")
    import json
    import numpy as np
    from scipy.stats import beta as bdist
    env = json.load(open(OUT_MAIN / "step1_embed.json", encoding="utf-8"))
    assert sha256_file(EMB_NPZ) == env["embeddings_npz_sha256"]
    import fuzzycommit.common as common
    common.DATA_DIR = OUT_MAIN
    from fuzzycommit.common import (DECODE_FAILURE_TARGET as TARGET, load_and_center, get_W, build_subject_info,
                            select_capacity, bch_parity, bch_true_t, compute_bch_m)
    from fuzzycommit.entropy import enrolled_bitstrings
    emb, cen = load_and_center(); subs = sorted(cen); NS = len(subs)
    R = {"git": st, "seeds": SEEDS, "lengths": LENGTHS, "rows": {}}
    t0 = time.time()
    for ps in SEEDS:
        W = get_W(ps)
        TB = enrolled_bitstrings(cen, W).astype(np.uint8)
        for n in LENGTHS:
            info, sels = build_subject_info(cen, W, n, subs)
            hds = [x[3] for x in info]; m = compute_bch_m(n)
            _, p_pw, _ = select_capacity(np.concatenate(hds), n)
            p_sw = next(bch_parity(m, t) for t in range(n + 1)
                        if float(np.mean(np.array([float(np.mean(h > bch_true_t(m, t))) for h in hds]))) < TARGET)
            ks = np.array([np.unique(TB[:, s], axis=0, return_counts=True)[1].max() for _, s in sels])
            fl = -np.log2(bdist.ppf(0.99, ks + 1, NS - ks))
            flb = -np.log2(bdist.ppf(1 - 0.01 / NS, ks + 1, NS - ks))
            row = {}
            for w, p in (("probe", p_pw), ("subject", p_sw)):
                row[w] = {"parity": int(p), "bound_mean": float(fl.mean() - p), "bound_min": float(fl.min() - p),
                          "bound_min_bonferroni": float(flb.min() - p)}
            R["rows"][f"{ps}/{n}"] = row
            log(f"seed {ps} n={n}: " + " | ".join(
                f"{w} p={r['parity']} mean {r['bound_mean']:+.3f} min {r['bound_min']:+.3f} bonf {r['bound_min_bonferroni']:+.3f}"
                for w, r in row.items()))
    both = {n: all(R["rows"][f"{ps}/{n}"][w]["bound_mean"] > 0 for ps in SEEDS for w in ("probe", "subject"))
            for n in LENGTHS}
    R["positive_both_weightings_all_seeds"] = both
    R["longest_positive_both_weightings_all_seeds"] = max([n for n in LENGTHS if both[n]] or [None])
    R["seed43_positive_both_weightings"] = {n: all(R["rows"][f"43/{n}"][w]["bound_mean"] > 0
                                                   for w in ("probe", "subject")) for n in LENGTHS}
    R["wall_seconds"] = round(time.time() - t0, 1)
    dump(R, "step5_bound_seeds.json")
    log("positive under both weightings for all five seeds:", both)


if __name__ == "__main__":
    main()
