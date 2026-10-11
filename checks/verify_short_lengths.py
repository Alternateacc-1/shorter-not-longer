#!/usr/bin/env python3
"""Recompute every length and compare with the recorded values in
checks/expected/regen_results.json and short_sweep_canonical.json (290 checks).

This tests that a fresh run reproduces the recorded numbers exactly: it is a
determinism and consistency check, and it reuses the library, so it is not an
independent reimplementation and does not prove the method correct. N_eff and
the Ledoit-Wolf eta are second-order statistics with no min-entropy guarantee.

The sampled false-accept rate is re-run only at n=15 and n=24 (the two lengths
the paper quotes); at the other lengths the accept counts are checked against
the full impostor set instead, which is cheap and has no sampling error.

    python checks/verify_short_lengths.py      (from the repository root)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuzzycommit.common import (GLOBAL_SEED, load_and_center, get_W, build_subject_info,
                                build_subject_info_random, subject_list_hash, probe_set_hash)
from fuzzycommit.bit_errors import compute_sweep_row
from fuzzycommit.entropy import compute_eta
from fuzzycommit.false_accept import compute_neff, run_exhaustive_far, run_sampled_far, compute_analytical_far
from fuzzycommit.bootstrap import run_bootstrap
from fuzzycommit.weighting import run_capped_and_weighted, run_tertiles

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected")
SWEEP = json.load(open(os.path.join(RES, "regen_results.json")))
SHORT = json.load(open(os.path.join(RES, "short_sweep_canonical.json")))
FAR_LENGTHS = [15, 24]

class V:
    def __init__(self): self.p = self.f = 0; self.bad = []
    def chk(self, name, got, exp):
        ok = (got == exp)
        if ok: self.p += 1
        else:
            self.f += 1; self.bad.append((name, got, exp))
        return ok

v = V()
emb, cen = load_and_center()
subs = sorted(cen.keys()); W = get_W()
print("verify_short_lengths: comparing with checks/expected/")
print("=" * 72)

c = SWEEP["corpus"]
info24, _ = build_subject_info(cen, W, 24, subs)
v.chk("corpus n_subjects", len(subs), c["n_subjects"])
v.chk("corpus n_images", sum(len(x) for x in emb.values()), c["n_images"])
v.chk("corpus n_probes", sum(len(x[3]) for x in info24), c["n_probes"])
v.chk("corpus subject hash", subject_list_hash(subs), c["subject_list_sha256"])
v.chk("corpus probe hash", probe_set_hash(info24), c["probe_set_sha256"])

for tag, table, keys in [
    ("sweep", {int(k): x for k, x in SWEEP["sweep"].items()},
     [("beta", "beta", 10), ("m", "m", None), ("t_delivered", "t_delivered", None),
      ("parity", "parity", None), ("k", "k", None),
      ("decode_fail_pct", "decode_fail_pct", 2), ("neff", "neff", 4),
      ("eta_lw", "eta_lw", 4), ("residual_neff", "residual_neff", 3),
      ("residual_eta", "residual_eta", 3), ("far_analytical", "far_analytical", None)]),
    ("short", {int(k): x for k, x in SHORT.items()},
     [("beta", "beta", 10), ("t", "t", None), ("parity", "parity", None), ("k", "k", None),
      ("fail", "fail", 3), ("neff", "neff", 4), ("eta_lw", "eta_lw", 4),
      ("resid", "resid", 3), ("resid_eta", "resid_eta", 3),
      ("far_anal", "far_anal", None)])]:
    for n in sorted(table):
        info, sels = build_subject_info(cen, W, n, subs)
        row = compute_sweep_row(info, n)
        ne = float(compute_neff(cen, W, n, info))
        _, elw, _ = compute_eta(cen, W, sels, n)
        got = {"beta": float(row["beta"]), "m": row["m"], "t_delivered": int(row["t"]),
               "t": int(row["t"]), "parity": int(row["parity"]), "k": int(row["k"]),
               "decode_fail_pct": 100 * float(row["decode_fail_frac"]),
               "fail": 100 * float(row["decode_fail_frac"]),
               "neff": ne, "eta_lw": float(elw),
               "residual_neff": ne - row["parity"], "resid": ne - row["parity"],
               "residual_eta": float(elw) * n - row["parity"],
               "resid_eta": float(elw) * n - row["parity"],
               "far_analytical": compute_analytical_far(n, row["t"]),
               "far_anal": compute_analytical_far(n, row["t"])}
        for gk, ek, nd in keys:
            g = got[gk]; e = table[n][ek]
            if nd is not None and isinstance(g, float):
                g = round(g, nd); e = round(float(e), nd)
            v.chk("%s n=%d %s" % (tag, n, gk), g, e)
        exh = run_exhaustive_far(cen, W, n, row["t"], subs)["far_pair_uniform"]
        v.chk("%s n=%d far_exhaustive" % (tag, n), exh,
              table[n].get("far_exhaustive", table[n].get("far_exh")))
        if n in FAR_LENGTHS:
            acc, far, _, _ = run_sampled_far(info, cen, W, n, row["t"], seed=GLOBAL_SEED + 300 + n)
            v.chk("%s n=%d far_accepts (1e6 re-run)" % (tag, n), int(acc),
                  table[n].get("far_accepts", table[n].get("accepts")))
        pct, _ = run_bootstrap(info, n, ne, cen, W)
        exp_c = table[n].get("closure", table[n].get("bootstrap", {}).get("closure_pct"))
        v.chk("%s n=%d closure" % (tag, n), round(pct, 1), exp_c)

for a in SWEEP["ablation"]:
    n = a["n"]
    if a["arm"] == "A": inf, sl = build_subject_info(cen, W, n, subs)
    else: inf, sl = build_subject_info_random(cen, W, n, subs, seed_base=GLOBAL_SEED + 2000)
    row = compute_sweep_row(inf, n); ne = float(compute_neff(cen, W, n, inf))
    _, elw, _ = compute_eta(cen, W, sl, n)
    v.chk("ablation %s n=%d beta" % (a["arm"], n), round(float(row["beta"]), 10), round(a["beta"], 10))
    v.chk("ablation %s n=%d t" % (a["arm"], n), int(row["t"]), a["t_delivered"])
    v.chk("ablation %s n=%d parity" % (a["arm"], n), int(row["parity"]), a["parity"])
    v.chk("ablation %s n=%d k" % (a["arm"], n), int(row["k"]), a["k"])
    v.chk("ablation %s n=%d neff" % (a["arm"], n), round(ne, 4), a["neff"])
    v.chk("ablation %s n=%d eta_lw" % (a["arm"], n), round(float(elw), 4), a["eta_lw"])

rw = run_capped_and_weighted(cen, W, 24)
for k in ["baseline", "capped", "subject_weighted"]:
    for f in ["beta", "t", "parity", "k", "residual"]:
        v.chk("robust %s %s" % (k, f), rw[k][f], SWEEP["robustness"][k][f])
for got, exp in zip(run_tertiles(cen, emb, W, 24), SWEEP["tertiles"]):
    for f in ["beta", "t", "parity", "neff", "residual", "closes"]:
        v.chk("tertile %s %s" % (exp["label"][:10], f), got[f], exp[f])

print("  %d PASS, %d FAIL" % (v.p, v.f))
for name, g, e in v.bad[:20]:
    print("    FAIL %-44s got %s expected %s" % (name, g, e))
print("=" * 72)
print("  Verifies pipeline-vs-recorded-output equivalence ONLY.")
print("  Finding A-4 stands: this says nothing about semantic correctness.")
sys.exit(0 if v.f == 0 else 1)
