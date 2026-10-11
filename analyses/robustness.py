"""Robustness of the single-draw results (paper Sect. 4.2 and 5.2).

Several numbers in the paper come from one random choice: one bootstrap seed per length,
one projection matrix (seed 43), one enrollment set (the first 5 images of each subject),
one draw of 800 outside images for the centering check, one random-selection arm, one
noise-only control and one split for the lookalike attacker. This script repeats each of
them with other seeds or draws and reports how far the results move.

  bootstrap    60 bootstrap seeds (the logged 542+n and 59 others), lengths 8-16, the paper's
               data otherwise. Every length with a lower false-accept rate than n=15 closes in
               at most 54% of replicates (results/codes_lookalike/manifest.csv), so lengths
               above 16 cannot change the rule's choice.
  projection   20 projection matrices: the five seeds of step 4 (43, 101, 202, 303, 404)
               and seeds 1-15; the rule over shortened-BCH lengths 8-24 and 32.
  enrollment   20 choices of the five enrollment images: draw 0 is the paper's order (the
               first five images), draws 1-19 shuffle each subject's images. The probes, and
               so the first 10 probes used by capped weighting, change with them.
  centering    21 draws of the 800 outside images for the external-mean centering check:
               seed 7 (the paper's) and seeds 1-6 and 8-21, from one embedding of the whole
               outside pool (7,808 images).
  random_arm   20 seeds of the uniform-random selection arm (Table 2, Arm B): the N_eff gap
               at n = 16, 24 and 32.
  noise        8 seed pairs of the noise-only control (Sect. 5.2, "Attribution").
  lookalike    20 seeds of the half-A / half-B split behind the realistic lookalike
               attacker at n = 15 (Sect. 4.3), with its binomial null.

The rule, the bootstrap and the false-accept rate are those of analyses/centering_check
(full-set pair-uniform FAR at the largest delivered t; shortened BCH). The first seed or
draw of every experiment is the paper's, and the script stops if it does not reproduce the
published value.

Run after step 2 and the centering check (it needs results/pipeline/embeddings_cache.npz,
the LFW tree and models/w600k_r50.onnx):
    python -m analyses.robustness
Writes results/robustness/robustness.json and results/robustness/manifest.csv
(about 45 minutes on a 12-thread laptop, 15 of them embedding the outside pool).
"""
import csv
import json
import time

import numpy as np

from analyses.centering_check import (apply_rule, closure, draw_indices, evaluate, external_list_hash,
                                      load_corpus, outside_pool, row, setup_length)
from analyses.helpers import PIPELINE_OUT, out_dir, require_committed, write_manifest
from fuzzycommit.common import GLOBAL_SEED, build_subject_info, build_subject_info_random, get_W
from fuzzycommit.false_accept import compute_analytical_far, compute_neff

OUT = out_dir("robustness")
SCRIPT = "analyses/robustness.py"

BOOT_SEEDS = ["paper"] + list(range(1, 60))           # "paper" = GLOBAL_SEED + 500 + n
BOOT_LENGTHS = list(range(8, 17))
PROJ_SEEDS = [43, 101, 202, 303, 404] + list(range(1, 16))
ENROLL_DRAWS = list(range(20))                         # 0 = the first five images
CENTER_SEEDS = [7] + [s for s in range(1, 22) if s != 7]
ARM_B_BASES = [GLOBAL_SEED + 2000] + [50_000 + 1000 * k for k in range(1, 20)]
NOISE_SEEDS = [(11, 500)] + [(11 + 100 * k, 500 + 10_000 * k) for k in range(1, 8)]
SPLIT_SEEDS = [3042] + [3042 + 1000 * k for k in range(1, 20)]


def summary(rows, eligible, pick):
    r15 = rows[15]
    return {"pick": pick, "eligible": eligible, "n15_residual": r15["residual"],
            "n15_closure": r15["closure_pct"], "n15_t": r15["t"],
            "eligible_above_15": [n for n in eligible if n > 15],
            "n24_probe_residual": rows[24]["residual"]["probe"] if 24 in rows else None}


def bootstrap_seeds(emb, subs, W):
    mu = np.concatenate([emb[s] for s in subs]).mean(axis=0)
    cen = {s: emb[s] - mu for s in subs}
    setups = {n: setup_length(cen, subs, W, n) for n in BOOT_LENGTHS}
    out = []
    for sd in BOOT_SEEDS:
        rows = {n: row(L, closure(L, GLOBAL_SEED + 500 + n if sd == "paper" else 10_000 * sd + n))
                for n, L in setups.items()}
        eligible, pick = apply_rule(rows)
        out.append({"seed": sd, "pick": pick, "eligible": eligible,
                    "closure": {n: rows[n]["closure_pct"] for n in rows}})
        print(f"bootstrap seed {sd}: n=15 closure {rows[15]['closure_pct']}, rule picks {pick}", flush=True)
    assert out[0]["closure"][15]["subject"] == 96.1 and out[0]["pick"] == 15, "logged seed does not reproduce"
    return out


def projection_seeds(emb, subs):
    mu = np.concatenate([emb[s] for s in subs]).mean(axis=0)
    out = []
    for sd in PROJ_SEEDS:
        rows, eligible, pick = evaluate(emb, subs, mu, get_W(sd))
        out.append(dict(seed=sd, **summary(rows, eligible, pick)))
        print(f"projection seed {sd}: rule picks {pick}, eligible {eligible}", flush=True)
    assert out[0]["pick"] == 15 and out[0]["n15_closure"]["subject"] == 96.1, "seed 43 does not reproduce"
    return out


def enrollment_draws(emb, subs, W):
    mu = np.concatenate([emb[s] for s in subs]).mean(axis=0)   # the corpus mean does not depend on order
    out = []
    for d in ENROLL_DRAWS:
        perm = emb if d == 0 else {s: emb[s][np.random.RandomState(70_000 + 1000 * d + i).permutation(len(emb[s]))]
                                   for i, s in enumerate(subs)}
        rows, eligible, pick = evaluate(perm, subs, mu, W)
        out.append(dict(draw=d, **summary(rows, eligible, pick)))
        print(f"enrollment draw {d}: rule picks {pick}, eligible {eligible}", flush=True)
    assert out[0]["pick"] == 15 and out[0]["n15_closure"]["subject"] == 96.1, "first-five order does not reproduce"
    return out


def centering_draws(emb, subs, W):
    from fuzzycommit.embedding import extract_embeddings
    pool = outside_pool()
    tag = external_list_hash(pool)[:12]
    cache = OUT / f"outside_pool_{tag}.npz"
    if not cache.exists():
        extract_embeddings({"pool": pool}, cache_path=cache)
    pe = np.load(cache, allow_pickle=True)["embeddings"].item()["pool"]
    assert len(pe) == len(pool)
    out = []
    for sd in CENTER_SEEDS:
        idx = draw_indices(len(pool), sd)
        rows, eligible, pick = evaluate(emb, subs, pe[idx].mean(axis=0), W)
        out.append(dict(seed=sd, draw_sha256=external_list_hash([pool[i] for i in idx]),
                        **summary(rows, eligible, pick)))
        print(f"centering draw {sd}: rule picks {pick}, eligible {eligible}", flush=True)
    paper = json.load(open(out_dir("centering") / "centering_check.json", encoding="utf-8"))
    assert out[0]["draw_sha256"] == paper["external_list_sha256"], "seed 7 is not the centering check's draw"
    assert out[0]["pick"] == paper["variants"]["X_external_mean"]["rule_selects"] == 15
    return {"pool_images": len(pool), "pool_sha256": external_list_hash(pool), "draws": out}


def random_arm(cen, subs, W, sweep):
    neA = {n: sweep["ablation"][f"A{n}"]["neff"] for n in (16, 24, 32)}
    out = []
    for base in ARM_B_BASES:
        r = {}
        for n in (16, 24, 32):
            info, _ = build_subject_info_random(cen, W, n, subs, seed_base=base)
            neB = float(compute_neff(cen, W, n, info))
            r[n] = {"neff_B": neB, "gap": neB - neA[n]}
        out.append({"seed_base": base, "rows": r})
        print(f"random arm seed base {base}: gaps {[round(r[n]['gap'], 2) for n in r]}", flush=True)
    assert abs(out[0]["rows"][24]["neff_B"] - sweep["ablation"]["B24"]["neff"]) < 1e-9, "Arm B does not reproduce"
    return out


def noise_control(subs, sizes, W, sweep):
    """Same construction as step 2: isotropic Gaussian subjects, sigma bisected until the
    probe bit-error rate at n = 24 matches LFW's, then the pair-uniform FAR at t = 3."""
    NS = len(subs)
    owner = np.repeat(np.arange(NS), sizes)
    beta24 = sweep["lengths_detail"]["24"]["weightings"]["probe"]["beta"]
    out = []
    for smu, snoise in NOISE_SEEDS:
        mus = np.random.RandomState(smu).randn(NS, 512)
        noise = [np.random.RandomState(snoise + i).randn(sizes[i], 512) for i in range(NS)]

        def synth(sig, n=24, t=3):
            V = np.concatenate([mus[i] + sig * noise[i] for i in range(NS)]); V = V - V.mean(0)
            pj = V @ W; bt = pj > 0; s0 = 0; hs = []; Ar = np.zeros((NS, NS))
            for i in range(NS):
                pr = pj[s0:s0 + 5]
                sel = np.lexsort((-np.abs(pr).mean(0), -np.abs((pr > 0).mean(0) - .5) * 2))[:n]
                tm = (pr.mean(0) > 0)[sel]
                hs.append((bt[s0 + 5:s0 + sizes[i]][:, sel] != tm).sum(1))
                h = (bt[:, sel] != tm).sum(1)
                Ar[i] = np.bincount(owner, (h <= t).astype(float), NS) / sizes; s0 += sizes[i]
            np.fill_diagonal(Ar, np.nan)
            return float(np.concatenate(hs).mean() / n), float(np.nanmean(Ar))
        lo, hi = 0.05, 2.0
        for _ in range(22):
            mid = (lo + hi) / 2
            b, _ = synth(mid)
            lo, hi = (mid, hi) if b < beta24 else (lo, mid)
        bn, fn = synth((lo + hi) / 2)
        out.append({"mu_seed": smu, "noise_seed": snoise, "beta": bn, "ratio": fn / compute_analytical_far(24, 3)})
        print(f"noise control seeds {smu}/{snoise}: ratio {out[-1]['ratio']:.3f}", flush=True)
    assert abs(out[0]["ratio"] - sweep["noise_control"]["ratio"]) < 1e-12, "noise control does not reproduce"
    return out


def lookalike_splits(emb, cen, subs, sizes, W, sweep, far_sampled):
    """The realistic attacker of analyses/lookalike_attack at n = 15 for other split seeds."""
    n, t = 15, sweep["lengths_detail"]["15"]["weightings"]["probe"]["t"]
    NS = len(subs)
    owner = np.repeat(np.arange(NS), sizes)
    allbits = np.concatenate([cen[s] for s in subs], axis=0) @ W > 0
    off = ~np.eye(NS, dtype=bool)
    info, _ = build_subject_info(cen, W, n, subs)
    oks = [(allbits[:, sel] != tmpl.astype(bool)).sum(axis=1) <= t for _, sel, tmpl, _ in info]
    acc = np.array([np.bincount(owner, ok.astype(float), NS) for ok in oks]); acc[~off] = 0
    pv_comp = acc.sum(axis=1) / (sizes.sum() - sizes)

    def unit(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    vknow = unit(np.array([unit(emb[s][5:]).mean(axis=0) for s in subs]))
    st0 = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = []
    for sd in SPLIT_SEEDS:
        rng = np.random.RandomState(sd); halfA, halfB = [], []
        for s in subs:
            perm = rng.permutation(len(emb[s])); h = len(perm) // 2
            halfA.append(np.sort(perm[:h])); halfB.append(np.sort(perm[h:]))
        uA = unit(np.array([unit(emb[s][halfA[i]]).mean(axis=0) for i, s in enumerate(subs)]))
        cosm = vknow @ uA.T; np.fill_diagonal(cosm, -np.inf); ustar = cosm.argmax(axis=1)
        inB = np.zeros(len(owner), bool)
        for i in range(NS):
            inB[st0[i] + halfB[i]] = True
        sizesB = np.array([len(b) for b in halfB])
        accB = np.array([np.bincount(owner, (ok & inB).astype(float), NS) for ok in oks]); accB[~off] = 0
        real = (accB / sizesB[None, :])[np.arange(NS), ustar]
        nrng = np.random.RandomState(sd + 1); nul = np.zeros(NS)
        for _ in range(200):
            X = nrng.binomial(np.broadcast_to(sizes, (NS, NS)), pv_comp[:, None]) / sizes[None, :]
            nul += np.where(off, X, -1).max(axis=1) / 200
        out.append({"seed": sd, "mean": float(real.mean()), "median": float(np.median(real)),
                    "ratio_to_far_sampled": float(real.mean() / far_sampled),
                    "victims_gt_10pct": int((real > .10).sum()), "victims_gt_50pct": int((real > .50).sum()),
                    "oracle_null_mean": float(nul.mean())})
        print(f"lookalike split {sd}: mean {out[-1]['mean']:.4f}", flush=True)
    return out


def manifest_rows(R, st):
    rows = []

    def add(q, v, u, f, seeds):
        rows.append({"quantity": q, "value": repr(float(v)) if isinstance(v, float) else v, "units": u,
                     "script": SCRIPT, "function": f, "seeds": seeds, "commit": st["commit"][:12]})

    def rng(xs):
        return [round(min(xs), 2), round(max(xs), 2)]

    B = R["bootstrap"]; c15 = [b["closure"][15]["subject"] for b in B]
    bs = "542+n (logged) and 10000*k+n, k=1..59"
    add("robust bootstrap seeds", len(B), "count", "bootstrap_seeds", bs)
    add("robust bootstrap n=15 subject closure logged seed", c15[0], "%", "bootstrap_seeds", "542+n")
    add("robust bootstrap n=15 subject closure pooled", float(np.mean(c15)), "%", "mean over seeds (1000 replicates each)", bs)
    add("robust bootstrap n=15 subject closure range", rng(c15), "% [min,max]", "over seeds", bs)
    add("robust bootstrap seeds above logged n=15 subject closure", sum(c > c15[0] for c in c15), "count", "over seeds", bs)
    add("robust bootstrap seeds n=15 below 95%", sum(c < 95 for c in c15), "count", "over seeds", bs)
    for k in (14, 15):
        add(f"robust bootstrap seeds rule picks n={k}", sum(b["pick"] == k for b in B), "count", "apply_rule", bs)
    add("robust bootstrap other picks", sum(b["pick"] not in (14, 15) for b in B), "count", "apply_rule", bs)
    c9 = [b["closure"][9]["capped"] for b in B]
    add("robust bootstrap n=9 capped closure pooled", float(np.mean(c9)), "%", "mean over seeds", bs)

    P, E = R["projection"], R["enrollment"]
    for name, X, sd in (("projection", P, "projection seeds 43,101,202,303,404,1-15"),
                        ("enrollment", E, "enrollment draws 0-19 (0 = first five images)")):
        add(f"robust {name} runs", len(X), "count", f"{name}", sd)
        add(f"robust {name} rule picks n=15", sum(x["pick"] == 15 for x in X), "count", "apply_rule", sd)
        add(f"robust {name} rule picks n=14", sum(x["pick"] == 14 for x in X), "count", "apply_rule", sd)
        add(f"robust {name} rule picks none", sum(x["pick"] is None for x in X), "count", "apply_rule", sd)
        add(f"robust {name} rule picks list", json.dumps([x["pick"] for x in X]), "list", "apply_rule", sd)
        add(f"robust {name} n=15 eligible", sum(15 in x["eligible"] for x in X), "count", "apply_rule", sd)
        add(f"robust {name} n=15 positive under every weighting", sum(min(x["n15_residual"].values()) > 0 for x in X),
            "count", "min over weightings of N_eff - parity", sd)
        add(f"robust {name} n=15 min-over-weighting residual range", rng([min(x["n15_residual"].values()) for x in X]),
            "bits [min,max]", "evaluate", sd)
        add(f"robust {name} runs with an eligible length above 15", sum(bool(x["eligible_above_15"]) for x in X),
            "count", "apply_rule", sd)
    # the same counts without the reference run of each experiment (seed 43, draw 0), which is the
    # paper's own configuration and would otherwise be counted twice
    redraws = P[1:] + E[1:]
    rs = "projection seeds 101,202,303,404,1-15 and enrollment draws 1-19"
    add("robust redraws count", len(redraws), "count", "projection[1:] + enrollment[1:]", rs)
    add("robust redraws rule picks n=15", sum(x["pick"] == 15 for x in redraws), "count", "apply_rule", rs)
    add("robust redraws rule picks n=14", sum(x["pick"] == 14 for x in redraws), "count", "apply_rule", rs)
    add("robust redraws rule picks another length", sum(x["pick"] not in (14, 15, None) for x in redraws),
        "count", "apply_rule", rs)
    add("robust redraws rule picks none", sum(x["pick"] is None for x in redraws), "count", "apply_rule", rs)
    add("robust redraws n=15 positive under every weighting", sum(min(x["n15_residual"].values()) > 0 for x in redraws),
        "count", "evaluate", rs)
    add("robust redraws n=15 min-over-weighting residual range",
        rng([min(x["n15_residual"].values()) for x in redraws]), "bits [min,max]", "evaluate", rs)
    both = P + E
    add("robust projection+enrollment runs", len(both), "count", "projection + enrollment", "both")
    add("robust projection+enrollment rule picks n=15", sum(x["pick"] == 15 for x in both), "count", "apply_rule", "both")
    add("robust projection+enrollment rule picks n=14", sum(x["pick"] == 14 for x in both), "count", "apply_rule", "both")
    add("robust projection+enrollment rule picks another length",
        sum(x["pick"] not in (14, 15, None) for x in both), "count", "apply_rule", "both")
    add("robust projection+enrollment other picks range",
        rng([x["pick"] for x in both if x["pick"] not in (14, 15, None)]), "n [min,max]", "apply_rule", "both")
    add("robust projection+enrollment rule picks none", sum(x["pick"] is None for x in both), "count", "apply_rule", "both")
    add("robust projection+enrollment n=15 positive under every weighting",
        sum(min(x["n15_residual"].values()) > 0 for x in both), "count", "evaluate", "both")
    add("robust projection+enrollment n=15 min-over-weighting residual range",
        rng([min(x["n15_residual"].values()) for x in both]), "bits [min,max]", "evaluate", "both")

    C = R["centering"]["draws"]; cs = "random.Random(seed), seeds 7 and 1-6, 8-21"
    add("robust centering outside pool images", R["centering"]["pool_images"], "images", "outside_pool", "-")
    add("robust centering draws", len(C), "count", "centering_draws", cs)
    add("robust centering rule picks n=15", sum(c["pick"] == 15 for c in C), "count", "apply_rule", cs)
    add("robust centering other picks", json.dumps(sorted(c["pick"] for c in C if c["pick"] != 15)), "list", "apply_rule", cs)
    add("robust centering n=15 residual range (every weighting)",
        rng([v for c in C for v in c["n15_residual"].values()]), "bits [min,max]", "evaluate", cs)
    add("robust centering n=15 min closure (min over draws and weightings)",
        min(min(c["n15_closure"].values()) for c in C), "%", "evaluate", cs)
    add("robust centering n=24 probe residual range", rng([c["n24_probe_residual"] for c in C]),
        "bits [min,max]", "evaluate", cs)

    A = R["random_arm"]; asd = "seed_base 2042 (paper) and 51000-69000"
    for n in (16, 24, 32):
        add(f"robust random arm gap n={n} range", rng([a["rows"][n]["gap"] for a in A]), "bits [min,max]", "random_arm", asd)
        add(f"robust random arm gap n={n} paper draw is the smallest",
            A[0]["rows"][n]["gap"] == min(a["rows"][n]["gap"] for a in A), "bool", "random_arm", asd)
    add("robust random arm gap n=24 median", float(np.median([a["rows"][24]["gap"] for a in A])), "bits", "random_arm", asd)
    N = R["noise"]
    add("robust noise control ratio range", rng([x["ratio"] for x in N]), "x [min,max]", "noise_control",
        "mu/noise seeds (11,500) and 7 others")
    L = R["lookalike"]; ls = "split seeds 3042 (paper) and 4042-22042"
    add("robust lookalike realistic mean range", rng([100 * x["mean"] for x in L]), "% [min,max]", "lookalike_splits", ls)
    add("robust lookalike ratio range", rng([x["ratio_to_far_sampled"] for x in L]), "x [min,max]", "lookalike_splits", ls)
    add("robust lookalike median always zero", all(x["median"] == 0 for x in L), "bool", "lookalike_splits", ls)
    add("robust lookalike victims >10% range", rng([x["victims_gt_10pct"] for x in L]), "victims [min,max]", "lookalike_splits", ls)
    add("robust lookalike victims >50% range", rng([x["victims_gt_50pct"] for x in L]), "victims [min,max]", "lookalike_splits", ls)
    add("robust lookalike null mean range", rng([100 * x["oracle_null_mean"] for x in L]), "% [min,max]", "lookalike_splits", ls)
    return rows


def main():
    st = require_committed()
    T0 = time.time()
    emb, subs = load_corpus()
    W = get_W()
    sizes = np.array([len(emb[s]) for s in subs])
    mu = np.concatenate([emb[s] for s in subs]).mean(axis=0)
    cen = {s: emb[s] - mu for s in subs}
    sweep = json.load(open(PIPELINE_OUT / "step2_sweep.json", encoding="utf-8"))
    M = {r["quantity"]: r["value"] for r in csv.DictReader(open(PIPELINE_OUT / "manifest.csv", encoding="utf-8"))}
    R = {"git": st}
    R["bootstrap"] = bootstrap_seeds(emb, subs, W)
    R["projection"] = projection_seeds(emb, subs)
    R["enrollment"] = enrollment_draws(emb, subs, W)
    R["random_arm"] = random_arm(cen, subs, W, sweep)
    R["noise"] = noise_control(subs, sizes, W, sweep)
    R["lookalike"] = lookalike_splits(emb, cen, subs, sizes, W, sweep, float(M["FAR_sampled n=15"]))
    R["centering"] = centering_draws(emb, subs, W)
    R["wall_seconds"] = round(time.time() - T0, 1)
    json.dump(R, open(OUT / "robustness.json", "w", encoding="utf-8"), indent=1, default=float)
    write_manifest(OUT / "manifest.csv", manifest_rows(R, st))
    print("wrote", OUT / "robustness.json", "and", OUT / "manifest.csv", flush=True)


if __name__ == "__main__":
    main()
