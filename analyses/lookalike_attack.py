"""Lookalike (non-zero-effort) attack at n = 9 and n = 15, from the step 1 embeddings.

Threat: knowing the victim v, the attacker picks one of the 310 other identities u and presents u's
images. An image is accepted against v's commitment iff its bit-string at v's index set lies within
the delivered t of v's enrolled bit-string (the full-set FAR's acceptance event). Every image of u
is a presentation (all of u's images, as in the full-set FAR, so the totals reconcile with it).

  a  a(v,u) = fraction of u's images accepted against v (311 x 310 rates)
  b  lookalike rate L(v) = max_u a(v,u); its distribution over the 311 victims
  c  (i) oracle choice (best u in hindsight, upper bound), all images and half-B images;
     (ii) realistic choice: u* = argmax cosine between v's mean embedding (v's probe images, never
     the enrolment images) and u's mean embedding over a random half A of u's images (seed 3042);
     the rate a(v,u*) is measured on u's other half B
  d  mean lookalike rate / zero-effort FAR (sampled, and full-set pair-uniform)
  e  per-victim zero-effort FAR: pair-uniform mean_u a(v,u) and comparison-uniform, over 311 victims
  null: the oracle under a binomial null (no lookalike structure, same image counts, seed 3043)
Delivered t: probe, capped and subject weighting deliver the same t at both lengths (asserted),
so one evaluation covers all three. Cross-checks: totals == FAR_full_accepts, mean == FAR_full_pair
(results/pipeline/manifest.csv); the whole accept matrix is recomputed with analyses/crosscheck
and compared exactly.

    python -m analyses.lookalike_attack
Writes results/codes_lookalike/lookalike_attack.json and part_lookalike.csv."""
import csv
import json
import time

from analyses.helpers import PIPELINE_OUT as OUT_MAIN, out_dir, require_committed, write_manifest

OUT = out_dir("codes_lookalike")

LENGTHS = (9, 15)
SPLIT_SEED = 3042                       # GLOBAL_SEED + 3000: the half-A / half-B split of every identity


def stats(x):
    import numpy as np
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()), "min": float(x.min()),
            "n_gt_10pct": int((x > 0.10).sum()), "n_gt_50pct": int((x > 0.50).sum()),
            "n_eq_100pct": int((x == 1.0).sum()), "n_zero": int((x == 0).sum()), "victims": int(len(x))}


def main():
    st = require_committed()
    T0 = time.time()
    import numpy as np
    from pipeline.provenance import EMB_NPZ, sha256_file
    from analyses import crosscheck as V
    env = json.load(open(OUT_MAIN / "step1_embed.json", encoding="utf-8"))
    assert sha256_file(EMB_NPZ) == env["embeddings_npz_sha256"]
    LD = json.load(open(OUT_MAIN / "step2_sweep.json", encoding="utf-8"))["lengths_detail"]
    M = {r["quantity"]: r["value"] for r in csv.DictReader(open(OUT_MAIN / "manifest.csv", encoding="utf-8"))}
    import fuzzycommit.common as common
    common.DATA_DIR = OUT_MAIN
    from fuzzycommit.common import load_and_center, get_W, build_subject_info
    emb, cen = load_and_center()
    subs = sorted(cen); NS = len(subs)
    W = get_W()
    sizes = np.array([len(cen[s]) for s in subs])
    owner = np.repeat(np.arange(NS), sizes)
    allbits = np.concatenate([cen[s] for s in subs], axis=0) @ W > 0
    off = ~np.eye(NS, dtype=bool)

    # the attacker's own face matcher: raw ArcFace embeddings, L2-normalised, averaged, normalised
    def unit(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    rng = np.random.RandomState(SPLIT_SEED)
    halfA, halfB = [], []
    for s in subs:
        perm = rng.permutation(len(emb[s])); h = len(perm) // 2
        halfA.append(np.sort(perm[:h])); halfB.append(np.sort(perm[h:]))
    vknow = unit(np.array([unit(emb[s][5:]).mean(axis=0) for s in subs]))          # v's probe images only
    uA = unit(np.array([unit(emb[s][halfA[i]]).mean(axis=0) for i, s in enumerate(subs)]))
    cosm = vknow @ uA.T
    np.fill_diagonal(cosm, -np.inf)
    ustar = cosm.argmax(axis=1)
    inB = np.zeros(len(owner), bool)
    st0 = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    for i in range(NS):
        inB[st0[i] + halfB[i]] = True
    sizesB = np.array([len(b) for b in halfB])

    R = {"git": st, "split_seed": SPLIT_SEED, "lengths": {}}
    emb_v, cen_v = V.load(); Wv, _ = V.W_of(43); SV = V.Setup(cen_v, Wv)
    rows = []

    def add(q, v, u, f):
        rows.append({"quantity": q, "value": repr(float(v)) if isinstance(v, float) else v, "units": u,
                     "script": "analyses/lookalike_attack.py", "function": f,
                     "seeds": f"projection 43; split {SPLIT_SEED}", "commit": st["commit"][:12]})
    for n in LENGTHS:
        ts = {w: LD[str(n)]["weightings"][w]["t"] for w in ("probe", "capped", "subject")}
        assert len(set(ts.values())) == 1, ts
        t = ts["probe"]
        info, _ = build_subject_info(cen, W, n, subs)
        acc = np.zeros((NS, NS)); accB = np.zeros((NS, NS))
        for i, (_, sel, tmpl, _) in enumerate(info):
            ok = (allbits[:, sel] != tmpl.astype(bool)).sum(axis=1) <= t
            acc[i] = np.bincount(owner, ok.astype(float), NS)
            accB[i] = np.bincount(owner, (ok & inB).astype(float), NS)
        acc[~off] = 0; accB[~off] = 0
        a = acc / sizes[None, :]; aB = accB / sizesB[None, :]
        # cross-checks against step 2: totals and the pair-uniform mean
        tot = int(acc[off].sum())
        assert tot == int(M[f"FAR_full_accepts n={n}"]) == LD[str(n)]["exhaustive"]["accepts_by_t"][t]
        assert abs(a[off].mean() - float(M[f"FAR_full_pair n={n}"])) < 1e-15
        # second implementation: the whole accept matrix from crosscheck's impostor distances
        L = V.Length(SV, n)
        accv = np.array([np.bincount(SV.owner, (L.IH[i] <= t).astype(float), SV.NS) for i in range(SV.NS)])
        accv[~off] = 0
        vl_same = bool(np.array_equal(accv, acc))
        pair_v, comp_v, acc_v, _ = L.far_full(t)
        assert vl_same and acc_v == tot and abs(pair_v - a[off].mean()) < 1e-15
        am = np.where(off, a, -1.0); aBm = np.where(off, aB, -1.0)
        orc = am.max(axis=1); orc_u = am.argmax(axis=1)
        orcB = aBm.max(axis=1)
        real = aB[np.arange(NS), ustar]
        far_s = float(M[f"FAR_sampled n={n}"]); far_p = float(M[f"FAR_full_pair n={n}"])
        pv_pair = a[off].reshape(NS, NS - 1).mean(axis=1)
        others = sizes.sum() - sizes
        pv_comp = acc.sum(axis=1) / others
        d = {"t": t, "t_by_weighting": ts, "accepts_total": tot, "far_full_pair": far_p, "far_sampled": far_s,
             "crosscheck_matrix_identical": vl_same,
             "a_vu_pairs": {"pairs": int(off.sum()), "zero": int((a[off] == 0).sum()), "max": float(a[off].max()),
                            "pairs_gt_10pct": int((a[off] > 0.10).sum()), "pairs_gt_50pct": int((a[off] > 0.50).sum())},
             "oracle_all_images": stats(orc), "oracle_half_B": stats(orcB), "realistic_half_B": stats(real),
             "realistic_picks_oracle_best_u": int((ustar == orc_u).sum()),
             "realistic_attacker_rank_of_chosen_u": None,
             "ratio_to_zero_effort": {"oracle_mean_over_far_sampled": float(orc.mean() / far_s),
                                      "oracle_mean_over_far_full_pair": float(orc.mean() / far_p),
                                      "oracle_half_B_mean_over_far_sampled": float(orcB.mean() / far_s),
                                      "realistic_mean_over_far_sampled": float(real.mean() / far_s),
                                      "realistic_mean_over_far_full_pair": float(real.mean() / far_p),
                                      "oracle_median_over_far_sampled": float(np.median(orc) / far_s),
                                      "realistic_median_over_far_sampled": float(np.median(real) / far_s)},
             "per_victim_far_pair_uniform": stats(pv_pair), "per_victim_far_comparison_uniform": stats(pv_comp),
             "per_victim_far_max_over_mean": float(pv_pair.max() / pv_pair.mean())}
        # null baseline for the oracle: no lookalike structure (every u accepted against v at v's own
        # zero-effort rate, comparison-uniform), same image counts; the max over 310 noisy proportions
        nrng = np.random.RandomState(SPLIT_SEED + 1); REPS = 200
        nul, nulB = np.zeros(NS), np.zeros(NS)
        for _ in range(REPS):
            X = nrng.binomial(np.broadcast_to(sizes, (NS, NS)), pv_comp[:, None]) / sizes[None, :]
            XB = nrng.binomial(np.broadcast_to(sizesB, (NS, NS)), pv_comp[:, None]) / sizesB[None, :]
            nul += np.where(off, X, -1).max(axis=1) / REPS; nulB += np.where(off, XB, -1).max(axis=1) / REPS
        d["oracle_null_binomial"] = {"reps": REPS, "seed": SPLIT_SEED + 1, "all_images_mean": float(nul.mean()),
                                     "half_B_mean": float(nulB.mean()),
                                     "oracle_excess_over_null_mean": float((orc - nul).mean()),
                                     "victims_oracle_above_null": int((orc > nul).sum())}
        ranks = [int((aBm[v] > aB[v, ustar[v]]).sum()) + 1 for v in range(NS)]
        d["realistic_attacker_rank_of_chosen_u"] = stats(np.array(ranks, float))
        d["top_victims_oracle"] = [{"victim": subs[v], "attacker": subs[orc_u[v]], "rate": float(orc[v]),
                                    "attacker_images": int(sizes[orc_u[v]])} for v in np.argsort(-orc)[:5]]
        R["lengths"][n] = d
        print(f"n={n} t={t}: accepts {tot} (= manifest); crosscheck matrix identical {vl_same}", flush=True)
        for k in ("oracle_all_images", "oracle_half_B", "realistic_half_B", "per_victim_far_pair_uniform"):
            s = d[k]
            print(f"  {k:<28} mean {s['mean']:.4f} median {s['median']:.4f} p90 {s['p90']:.4f} p99 {s['p99']:.4f} "
                  f"max {s['max']:.4f} >10% {s['n_gt_10pct']} >50% {s['n_gt_50pct']} =100% {s['n_eq_100pct']} zero {s['n_zero']}",
                  flush=True)
        print(f"  ratios {json.dumps({k: round(v, 2) for k, v in d['ratio_to_zero_effort'].items()})}; realistic picks the "
              f"oracle's u for {d['realistic_picks_oracle_best_u']} victims", flush=True)
        for k in ("oracle_all_images", "oracle_half_B", "realistic_half_B", "per_victim_far_pair_uniform",
                  "per_victim_far_comparison_uniform"):
            for f, v in d[k].items():
                add(f"w2 lookalike n={n} {k} {f}", v, "rate" if isinstance(v, float) else "victims",
                    "max_u a(v,u)" if k.startswith(("oracle", "realistic")) else "zero-effort FAR of victim v")
        for f, v in d["ratio_to_zero_effort"].items():
            add(f"w2 lookalike n={n} ratio {f}", v, "x", "mean (median) lookalike rate / zero-effort FAR")
        add(f"w2 lookalike n={n} t", t, "errors", "delivered t (probe = capped = subject)")
        add(f"w2 lookalike n={n} accepts total", tot, "of 1681750", "sum_v sum_u accepts (= FAR_full_accepts)")
        add(f"w2 lookalike n={n} realistic picks oracle u", d["realistic_picks_oracle_best_u"], "victims", "u* == argmax_u a(v,u)")
        add(f"w2 lookalike n={n} per-victim FAR max over mean", d["per_victim_far_max_over_mean"], "x", "pair-uniform")
        for f, v in d["oracle_null_binomial"].items():
            add(f"w2 lookalike n={n} oracle null {f}", v, "rate" if isinstance(v, float) else "-",
                "binomial null: no lookalike structure, same image counts (seed 3043)")
        for f, v in d["a_vu_pairs"].items():
            add(f"w2 lookalike n={n} a(v,u) {f}", v, "pairs" if isinstance(v, int) else "rate", "per-attacker-identity rate")
    R["wall_seconds"] = round(time.time() - T0, 1)
    json.dump(R, open(OUT / "lookalike_attack.json", "w", encoding="utf-8"), indent=1, default=V._jsonable)
    write_manifest(OUT / "part_lookalike.csv", rows)
    print(f"{len(rows)} manifest rows; {R['wall_seconds']}s")


if __name__ == "__main__":
    main()
