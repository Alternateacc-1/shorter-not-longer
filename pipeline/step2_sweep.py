"""Step 2: the bit-string length sweep (Table 1, Table 2, Fig. 2).

Reads the embeddings from step 1 (their SHA-256 is checked against
step1_embed.json) and computes, for every length n: the least-parity BCH code
under probe, capped and subject weighting, N_eff and the residual, the full
impostor set at every radius, the Clopper-Pearson bound, the subject bootstrap,
the probe-count tertiles and per-subject decode failure. Then the selection
ablation (Arm A vs Arm B), the decorrelation rule and the noise-only control.
Where this script extends a library function (more weightings, every radius),
it also asserts that it agrees with that function on what the function returns.

Lengths: n = 8..24 and 32, 40, 48, 56, 64. The paper's seventeen are 8..24.

    python -m pipeline.step2_sweep
"""
import hashlib
import inspect
import json
import os
import time

from pipeline.provenance import EMB_NPZ, OUT_MAIN, Log, dump, require_committed_code, sha256_file

LENGTHS = list(range(8, 25)) + [32, 40, 48, 56, 64]
DECORR_TAUS = (0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05)


def main():
    st = require_committed_code()
    log = Log("step2_sweep")
    T0 = time.time()
    quick = bool(os.environ.get("QUICK_RUN"))          # n = 9, 15, 24 only: a fast smoke test
    lengths = [9, 15, 24] if quick else LENGTHS

    import multiprocessing as mp
    import numpy as np
    from scipy.stats import binom, beta as bdist

    env = json.load(open(OUT_MAIN / "step1_embed.json", encoding="utf-8"))
    assert sha256_file(EMB_NPZ) == env["embeddings_npz_sha256"], "embedding file is not s0's output"

    import fuzzycommit.common as common
    common.DATA_DIR = OUT_MAIN
    from fuzzycommit.common import (GLOBAL_SEED, DECODE_FAILURE_TARGET as TARGET, PROBE_CAP, N_BOOTSTRAP,
                            N_IMPOSTOR, load_and_center, get_W, build_subject_info,
                            build_subject_info_random, probe_hds, select_capacity, bch_parity,
                            bch_true_t, t_empirical, subject_list_hash, probe_set_hash)
    from fuzzycommit.bit_errors import compute_sweep_row
    from fuzzycommit.false_accept import (impostor_pair_stats, neff_from_stats, compute_neff, run_exhaustive_far,
                         compute_analytical_far)
    from fuzzycommit.bootstrap import run_bootstrap
    from fuzzycommit.weighting import run_capped_and_weighted, run_tertiles
    from fuzzycommit.entropy import enrolled_bitstrings

    emb, cen = load_and_center()
    subs = sorted(cen)
    NS = len(subs)
    assert subject_list_hash(subs) == env["subject_list_sha256"]
    W = get_W()
    sizes = np.array([len(cen[s]) for s in subs])
    owner = np.repeat(np.arange(NS), sizes)
    allbits = np.concatenate([cen[s] for s in subs], axis=0) @ W > 0
    TB = enrolled_bitstrings(cen, W).astype(np.uint8)
    assert TB.shape == (NS, 512)
    ONES = np.ones(NS)
    R = {"git": st, "lengths": lengths, "quick": quick,
         "corpus": {"subjects": NS, "images": int(sizes.sum()), "probes": int((sizes - 5).sum()),
                    "subject_list_sha256": subject_list_hash(subs)}}
    log(f"corpus {NS} subjects, {sizes.sum()} images, {(sizes - 5).sum()} probes")

    # ------------------------------------------------------------------ shared projection
    W_sha = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()
    seen, calls = set(), [0]
    orig = common.enroll_select

    def spy(ev, Wx, n):
        calls[0] += 1
        seen.add(hashlib.sha256(np.ascontiguousarray(Wx).tobytes()).hexdigest())
        return orig(ev, Wx, n)
    common.enroll_select = spy
    try:
        build_subject_info(cen, W, 15, subs)
    finally:
        common.enroll_select = orig
    assert seen == {W_sha} and calls[0] == 2 * NS
    assert hashlib.sha256(np.ascontiguousarray(get_W()).tobytes()).hexdigest() == W_sha
    src_lines, start = inspect.getsourcelines(common.build_subject_info)
    quoted = [f"common.py:{start + i}: {l.rstrip()}" for i, l in enumerate(src_lines)
              if any(k in l for k in ("def build_subject_info", "for s in subjects",
                                      "enroll_select(", "probe_hds("))]
    gw_lines, gw_start = inspect.getsourcelines(common.get_W)
    R["shared_projection"] = {
        "W_sha256": W_sha, "seed": GLOBAL_SEED + 1, "shape": list(W.shape),
        "enroll_select_calls_observed": calls[0], "distinct_matrices_observed": len(seen),
        "get_W_source": [f"common.py:{gw_start + i}: {l.rstrip()}" for i, l in enumerate(gw_lines)],
        "build_subject_info_quote": quoted,
        "statement": "get_W() is called once per run; build_subject_info passes that one W to "
                     "enroll_select for every enrollee; no seed or matrix is per-subject. "
                     "Observed at run time: 622 enroll_select calls (311 + 311 re-derivation "
                     "checks) all received the matrix with the SHA-256 above."}
    log("shared projection proven:", W_sha[:16], calls[0], "calls, 1 matrix")

    # ------------------------------------------------------------------ sampled FAR in parallel
    os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from pipeline.sampled_far import worker
    pool = mp.get_context("spawn").Pool(processes=10)
    far_async = pool.map_async(worker, [(n, 43) for n in lengths])
    log("sampled-FAR workers started for", lengths)

    def least_parity(m, n, failfn):
        """select_capacity's rule for an arbitrary failure function."""
        for t in range(n + 1):
            ta = bch_true_t(m, t)
            if failfn(ta) < TARGET:
                return t, bch_parity(m, t), ta
        return n, bch_parity(m, n), bch_true_t(m, n)

    def mixture(n, par, neff):
        par = np.asarray(par); neff = np.asarray(neff)
        res = neff - par
        ok = (n - par > 0) & (res > 0)
        br = []
        for p in sorted(set(par.tolist())):
            sel = par == p
            br.append({"parity": int(p), "share_pct": 100 * float(sel.mean()),
                       "residual_mean": float(res[sel].mean()), "residual_sd": float(res[sel].std()),
                       "pct_positive_within": 100 * float(ok[sel].mean()),
                       "contribution_pct": 100 * float((ok & sel).mean())})
        clo = 100 * float(ok.mean())
        assert abs(sum(b["contribution_pct"] for b in br) - clo) < 1e-9
        return {"closure_pct": clo, "residual_median": float(np.median(res)),
                "residual_ci95": [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))],
                "branches": br}

    def frr_summary(v):
        v = np.asarray(v, float)
        return {"subjects": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)),
                "max": float(v.max()), "n_gt_5pct": int((v > 0.05).sum()),
                "n_gt_10pct": int((v > 0.10).sum()), "n_zero": int((v == 0).sum())}

    # ------------------------------------------------------------------ per-length loop
    L = {}
    for n in lengths:
        t1 = time.time()
        info, sels = build_subject_info(cen, W, n, subs)
        assert [x[0] for x in info] == subs
        row = compute_sweep_row(info, n)
        m = row["m"]
        hds = [x[3] for x in info]
        all_hd = row["all_hd"]
        d = {"n": n, "m": m, "beta": row["beta"], "std_beta": row["std_beta"],
             "dispersion": row["overdispersion"], "n_probes": row["n_probes"],
             "probe_set_sha256": probe_set_hash(info)}

        # --- three weightings: least parity meeting 1.5%, delivered capacity
        cap = [h[:PROBE_CAP] if len(h) > PROBE_CAP else h for h in hds]
        all_cap = np.concatenate(cap)
        PF = np.array([[float(np.mean(h > ta)) for ta in range(n + 1)] for h in hds])
        sw_fail = lambda ta: float(np.mean(np.array([float(np.mean(h > ta)) for h in hds])))
        sel_w = {"probe": (row["t_designed"], row["parity"], row["t"]),
                 "capped": select_capacity(all_cap, n),
                 "subject": least_parity(m, n, sw_fail)}
        beta_w = {"probe": row["beta"], "capped": float(np.mean(all_cap / n)),
                  "subject": float(np.mean([float(np.mean(h / n)) for h in hds]))}
        fail_w = {"probe": row["decode_fail_frac"],
                  "capped": float(np.mean(all_cap > sel_w["capped"][2])),
                  "subject": sw_fail(sel_w["subject"][2])}
        rw = run_capped_and_weighted(cen, W, n)                # library cross-check
        assert (rw["capped"]["parity"], rw["capped"]["t"]) == sel_w["capped"][1:]
        assert (rw["subject_weighted"]["parity"], rw["subject_weighted"]["t"]) == sel_w["subject"][1:]
        assert rw["capped"]["beta"] == round(beta_w["capped"], 4)
        assert rw["subject_weighted"]["beta"] == round(beta_w["subject"], 4)

        # --- entropy budget
        S1, S2, CNT = impostor_pair_stats(cen, W, n, info)
        ne = neff_from_stats(S1, S2, CNT, ONES)
        assert ne == float(compute_neff(cen, W, n, info)) == rw["entropy_bits"]
        d["neff"] = ne
        d["neff_over_n"] = ne / n
        d["weightings"] = {w: {"t_designed": int(sel_w[w][0]), "parity": int(sel_w[w][1]),
                               "t": int(sel_w[w][2]), "k": int(n - sel_w[w][1]),
                               "beta": beta_w[w], "decode_fail": fail_w[w],
                               "residual": ne - sel_w[w][1]} for w in sel_w}
        t_pw, p_pw = int(row["t"]), int(row["parity"])
        d["t_required_designed"] = int(t_empirical(all_hd, n))
        d["t_binomial"] = int(next(t for t in range(n + 1) if binom.sf(t, n, row["beta"]) < TARGET))
        d["decode_fail_by_t"] = {t: {"empirical": float(np.mean(all_hd > t)),
                                     "binomial": float(binom.sf(t, n, row["beta"]))}
                                 for t in range(min(n, 16) + 1)}

        # --- full impostor set: every ordered subject pair, every radius
        exh = run_exhaustive_far(cen, W, n, t_pw, subs)
        A = np.zeros((NS, NS, n + 1), np.int32)
        for i, (_, sel, tmpl, _) in enumerate(info):
            h = (allbits[:, sel] != tmpl.astype(bool)).sum(axis=1)
            cnt = np.bincount(owner * (n + 1) + h, minlength=NS * (n + 1)).reshape(NS, n + 1)
            A[i] = np.cumsum(cnt, axis=1)
        eye = np.eye(NS, dtype=bool)
        A[eye] = 0
        off = ~eye
        comps = int(np.broadcast_to(sizes, (NS, NS))[off].sum())
        far_pair = [float((A[:, :, t] / sizes[None, :])[off].mean()) for t in range(n + 1)]
        far_comp = [float(A[:, :, t][off].sum() / comps) for t in range(n + 1)]
        assert comps == exh["n_comparisons"]
        assert far_comp[t_pw] == exh["far_comparison_uniform"]
        assert abs(far_pair[t_pw] - exh["far_pair_uniform"]) < 1e-12
        assert abs(round(ne, 4) - exh["n_eff"]) <= 1.01e-4
        d["exhaustive"] = {"comparisons": comps, "ordered_pairs": NS * (NS - 1),
                           "far_pair_uniform_by_t": far_pair, "far_comparison_uniform_by_t": far_comp,
                           "accepts_by_t": [int(A[:, :, t][off].sum()) for t in range(n + 1)],
                           "repo_far_pair_uniform": exh["far_pair_uniform"],
                           "repo_far_comparison_uniform": exh["far_comparison_uniform"],
                           "impostor_mean": exh["p_bar"], "repo_n_eff_4dp": exh["n_eff"]}
        d["far_analytical_by_t"] = [compute_analytical_far(n, t) for t in range(n + 1)]

        # --- distribution-free most-common-value bound (Clopper-Pearson, one-sided 99%)
        ks = np.array([np.unique(TB[:, sel], axis=0, return_counts=True)[1].max() for _, sel in sels])
        fl = -np.log2(bdist.ppf(0.99, ks + 1, NS - ks))
        flb = -np.log2(bdist.ppf(1 - 0.01 / NS, ks + 1, NS - ks))
        d["mcv"] = {"max_multiplicity_mean": float(ks.mean()), "max_multiplicity_max": int(ks.max()),
                    "floor_mean": float(fl.mean()), "floor_min": float(fl.min()),
                    "floor_bonf_mean": float(flb.mean()), "floor_bonf_min": float(flb.min()),
                    "bonferroni_alpha": 0.01 / NS}
        for w in sel_w:
            p = sel_w[w][1]
            d["weightings"][w]["bound"] = {
                "mean": float(fl.mean() - p), "min": float(fl.min() - p),
                "min_bonferroni": float(flb.min() - p), "mean_bonferroni": float(flb.mean() - p),
                "index_sets_positive": int((fl > p).sum()),
                "index_sets_positive_bonferroni": int((flb > p).sum())}

        # --- subject-level cluster bootstrap, N_eff recomputed per replicate
        pct_repo, boot_repo = run_bootstrap(info, n, ne, cen, W)
        rng = np.random.RandomState(GLOBAL_SEED + 500 + n)
        par = {"probe": [], "capped": [], "subject": []}
        nes = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(NS, NS, replace=True)
            par["probe"].append(select_capacity(np.concatenate([hds[i] for i in idx]), n)[1])
            par["capped"].append(select_capacity(np.concatenate([cap[i] for i in idx]), n)[1])
            par["subject"].append(least_parity(m, n, lambda ta: float(np.mean(PF[idx, ta])))[1])
            nes.append(neff_from_stats(S1, S2, CNT, np.bincount(idx, minlength=NS)))
        nes = np.array(nes)
        assert np.array_equal(nes - np.array(par["probe"]), boot_repo["residuals"])
        d["bootstrap"] = {"seed": GLOBAL_SEED + 500 + n, "replicates": N_BOOTSTRAP,
                          "neff_sd": float(nes.std()), "neff_mean": float(nes.mean()),
                          "repo_closure_pct": pct_repo}
        for w in par:
            d["bootstrap"][w] = mixture(n, par[w], nes)
        assert d["bootstrap"]["probe"]["closure_pct"] == pct_repo

        # --- tertiles (subjects sorted by image count, cut at NS//3 and 2NS//3)
        ter_repo = run_tertiles(cen, emb, W, n)
        cnts = sorted([(x[0], len(emb[x[0]])) for x in info], key=lambda x: x[1])
        cuts = [NS // 3, 2 * NS // 3, NS]
        groups = [cnts[:cuts[0]], cnts[cuts[0]:cuts[1]], cnts[cuts[1]:]]
        d["tertiles"] = []
        for g, rep in zip(groups, ter_repo):
            names = {x[0] for x in g}
            ids = [i for i, s in enumerate(subs) if s in names]
            c = np.zeros(NS); c[ids] = 1
            th = [hds[i] for i in ids]
            tcap = [cap[i] for i in ids]
            ne_t = neff_from_stats(S1, S2, CNT, c)
            selt = {"probe": select_capacity(np.concatenate(th), n),
                    "capped": select_capacity(np.concatenate(tcap), n),
                    "subject": least_parity(m, n, lambda ta: float(np.mean(PF[ids, ta])))}
            assert (rep["parity"], rep["t"]) == selt["probe"][1:]
            assert rep["neff"] == round(ne_t, 3) and rep["n_subjects"] == len(ids)
            d["tertiles"].append({
                "label": rep["label"], "subjects": len(ids), "probes": int(sum(len(h) for h in th)),
                "images_min": int(g[0][1]), "images_max": int(g[-1][1]), "neff": ne_t,
                "beta": {"probe": float(np.mean(np.concatenate(th) / n)),
                         "subject": float(np.mean([np.mean(h / n) for h in th]))},
                "weightings": {w: {"parity": int(selt[w][1]), "t": int(selt[w][2]),
                                   "residual": ne_t - selt[w][1]} for w in selt},
                "frr_at_global_t": frr_summary(PF[ids, t_pw])})

        # --- per-subject decode failure at the delivered capacity
        d["per_subject_frr"] = {"t": t_pw, "summary": frr_summary(PF[:, t_pw]),
                                "values": PF[:, t_pw].tolist()}
        L[n] = d
        log(f"n={n:>2} m={m} beta={row['beta']:.4f} t={t_pw} p={p_pw} N_eff={ne:.4f} "
            f"res pw/cap/sw {ne - p_pw:+.3f}/{ne - sel_w['capped'][1]:+.3f}/{ne - sel_w['subject'][1]:+.3f} "
            f"closure {pct_repo:.1f} floor {fl.mean():.3f} ({time.time() - t1:.1f}s)")
    R["lengths_detail"] = L

    # ------------------------------------------------------------------ ablation (Table 2)
    AB = {}
    for n in ([16, 24, 32] if not quick else [24]):
        for arm in ("A", "B"):
            if arm == "A":
                info, _ = build_subject_info(cen, W, n, subs)
            else:
                info, _ = build_subject_info_random(cen, W, n, subs, seed_base=GLOBAL_SEED + 2000)
            row = compute_sweep_row(info, n)
            ne = float(compute_neff(cen, W, n, info))
            AB[f"{arm}{n}"] = {"beta": row["beta"], "dispersion": row["overdispersion"],
                               "t": int(row["t"]), "t_designed": int(row["t_designed"]),
                               "parity": int(row["parity"]), "k": int(row["k"]), "neff": ne,
                               "residual": ne - row["parity"],
                               "t_required_designed": int(t_empirical(row["all_hd"], n)),
                               "t_binomial": int(next(t for t in range(n + 1)
                                                      if binom.sf(t, n, row["beta"]) < TARGET))}
    R["ablation"] = AB
    log("ablation", {k: (round(v["beta"], 4), v["parity"], round(v["residual"], 2)) for k, v in AB.items()})

    # ------------------------------------------------------------------ decorrelation sweep
    if not quick:
        corr = np.abs(np.nan_to_num(np.corrcoef(allbits.astype(float), rowvar=False)))
        np.fill_diagonal(corr, 0.0)

        def decorr(n, tau):
            info = []
            for s in subs:
                ev = cen[s][:5]; proj = ev @ W
                cons = np.abs(np.mean((proj > 0).astype(float), axis=0) - 0.5) * 2.0
                order = np.lexsort((-np.mean(np.abs(proj), axis=0), -cons))
                sel = []
                for j in order:
                    if len(sel) == n: break
                    if not sel or corr[j, sel].max() < tau: sel.append(int(j))
                for j in order:
                    if len(sel) == n: break
                    if int(j) not in sel: sel.append(int(j))
                sel = np.array(sel)
                tmpl = ((np.mean(ev, axis=0) @ W)[sel] > 0).astype(int)
                info.append((s, sel, tmpl, probe_hds(cen[s][5:], W, sel, tmpl)))
            hd = np.concatenate([x[3] for x in info])
            _, p, _ = select_capacity(hd, n)
            ne = float(compute_neff(cen, W, n, info))
            return {"tau": tau, "beta": float(np.mean(hd / n)), "parity": int(p), "neff": ne,
                    "residual": ne - p}
        R["decorrelation"] = {}
        for n in (15, 24):
            base = L[n]["weightings"]["probe"]["residual"]
            rows = [decorr(n, tau) for tau in DECORR_TAUS]
            best = max(rows, key=lambda r: r["residual"])
            R["decorrelation"][n] = {"baseline_residual": base, "rows": rows,
                                     "best_tau": best["tau"], "gain": best["residual"] - base}
            log(f"decorrelation n={n}: gain {best['residual'] - base:+.3f} at tau={best['tau']}")

        # -------------------------------------------------------------- noise-only control
        rngn = np.random.RandomState(11)
        mus = rngn.randn(NS, 512)
        noise = [np.random.RandomState(500 + i).randn(sizes[i], 512) for i in range(NS)]

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
            lo, hi = (mid, hi) if b < L[24]["beta"] else (lo, mid)
        bn, fn = synth((lo + hi) / 2)
        R["noise_control"] = {"sigma": (lo + hi) / 2, "beta": bn, "far_pair": fn,
                              "ratio": fn / compute_analytical_far(24, 3),
                              "lfw_ratio_pair": L[24]["exhaustive"]["far_pair_uniform_by_t"][3]
                              / compute_analytical_far(24, 3)}
        log("noise control", R["noise_control"])

    # ------------------------------------------------------------------ collect sampled FAR
    far_rows = far_async.get()
    pool.close(); pool.join()
    for fr in far_rows:
        n = fr["n"]
        assert fr["t"] == L[n]["weightings"]["probe"]["t"]
        L[n]["sampled"] = fr
        L[n]["sampled"]["far_ratio"] = fr["accepts"] / N_IMPOSTOR / compute_analytical_far(n, fr["t"])
        log(f"sampled FAR n={n}: {fr['accepts']} accepts at t={fr['t']} (seed {fr['seed']}), "
            f"ratio {L[n]['sampled']['far_ratio']:.2f}")
    R["wall_seconds"] = round(time.time() - T0, 1)
    dump(R, "step2_sweep.json")
    log("done", R["wall_seconds"], "s")


if __name__ == "__main__":
    main()
