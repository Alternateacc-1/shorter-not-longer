"""Step 4: robustness of the operating point.

  a  Golay [23,12,7] at n=23; extended BCH at n=16, 24, 32 (and extended Golay at 24):
     residual, decode failure, closure with mixture, FAR, under every weighting
  b  distribution-free bound at n=8..11, 15: mean, minimum over the 311 index sets,
     minimum after Bonferroni (alpha = 0.01/311), probe and subject weighting
  c  per-subject FRR at n=9 and n=15, overall and by tertile
  d  decode-failure target 1.0/1.5/2.0/3.0% -> chosen n, t, parity, residual
  e  decode-failure and FAR ratios at n=9, 15, 24
  f  projection seeds {43,101,202,303,404} at n=9, 11, 15, in-sample and held-out
     (400 subject splits; split seeds labelled separately), plus the held-out
     length-selection criteria used by the decision rule

Reads step2_sweep.json (same run) for the sampled-FAR histograms and the
per-length values; everything else is recomputed here from the embeddings.

    python -m pipeline.step4_robustness
"""
import json
import math
import time

from pipeline.provenance import EMB_NPZ, OUT_MAIN, Log, dump, require_committed_code, sha256_file

PROJ_SEEDS = (43, 101, 202, 303, 404)
SPLIT_SEEDS = {"paper": 20260903, "review": 20261001}
N_SPLITS = 400
TARGETS = (0.010, 0.015, 0.020, 0.030)
MEASURED = [9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 32, 40, 48, 56, 64]


def main():
    st = require_committed_code()
    log = Log("step4_robustness")
    T0 = time.time()
    import numpy as np
    from scipy.stats import beta as bdist

    env = json.load(open(OUT_MAIN / "step1_embed.json", encoding="utf-8"))
    assert sha256_file(EMB_NPZ) == env["embeddings_npz_sha256"]
    S1R = json.load(open(OUT_MAIN / "step2_sweep.json", encoding="utf-8"))   # the committed run
    LD = S1R["lengths_detail"]

    import fuzzycommit.common as common
    common.DATA_DIR = OUT_MAIN
    from fuzzycommit.common import (GLOBAL_SEED, DECODE_FAILURE_TARGET as TARGET, PROBE_CAP, N_BOOTSTRAP,
                            N_IMPOSTOR, load_and_center, get_W, build_subject_info, bch_parity,
                            bch_true_t, compute_bch_m)
    from fuzzycommit.false_accept import impostor_pair_stats, neff_from_stats, compute_analytical_far
    from fuzzycommit.bootstrap import run_bootstrap_seed_sensitivity
    from fuzzycommit.entropy import enrolled_bitstrings
    from pipeline.bch_codes import alternative_codes, bch_generator

    emb, cen = load_and_center()
    subs = sorted(cen); NS = len(subs)
    sizes = np.array([len(cen[s]) for s in subs]); NPR = (sizes - 5).astype(float)
    NCAP = np.minimum(sizes - 5, PROBE_CAP).astype(float)
    owner = np.repeat(np.arange(NS), sizes)
    ALLV = np.concatenate([cen[s] for s in subs], axis=0)
    ONES = np.ones(NS)
    R = {"git": st}

    class Len:
        """Per-(projection seed, n) precomputation using the library functions."""
        def __init__(self, n, proj_seed=43, with_far=False):
            self.n = n; self.m = compute_bch_m(n); self.seed = proj_seed
            W = get_W(proj_seed)
            info, sels = build_subject_info(cen, W, n, subs)
            self.hds = [x[3] for x in info]
            self.H = np.array([np.bincount(h, minlength=n + 1) for h in self.hds], float)
            self.Hc = np.array([np.bincount(h[:PROBE_CAP], minlength=n + 1) for h in self.hds], float)
            # TAIL[i, t] = #probes of subject i with HD > t
            self.TAIL = np.concatenate([np.flip(np.flip(self.H, 1).cumsum(1), 1)[:, 1:],
                                        np.zeros((NS, 1))], 1)
            self.TAILC = np.concatenate([np.flip(np.flip(self.Hc, 1).cumsum(1), 1)[:, 1:],
                                         np.zeros((NS, 1))], 1)
            self.PF = self.TAIL / NPR[:, None]
            self.S1, self.S2, self.CNT = impostor_pair_stats(cen, W, n, info)
            # BCH menu: (parity, delivered t) for unique parities, least first
            seen = {}
            for t in range(n + 1):
                p = bch_parity(self.m, t)
                if p not in seen:
                    seen[p] = bch_true_t(self.m, t)
            self.menu = sorted(seen.items())
            TB = enrolled_bitstrings(cen, W).astype(np.int64)
            if n <= 62:
                pw = np.int64(1) << np.arange(n, dtype=np.int64)
                self.words = np.array([TB[:, sel] @ pw for _, sel in sels])   # [index set, subject]
            if with_far:
                bits = ALLV @ W > 0
                A = np.zeros((NS, NS, n + 1), np.int32)
                for i, (_, sel, tmpl, _) in enumerate(info):
                    h = (bits[:, sel] != tmpl.astype(bool)).sum(axis=1)
                    A[i] = np.cumsum(np.bincount(owner * (n + 1) + h, minlength=NS * (n + 1))
                                     .reshape(NS, n + 1), axis=1)
                A[np.arange(NS), np.arange(NS)] = 0
                self.A = A

        def fails(self, c, w):
            """failure fraction at every radius t = 0..n under weighting w, multiplicity c."""
            if w == "probe":
                return (c @ self.TAIL) / (c @ NPR)
            if w == "capped":
                return (c @ self.TAILC) / (c @ NCAP)
            return (c @ self.PF) / c.sum()

        def choose(self, c, w, target=TARGET, menu=None):
            f = self.fails(c, w)
            for p, td in (menu or self.menu):
                if f[td] < target:
                    return p, td
            mm = menu or self.menu
            return mm[-1]

        def neff(self, c):
            return neff_from_stats(self.S1, self.S2, self.CNT, c)

        def floors(self, idx_sets, subjects, conf):
            """-log2 CP upper bound on the most common value, per index set, using the
            enrolled syndromes of `subjects` only."""
            N = len(subjects); out = []
            for i in idx_sets:
                _, cnt = np.unique(self.words[i, subjects], return_counts=True)
                k = int(cnt.max())
                out.append(-math.log2(float(bdist.ppf(conf, k + 1, N - k))))
            return np.array(out)

        def far_pair(self, t, c=None):
            rate = self.A[:, :, t] / sizes[None, :]
            mask = ~np.eye(NS, dtype=bool)
            if c is not None:
                on = c > 0; mask &= on[:, None] & on[None, :]
            return float(rate[mask].mean())

    WGT = ("probe", "capped", "subject")

    # ======================================================================= (a) code families
    def clean(menu):
        best, out = -1, []
        for p, td in sorted(menu):
            if td > best:
                out.append((p, td)); best = td
        return out

    A_ = {}
    for n in (16, 23, 24, 32):
        Lo = Len(n)
        # sanity: these generator polynomials reproduce the library's exact parity
        for t in range(1, 6):
            assert bch_generator(Lo.m, t).bit_length() - 1 == bch_parity(Lo.m, t)
        alts = alternative_codes(n)
        fams = {"shortened BCH (paper)": Lo.menu}
        for fam in sorted({a["family"] for a in alts}):
            fams[f"shortened BCH + {fam}"] = clean(Lo.menu + [(a["parity"], a["t"]) for a in alts
                                                              if a["family"] == fam and a["k"] > 0])
        if n == 24:
            fams["shortened BCH + all alternatives"] = clean(Lo.menu + [(a["parity"], a["t"]) for a in alts
                                                                        if a["k"] > 0])
        ne = Lo.neff(ONES)
        hist = np.array(LD[str(n)]["sampled"]["hd_hist"])
        exh = LD[str(n)]["exhaustive"]
        rng = np.random.RandomState(GLOBAL_SEED + 500 + n)      # same replicate draws as Table 1
        draws = [np.bincount(rng.choice(NS, NS, replace=True), minlength=NS).astype(float)
                 for _ in range(N_BOOTSTRAP)]
        nes = np.array([Lo.neff(c) for c in draws])
        out = {"codes": alts, "neff": ne, "families": {}}
        assert abs(ne - LD[str(n)]["neff"]) < 1e-12
        for fam, menu in fams.items():
            rows = {}
            for w in WGT:
                p, td = Lo.choose(ONES, w, menu=menu)
                bp = np.array([Lo.choose(c, w, menu=menu)[0] for c in draws], float)
                res = nes - bp; ok = (n - bp > 0) & (res > 0)
                br = [{"parity": int(q), "share_pct": 100 * float((bp == q).mean()),
                       "pct_positive_within": 100 * float(ok[bp == q].mean()),
                       "residual_mean": float(res[bp == q].mean())} for q in sorted(set(bp))]
                rows[w] = {"parity": int(p), "t": int(td), "k": int(n - p), "residual": ne - p,
                           "decode_fail": float(Lo.fails(ONES, w)[td]),
                           "closure_pct": 100 * float(ok.mean()), "mixture": br,
                           "far_sampled": float(hist[: td + 1].sum() / N_IMPOSTOR),
                           "far_full_pair": exh["far_pair_uniform_by_t"][td],
                           "far_full_comparison": exh["far_comparison_uniform_by_t"][td],
                           "far_independent": compute_analytical_far(n, td)}
            out["families"][fam] = rows
        for w in WGT:                                      # same replicates as step 2
            assert out["families"]["shortened BCH (paper)"][w]["closure_pct"] == \
                LD[str(n)]["bootstrap"][w]["closure_pct"], (n, w)
        A_[n] = out
        for fam, rows in out["families"].items():
            log(f"(a) n={n} {fam:<40} " + " | ".join(
                f"{w[:3]} p={r['parity']} t={r['t']} res={r['residual']:+.2f} clo={r['closure_pct']:.1f}"
                for w, r in rows.items()))
    R["a_codes"] = A_
    sw15 = LD["15"]["weightings"]["subject"]["residual"]
    alt_sw = []
    for n, o in A_.items():
        for fam, rows in o["families"].items():
            if fam == "shortened BCH (paper)":
                continue
            r = rows["subject"]
            if r["k"] > 0 and r["residual"] > 0 and r["residual"] > sw15:
                alt_sw.append((n, fam, r["residual"]))
    R["a_decision"] = {"n15_subject_residual": sw15, "alternatives_beating_n15_under_subject": alt_sw}

    # ======================================================================= (b) bound
    B_ = {}
    for n in (8, 9, 10, 11, 15):
        d = LD[str(n)]
        B_[n] = {"floor_mean": d["mcv"]["floor_mean"], "floor_min": d["mcv"]["floor_min"],
                 "floor_bonf_min": d["mcv"]["floor_bonf_min"], "floor_bonf_mean": d["mcv"]["floor_bonf_mean"],
                 **{w: {"parity": d["weightings"][w]["parity"], **d["weightings"][w]["bound"]}
                    for w in ("probe", "subject", "capped")}}
        log(f"(b) n={n}: floor mean {B_[n]['floor_mean']:.3f} min {B_[n]['floor_min']:.3f} "
            f"bonf-min {B_[n]['floor_bonf_min']:.3f} | pw p={B_[n]['probe']['parity']} "
            f"bound mean/min/bonf {B_[n]['probe']['mean']:+.3f}/{B_[n]['probe']['min']:+.3f}/"
            f"{B_[n]['probe']['min_bonferroni']:+.3f} | sw p={B_[n]['subject']['parity']} "
            f"{B_[n]['subject']['mean']:+.3f}/{B_[n]['subject']['min']:+.3f}/{B_[n]['subject']['min_bonferroni']:+.3f}")
    R["b_bound"] = B_

    # ======================================================================= (c) per-subject FRR
    R["c_frr"] = {n: {"overall": LD[str(n)]["per_subject_frr"]["summary"],
                      "t": LD[str(n)]["per_subject_frr"]["t"],
                      "tertiles": [{"label": t["label"], "subjects": t["subjects"], "probes": t["probes"],
                                    **t["frr_at_global_t"]} for t in LD[str(n)]["tertiles"]]}
                  for n in (9, 15)}
    for n in (9, 15):
        log(f"(c) n={n}", R["c_frr"][n]["overall"])

    # ======================================================================= (d) throughput sensitivity
    D_ = {}
    LS = {n: Len(n) for n in sorted(set(range(8, 25)) | set(MEASURED))}
    for n, Lo in LS.items():                               # vectorised selection == step 2
        assert abs(Lo.neff(ONES) - LD[str(n)]["neff"]) < 1e-12
        for w in WGT:
            assert Lo.choose(ONES, w) == (LD[str(n)]["weightings"][w]["parity"],
                                          LD[str(n)]["weightings"][w]["t"]), (n, w)
    for target in TARGETS:
        rows = {}
        for n, Lo in LS.items():
            ne = Lo.neff(ONES)
            r = {}
            for w in WGT:
                p, td = Lo.choose(ONES, w, target=target)
                r[w] = {"parity": int(p), "t": int(td), "residual": ne - p,
                        "decode_fail": float(Lo.fails(ONES, w)[td])}
            rows[n] = r
        sw_pos = [n for n in MEASURED if rows[n]["subject"]["residual"] > 0 and n - rows[n]["subject"]["parity"] > 0]
        all_pos = [n for n in MEASURED if all(rows[n][w]["residual"] > 0 for w in WGT)]
        contig = 0
        for n in MEASURED:
            if n in sw_pos: contig = n
            else: break
        chosen = max(sw_pos) if sw_pos else None
        best = max(MEASURED, key=lambda n: rows[n]["subject"]["residual"])
        D_[target] = {"rows": rows, "subject_positive_measured": sw_pos, "all_weightings_positive_measured": all_pos,
                      "chosen_n_longest_subject_positive": chosen,
                      "longest_contiguous_from_shortest": contig,
                      "max_subject_residual_n": best,
                      "chosen": {"n": chosen, **rows[chosen]["subject"]} if chosen else None,
                      "n15": rows[15]["subject"],
                      "throughput_ppm_at_chosen_probe_fail": 63 * (1 - rows[chosen]["probe"]["decode_fail"]) if chosen else None,
                      "throughput_floor_ppm": 63 * (1 - target)}
        log(f"(d) target {100 * target:.1f}%: sw-positive {sw_pos}; chosen n={chosen} "
            f"t={rows[chosen]['subject']['t']} p={rows[chosen]['subject']['parity']} "
            f"res={rows[chosen]['subject']['residual']:+.3f}; max-res n={best}; n=15 {rows[15]['subject']}")
    R["d_throughput"] = D_

    # ======================================================================= (e) ratios
    E_ = {}
    for n in (9, 15, 24):
        d = LD[str(n)]; t = d["weightings"]["probe"]["t"]
        dt = d["decode_fail_by_t"][str(t)]
        fa = d["far_analytical_by_t"][t]
        E_[n] = {"t": t, "decode_fail_empirical": dt["empirical"], "decode_fail_binomial": dt["binomial"],
                 "decode_ratio": dt["empirical"] / dt["binomial"],
                 "far_sampled": d["sampled"]["accepts"] / N_IMPOSTOR, "far_independent": fa,
                 "far_ratio_sampled": d["sampled"]["accepts"] / N_IMPOSTOR / fa,
                 "far_ratio_full_pair": d["exhaustive"]["far_pair_uniform_by_t"][t] / fa,
                 "far_ratio_full_comparison": d["exhaustive"]["far_comparison_uniform_by_t"][t] / fa}
        log(f"(e) n={n}", {k: round(v, 4) if isinstance(v, float) else v for k, v in E_[n].items()})
    R["e_ratios"] = E_

    # ======================================================================= (f) seeds, held-out
    F_ = {"in_sample": {}, "held_out": {}, "split_seeds": SPLIT_SEEDS, "projection_seeds": PROJ_SEEDS}
    for ps in PROJ_SEEDS:
        for n in (9, 11, 15):
            Lo = Len(n, ps, with_far=True)
            ne = Lo.neff(ONES)
            fl = Lo.floors(range(NS), np.arange(NS), 0.99)
            flb = Lo.floors(range(NS), np.arange(NS), 1 - 0.01 / NS)
            if ps == 43:                                   # same quantities as step 2
                assert abs(fl.mean() - LD[str(n)]["mcv"]["floor_mean"]) < 1e-12
                assert abs(flb.min() - LD[str(n)]["mcv"]["floor_bonf_min"]) < 1e-12
            row = {"beta": float((Lo.H.sum(0) @ np.arange(n + 1)) / Lo.H.sum() / n), "neff": ne,
                   "floor_mean": float(fl.mean()), "floor_min": float(fl.min()),
                   "floor_bonf_min": float(flb.min())}
            for w in WGT:
                p, td = Lo.choose(ONES, w)
                row[w] = {"parity": int(p), "t": int(td), "residual": ne - p,
                          "bound_mean": float(fl.mean() - p), "bound_min": float(fl.min() - p),
                          "bound_min_bonferroni": float(flb.min() - p),
                          "decode_fail": float(Lo.fails(ONES, w)[td]),
                          "far_full_pair": Lo.far_pair(td)}
            # library seed-sensitivity bootstrap (seed = proj_seed*1000 + n), probe weighting
            W = get_W(ps)
            info, _ = build_subject_info(cen, W, n, subs)
            row["closure_pct_probe"], row["boot_seed"] = run_bootstrap_seed_sensitivity(
                info, n, ne, ps, centered=cen, W=W)
            F_["in_sample"][f"{ps}/{n}"] = row

            # ---- held-out: train on half the subjects, evaluate on the other half
            for tag, sseed in SPLIT_SEEDS.items():
                if tag != "paper" and ps != 43:
                    continue
                rng = np.random.RandomState(sseed)
                recs = []
                for _ in range(N_SPLITS):
                    perm = rng.permutation(NS)
                    tr, te = np.sort(perm[NS // 2:]), np.sort(perm[:NS // 2])
                    ctr = np.zeros(NS); ctr[tr] = 1; cte = np.zeros(NS); cte[te] = 1
                    ne_te = Lo.neff(cte)
                    fl_te = Lo.floors(te, te, 0.99)
                    # same bound on the TRAIN half: separates the half-sample (N~155) effect
                    # of the Clopper-Pearson limit from any held-out (selection) effect
                    fl_tr = Lo.floors(tr, tr, 0.99)
                    rec = {"neff_test": ne_te, "floor_test_mean": float(fl_te.mean()),
                           "floor_test_min": float(fl_te.min()), "floor_train_mean": float(fl_tr.mean())}
                    for w in WGT:
                        p, td = Lo.choose(ctr, w)
                        rec[w] = (p, td, ne_te - p, float(Lo.fails(cte, w)[td]),
                                  float(fl_te.mean() - p), float(fl_te.min() - p), float(fl_tr.mean() - p))
                    recs.append(rec)
                full = {w: row[w]["parity"] for w in WGT}
                summ = {"split_seed": sseed, "splits": N_SPLITS, "projection_seed": ps, "n": n,
                        "half_sample_floor_test_mean": float(np.mean([r["floor_test_mean"] for r in recs])),
                        "half_sample_floor_train_mean": float(np.mean([r["floor_train_mean"] for r in recs])),
                        "floor_cap_at_N155": -math.log2(float(bdist.ppf(0.99, 2, NS // 2 - 1)))}
                for w in WGT:
                    P = np.array([r[w][0] for r in recs]); RES = np.array([r[w][2] for r in recs])
                    FA = np.array([r[w][3] for r in recs]); BM = np.array([r[w][4] for r in recs])
                    BN = np.array([r[w][5] for r in recs]); BT = np.array([r[w][6] for r in recs])
                    summ[w] = {"train_bound_mean_mean": float(BT.mean()),
                               "train_bound_mean_positive_pct": 100 * float((BT > 0).mean()),"share_train_choice_eq_full_sample_pct": 100 * float((P == full[w]).mean()),
                               "parity_freq": {int(q): 100 * float((P == q).mean()) for q in sorted(set(P))},
                               "test_residual_mean": float(RES.mean()), "test_residual_sd": float(RES.std(ddof=1)),
                               "test_residual_positive_pct": 100 * float((RES > 0).mean()),
                               "test_fail_mean": float(FA.mean()),
                               "test_fail_meets_target_pct": 100 * float((FA < TARGET).mean()),
                               "test_bound_mean_mean": float(BM.mean()),
                               "test_bound_mean_positive_pct": 100 * float((BM > 0).mean()),
                               "test_bound_min_mean": float(BN.mean()),
                               "test_bound_min_positive_pct": 100 * float((BN > 0).mean())}
                F_["held_out"][f"{ps}/{n}/{tag}"] = summ
            log(f"(f) seed {ps} n={n}: beta {row['beta']:.4f} pw p={row['probe']['parity']} "
                f"res {row['probe']['residual']:+.3f} sw p={row['subject']['parity']} "
                f"res {row['subject']['residual']:+.3f} bound pw/sw {row['probe']['bound_mean']:+.3f}/"
                f"{row['subject']['bound_mean']:+.3f} min {row['probe']['bound_min']:+.3f}; held-out pw "
                f"keep {F_['held_out'][f'{ps}/{n}/paper']['probe']['share_train_choice_eq_full_sample_pct']:.1f}% "
                f"test res {F_['held_out'][f'{ps}/{n}/paper']['probe']['test_residual_mean']:+.3f}")

    # ---- held-out LENGTH selection (seed 43), the criteria behind "the held-out choice"
    NSET = list(range(8, 25))
    LF = {n: Len(n, 43, with_far=True) for n in NSET}
    SEL = {}
    for tag, sseed in SPLIT_SEEDS.items():
        rng = np.random.RandomState(sseed); irng = np.random.RandomState(sseed + 1)
        picks = {k: [] for k in ("C1_max_probe_residual", "C2_max_subject_residual", "C3_max_worst_residual",
                                 "C4_stability_then_worst_residual", "C5_min_train_FAR_among_all_positive",
                                 "P1_longest_4bit_both_weightings")}
        test15 = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(NS)
            tr, te = np.sort(perm[NS // 2:]), np.sort(perm[:NS // 2])
            ctr = np.zeros(NS); ctr[tr] = 1; cte = np.zeros(NS); cte[te] = 1
            res = {}
            for n, Lo in LF.items():
                ne = Lo.neff(ctr)
                res[n] = {w: (ne - Lo.choose(ctr, w)[0]) for w in WGT}
                res[n]["p_probe"], res[n]["t_probe"] = Lo.choose(ctr, "probe")
                res[n]["p_subject"] = Lo.choose(ctr, "subject")[0]
            worst = {n: min(res[n][w] for w in WGT) for n in NSET}
            pos = [n for n in NSET if worst[n] > 0 and n - res[n]["p_probe"] > 0]
            picks["C1_max_probe_residual"].append(max(NSET, key=lambda n: res[n]["probe"]))
            picks["C2_max_subject_residual"].append(max(NSET, key=lambda n: res[n]["subject"]))
            picks["C3_max_worst_residual"].append(max(NSET, key=worst.get))
            stab = {}
            for n in pos:
                same = 0
                for _ in range(30):
                    cb = np.bincount(irng.choice(tr, len(tr)), minlength=NS).astype(float)
                    same += LF[n].choose(cb, "probe")[0] == res[n]["p_probe"]
                stab[n] = same / 30
            picks["C4_stability_then_worst_residual"].append(
                max(pos, key=lambda n: (stab[n], worst[n])) if pos else None)
            picks["C5_min_train_FAR_among_all_positive"].append(
                min(pos, key=lambda n: LF[n].far_pair(res[n]["t_probe"], ctr)) if pos else None)
            prov = [n for n in NSET if n <= 11 and res[n]["p_probe"] == 4 and res[n]["p_subject"] == 4]
            picks["P1_longest_4bit_both_weightings"].append(max(prov) if prov else None)
            L15 = LF[15]; p15, _ = L15.choose(ctr, "probe"); ps15, _ = L15.choose(ctr, "subject")
            test15.append((p15, L15.neff(cte) - p15, L15.neff(cte) - ps15))
        freq = {}
        for k, v in picks.items():
            vals = [x if x is not None else -1 for x in v]
            u, c = np.unique(vals, return_counts=True)
            freq[k] = {int(a): 100 * float(b) / len(vals) for a, b in zip(u, c)}
        T15 = np.array(test15, float)
        SEL[tag] = {"split_seed": sseed, "selection_freq_pct": freq,
                    "modal_choice": {k: max(f, key=f.get) for k, f in freq.items()},
                    "n15_train_parity8_pct": 100 * float((T15[:, 0] == 8).mean()),
                    "n15_test_residual_probe_mean": float(T15[:, 1].mean()),
                    "n15_test_residual_probe_positive_pct": 100 * float((T15[:, 1] > 0).mean()),
                    "n15_test_residual_subject_mean": float(T15[:, 2].mean()),
                    "n15_test_residual_subject_positive_pct": 100 * float((T15[:, 2] > 0).mean())}
        log(f"(f) length selection, split seed {sseed}: modal {SEL[tag]['modal_choice']}")
        log(f"    C5 freq {freq['C5_min_train_FAR_among_all_positive']}")
    F_["length_selection"] = SEL
    R["f_seeds_heldout"] = F_

    # ======================================================================= decision rules
    ho = F_["held_out"]["43/15/paper"]
    R["decisions"] = {
        "n15_subject_residual_full_sample": LD["15"]["weightings"]["subject"]["residual"],
        "n15_subject_residual_positive": LD["15"]["weightings"]["subject"]["residual"] > 0,
        "n15_heldout_subject_test_residual_positive_pct": ho["subject"]["test_residual_positive_pct"],
        "n15_modal_C5_paper_split": SEL["paper"]["modal_choice"]["C5_min_train_FAR_among_all_positive"],
        "n9_bound_min_probe": B_[9]["probe"]["min"], "n9_bound_min_subject": B_[9]["subject"]["min"],
        "n9_bound_min_bonf_probe": B_[9]["probe"]["min_bonferroni"],
        "n9_bound_min_bonf_subject": B_[9]["subject"]["min_bonferroni"],
        "alternatives_beating_n15_under_subject": alt_sw}
    R["wall_seconds"] = round(time.time() - T0, 1)
    dump(R, "step4_robustness.json")
    log("decisions", R["decisions"])
    log("done", R["wall_seconds"], "s")


if __name__ == "__main__":
    main()
