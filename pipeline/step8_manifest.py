"""Step 8: write results/pipeline/manifest.csv, one row per number.

Columns: quantity | value | units | script | function | seeds | commit.
Every value is read from the JSON files written by steps 1-7 of this run (each
records the commit it ran from). Derived quantities (ratios, sets, counts the
paper states in words) are computed here and say so in the function column.
Run it after step 7, or the four check-script rows are left out.

    python -m pipeline.step8_manifest
"""
import csv
import json
import math

from pipeline.provenance import DEV, OUT, OUT_MAIN, require_committed_code

MEASURED = [9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 32, 40, 48, 56, 64]
WGT = ("probe", "capped", "subject")


def main():
    st = require_committed_code()
    J = lambda name: json.load(open(OUT_MAIN / name, encoding="utf-8"))
    env, sweep, robust = J("step1_embed.json"), J("step2_sweep.json"), J("step4_robustness.json")
    lfw = J("step3_lfw_benchmark.json")
    if DEV and not (OUT_MAIN / "step6_alignment.json").exists():      # preview only; never the frozen manifest
        al = {"git": {"commit": "dev-preview"}, "lengths": [], "corpus_fallback_center_crop": None}
    else:
        al = J("step6_alignment.json")
    ver = J("step7_checks.json") if (OUT_MAIN / "step7_checks.json").exists() else None
    LD = sweep["lengths_detail"]
    C = {"step1": env["git"]["commit"], "step2": sweep["git"]["commit"], "step3": lfw["git"]["commit"],
         "step4": robust["git"]["commit"], "step6": al["git"]["commit"],
         "step8": st["commit"]}
    rows = []

    def add(q, v, units, script, func, seeds, commit):
        if isinstance(v, bool):
            v = "True" if v else "False"
        elif isinstance(v, float):
            v = repr(v)
        elif isinstance(v, (list, dict)):
            v = json.dumps(v, separators=(",", ":"))
        rows.append({"quantity": q, "value": v, "units": units, "script": script, "function": func,
                     "seeds": seeds, "commit": commit[:12]})

    SWEEP = "pipeline/step2_sweep.py"; ROBUST = "pipeline/step4_robustness.py"; THIS = "pipeline/step8_manifest.py"
    proj = "projection 43"

    # ------------------------------------------------------------------ corpus / environment
    add("corpus subjects", env["corpus_subjects"], "count", "pipeline/step1_embed.py", "fuzzycommit.embedding.load_subject_images", "-", C["step1"])
    add("corpus images", env["corpus_images"], "count", "pipeline/step1_embed.py", "fuzzycommit.embedding.load_subject_images", "-", C["step1"])
    add("corpus probes", sweep["corpus"]["probes"], "count", SWEEP, "fuzzycommit.common.build_subject_info", proj, C["step2"])
    add("corpus enrollment images per subject", 5, "count", SWEEP, "fuzzycommit.common.ENROLLMENT_SIZE", "-", C["step2"])
    add("corpus min images per subject", 6, "count", SWEEP, "fuzzycommit.common.MIN_IMAGES_PER_SUBJECT", "-", C["step2"])
    add("embedding dimension", 512, "count", "pipeline/step1_embed.py", "w600k_r50.onnx output", "-", C["step1"])
    add("candidate projections", 512, "count", SWEEP, "fuzzycommit.common.N_PROJECTIONS", "-", C["step2"])
    add("bootstrap replicates", 1000, "count", SWEEP, "fuzzycommit.common.N_BOOTSTRAP", "-", C["step2"])
    add("impostor trials per length", 1000000, "count", SWEEP, "fuzzycommit.common.N_IMPOSTOR", "-", C["step2"])
    add("decode failure target", 0.015, "fraction", SWEEP, "fuzzycommit.common.DECODE_FAILURE_TARGET", "-", C["step2"])
    add("probe cap per subject", 10, "count", SWEEP, "fuzzycommit.common.PROBE_CAP", "-", C["step2"])
    add("confidence level of bound", 0.99, "fraction", SWEEP, "scipy beta.ppf(0.99,k+1,N-k)", "-", C["step2"])
    add("impostor comparisons full set", LD["24"]["exhaustive"]["comparisons"], "count", SWEEP,
        "fuzzycommit.false_accept.run_exhaustive_far", "-", C["step2"])
    add("lfw manifest sha256", env["lfw_manifest_sha256"], "sha256", "pipeline/step1_embed.py", "sorted image paths", "-", C["step1"])
    add("corpus content sha256", env["corpus_content_sha256"], "sha256", "pipeline/step1_embed.py", "paths+file bytes", "-", C["step1"])
    add("subject list sha256", env["subject_list_sha256"], "sha256", "pipeline/step1_embed.py", "fuzzycommit.common.subject_list_hash", "-", C["step1"])
    add("probe set sha256", env["probe_set_sha256"], "sha256", "pipeline/step1_embed.py", "fuzzycommit.common.probe_set_hash", proj, C["step1"])
    add("embedding npz sha256", env["embeddings_npz_sha256"], "sha256", "pipeline/step1_embed.py", "fuzzycommit.embedding.extract_embeddings", "-", C["step1"])
    add("projection matrix sha256", sweep["shared_projection"]["W_sha256"], "sha256", SWEEP, "fuzzycommit.common.get_W", proj, C["step2"])
    add("projection matrices observed across enrollees", sweep["shared_projection"]["distinct_matrices_observed"],
        "count", SWEEP, "spy on fuzzycommit.common.enroll_select over 622 calls", proj, C["step2"])

    # ------------------------------------------------------------------ per length
    for ns, d in LD.items():
        n = int(ns)
        fs = f"{proj}; boot {d['bootstrap']['seed']}; far {d['sampled']['seed']}"
        add(f"m n={n}", d["m"], "-", SWEEP, "fuzzycommit.common.compute_bch_m", "-", C["step2"])
        add(f"N_eff n={n}", d["neff"], "bits", SWEEP, "fuzzycommit.false_accept.compute_neff", proj, C["step2"])
        add(f"N_eff/n n={n}", d["neff_over_n"], "ratio", SWEEP, "fuzzycommit.false_accept.compute_neff / n", proj, C["step2"])
        add(f"dispersion n={n}", d["dispersion"], "x binomial SD", SWEEP, "fuzzycommit.bit_errors.compute_sweep_row", proj, C["step2"])
        add(f"t_required n={n}", d["t_required_designed"], "errors", SWEEP, "fuzzycommit.common.t_empirical", proj, C["step2"])
        add(f"t_binomial n={n}", d["t_binomial"], "errors", SWEEP, "binom.sf(t,n,beta)<0.015", proj, C["step2"])
        for w in WGT:
            x = d["weightings"][w]
            sfx = "" if w == "probe" else f" {w}"
            fn = {"probe": "fuzzycommit.common.select_capacity", "capped": "fuzzycommit.common.select_capacity (first 10 probes)",
                  "subject": "least parity, subject-mean failure (= fuzzycommit.weighting.run_capped_and_weighted)"}[w]
            add(f"beta n={n}{sfx}", x["beta"], "bit-error rate", SWEEP, "fuzzycommit.bit_errors.compute_sweep_row" if w == "probe" else fn, proj, C["step2"])
            add(f"t n={n}{sfx}", x["t"], "errors (delivered)", SWEEP, fn, proj, C["step2"])
            add(f"t_designed n={n}{sfx}", x["t_designed"], "errors (designed)", SWEEP, fn, proj, C["step2"])
            add(f"parity n={n}{sfx}", x["parity"], "bits", SWEEP, fn + "; fuzzycommit.common.bch_parity", proj, C["step2"])
            add(f"k n={n}{sfx}", x["k"], "bits", SWEEP, "n - parity", proj, C["step2"])
            add(f"decode_fail n={n}{sfx}", x["decode_fail"], "fraction", SWEEP, fn, proj, C["step2"])
            add(f"residual n={n}{sfx}", x["residual"], "bits", SWEEP, "N_eff - parity", proj, C["step2"])
            b = x["bound"]
            add(f"bound_mean n={n}{sfx}", b["mean"], "bits", SWEEP, "mean_i -log2 CP99(MCV_i) - parity", proj, C["step2"])
            add(f"bound_min n={n}{sfx}", b["min"], "bits", SWEEP, "min_i -log2 CP99(MCV_i) - parity", proj, C["step2"])
            add(f"bound_min_bonferroni n={n}{sfx}", b["min_bonferroni"], "bits", SWEEP,
                "min_i -log2 CP(1-0.01/311)(MCV_i) - parity", proj, C["step2"])
            add(f"bound_index_sets_positive n={n}{sfx}", b["index_sets_positive"], "of 311", SWEEP, "count floor_i > parity", proj, C["step2"])
            add(f"bound_index_sets_positive_bonferroni n={n}{sfx}", b["index_sets_positive_bonferroni"], "of 311", SWEEP,
                "count floor_i > parity at alpha 0.01/311", proj, C["step2"])
            bt = d["bootstrap"][w]
            add(f"closure_pct n={n}{sfx}", bt["closure_pct"], "%", SWEEP,
                "fuzzycommit.bootstrap.run_bootstrap (probe) / same replicates (others)", fs, C["step2"])
            add(f"closure_mixture n={n}{sfx}", [{"parity": r["parity"], "share": round(r["share_pct"], 2),
                                                "pos": round(r["pct_positive_within"], 2),
                                                "res": round(r["residual_mean"], 4)} for r in bt["branches"]],
                "json", SWEEP, "closure = sum_p share_p * P(res>0 | parity p)", fs, C["step2"])
            add(f"residual_ci95 n={n}{sfx}", bt["residual_ci95"], "bits", SWEEP, "bootstrap percentile", fs, C["step2"])
        add(f"bootstrap N_eff SD n={n}", d["bootstrap"]["neff_sd"], "bits", SWEEP, "N_eff per replicate", fs, C["step2"])
        mc = d["mcv"]
        add(f"mcv_floor_mean n={n}", mc["floor_mean"], "bits", SWEEP, "-log2 CP99 upper bound, mean over 311 index sets", proj, C["step2"])
        add(f"mcv_floor_min n={n}", mc["floor_min"], "bits", SWEEP, "min over 311 index sets", proj, C["step2"])
        add(f"mcv_floor_bonferroni_min n={n}", mc["floor_bonf_min"], "bits", SWEEP, "min, alpha 0.01/311", proj, C["step2"])
        t = d["weightings"]["probe"]["t"]
        sa = d["sampled"]
        add(f"FAR_sampled_accepts n={n}", sa["accepts"], "count", SWEEP, "fuzzycommit.false_accept.run_sampled_far", f"far {sa['seed']}", C["step2"])
        add(f"FAR_sampled n={n}", sa["accepts"] / 1e6, "rate", SWEEP, "fuzzycommit.false_accept.run_sampled_far", f"far {sa['seed']}", C["step2"])
        add(f"FAR_independent n={n}", d["far_analytical_by_t"][t], "rate", SWEEP, "fuzzycommit.false_accept.compute_analytical_far", "-", C["step2"])
        add(f"FAR_ratio_sampled n={n}", sa["far_ratio"], "x", SWEEP, "sampled / independent-bits", f"far {sa['seed']}", C["step2"])
        add(f"FAR_one_in n={n}", 1e6 / sa["accepts"], "impostors", THIS, "1 / FAR_sampled", f"far {sa['seed']}", C["step8"])
        ex = d["exhaustive"]
        add(f"FAR_full_pair n={n}", ex["far_pair_uniform_by_t"][t], "rate", SWEEP, "fuzzycommit.false_accept.run_exhaustive_far (pair-uniform)", "-", C["step2"])
        add(f"FAR_full_comparison n={n}", ex["far_comparison_uniform_by_t"][t], "rate", SWEEP, "fuzzycommit.false_accept.run_exhaustive_far (comparison-uniform)", "-", C["step2"])
        add(f"FAR_full_accepts n={n}", ex["accepts_by_t"][t], "of 1681750", SWEEP, "fuzzycommit.false_accept.run_exhaustive_far", "-", C["step2"])
        add(f"FAR_ratio_full_pair n={n}", ex["far_pair_uniform_by_t"][t] / d["far_analytical_by_t"][t], "x", SWEEP, "full pair-uniform / independent", "-", C["step2"])
        add(f"FAR_ratio_full_comparison n={n}", ex["far_comparison_uniform_by_t"][t] / d["far_analytical_by_t"][t], "x", SWEEP, "full comparison-uniform / independent", "-", C["step2"])
        for tt in range(0, min(n, 8) + 1):
            dt = d["decode_fail_by_t"][str(tt)]
            if dt["binomial"] > 0:
                add(f"decode_ratio n={n} t={tt}", dt["empirical"] / dt["binomial"], "x", SWEEP,
                    "empirical fail / binom.sf(t,n,beta)", proj, C["step2"])
        add(f"mt n={n}", d["m"] * d["weightings"]["probe"]["t_designed"], "bits", THIS, "m x designed t (probe)", proj, C["step8"])
        for i, tr in enumerate(d["tertiles"], 1):
            add(f"tertile{i} n={n} subjects", tr["subjects"], "count", SWEEP, "fuzzycommit.weighting.run_tertiles", proj, C["step2"])
            add(f"tertile{i} n={n} probes", tr["probes"], "count", SWEEP, "fuzzycommit.weighting.run_tertiles", proj, C["step2"])
            add(f"tertile{i} n={n} N_eff", tr["neff"], "bits", SWEEP, "N_eff within tertile", proj, C["step2"])
            add(f"tertile{i} n={n} beta", tr["beta"]["probe"], "bit-error rate", SWEEP, "fuzzycommit.weighting.run_tertiles", proj, C["step2"])
            for w in WGT:
                sfx = "" if w == "probe" else f" {w}"
                add(f"tertile{i} n={n} t{sfx}", tr["weightings"][w]["t"], "errors", SWEEP, "within-tertile selection", proj, C["step2"])
                add(f"tertile{i} n={n} parity{sfx}", tr["weightings"][w]["parity"], "bits", SWEEP, "within-tertile selection", proj, C["step2"])
                add(f"tertile{i} n={n} residual{sfx}", tr["weightings"][w]["residual"], "bits", SWEEP, "N_eff - parity within tertile", proj, C["step2"])
        fr = d["per_subject_frr"]["summary"]
        for k in ("mean", "median", "p90", "p99", "max", "n_gt_5pct", "n_gt_10pct", "n_zero"):
            add(f"frr_{k} n={n}", fr[k], "fraction" if k in ("mean", "median", "p90", "p99", "max") else "subjects",
                SWEEP, f"per-subject decode failure at t={d['per_subject_frr']['t']}", proj, C["step2"])

    # ------------------------------------------------------------------ ablation, decorrelation, noise, LFW
    for key, r in sweep["ablation"].items():
        for f in ("beta", "dispersion", "t", "t_designed", "parity", "k", "neff", "residual", "t_required_designed", "t_binomial"):
            # t_required = smallest raw Hamming radius meeting 1.5% (fuzzycommit.common.t_empirical)
            add(f"arm{key[0]} n={key[1:]} {f.replace('t_required_designed', 't_required')}", r[f], "-", SWEEP,
                "fuzzycommit.common.build_subject_info%s + compute_sweep_row" % ("" if key[0] == "A" else "_random (seed 2042+i)"),
                proj + ("" if key[0] == "A" else "; arm B 2042+i"), C["step2"])
    for n, r in sweep["decorrelation"].items():
        add(f"decorrelation gain n={n}", r["gain"], "bits", SWEEP, "greedy |corr|<tau selection, best tau", proj, C["step2"])
        add(f"decorrelation best tau n={n}", r["best_tau"], "-", SWEEP, "taus 0.5..0.05", proj, C["step2"])
    nc = sweep["noise_control"]
    add("noise control FAR ratio n=24", nc["ratio"], "x", SWEEP, "isotropic Gaussian, beta matched by bisection", "means 11; noise 500+i", C["step2"])
    add("noise control beta n=24", nc["beta"], "bit-error rate", SWEEP, "bisection", "means 11; noise 500+i", C["step2"])
    add("LFW FAR ratio pair-uniform n=24", nc["lfw_ratio_pair"], "x", SWEEP, "full pair-uniform / independent", "-", C["step2"])
    add("LFW 6000-pair accuracy", lfw["accuracy_mean_pct"], "%", "pipeline/step3_lfw_benchmark.py", "fuzzycommit.embedding.extract_embeddings + 10-fold", "-", C["step3"])
    add("LFW 6000-pair accuracy SD", lfw["accuracy_sd_pct"], "%", "pipeline/step3_lfw_benchmark.py", "10-fold SD", "-", C["step3"])

    # ------------------------------------------------------------------ step 4 (robustness)
    for n, o in robust["a_codes"].items():
        for fam, rws in o["families"].items():
            for w, r in rws.items():
                for f in ("parity", "t", "k", "residual", "decode_fail", "closure_pct", "far_sampled", "far_full_pair", "far_independent"):
                    add(f"code n={n} [{fam}] {w} {f}", r[f], "-", ROBUST, "pipeline.bch_codes + least parity under weighting",
                        f"{proj}; boot {42 + 500 + int(n)}; far {42 + 300 + int(n)}", C["step4"])
    for tg, r in robust["d_throughput"].items():
        ch = r["chosen"]
        add(f"throughput target={tg} chosen n (longest subject-positive)", r["chosen_n_longest_subject_positive"], "n", ROBUST, "longest measured n with subject residual>0", proj, C["step4"])
        if ch:
            for f in ("t", "parity", "residual", "decode_fail"):
                add(f"throughput target={tg} chosen {f}", ch[f], "-", ROBUST, "subject weighting", proj, C["step4"])
        add(f"throughput target={tg} subject-positive lengths", r["subject_positive_measured"], "list", ROBUST, "measured lengths", proj, C["step4"])
        add(f"throughput target={tg} max subject residual n", r["max_subject_residual_n"], "n", ROBUST, "argmax", proj, C["step4"])
        for f in ("t", "parity", "residual"):
            add(f"throughput target={tg} n=15 subject {f}", r["n15"][f], "-", ROBUST, "subject weighting", proj, C["step4"])
    for key, r in robust["f_seeds_heldout"]["in_sample"].items():
        ps, n = key.split("/")
        add(f"seed{ps} n={n} beta", r["beta"], "bit-error rate", ROBUST, "fuzzycommit.common.get_W(seed)", f"projection {ps}", C["step4"])
        add(f"seed{ps} n={n} N_eff", r["neff"], "bits", ROBUST, "fuzzycommit.false_accept.impostor_pair_stats", f"projection {ps}", C["step4"])
        add(f"seed{ps} n={n} closure_pct", r["closure_pct_probe"], "%", ROBUST, "fuzzycommit.bootstrap.run_bootstrap_seed_sensitivity",
            f"projection {ps}; boot {r['boot_seed']}", C["step4"])
        for w in WGT:
            for f in ("parity", "t", "residual", "bound_mean", "bound_min", "bound_min_bonferroni", "decode_fail", "far_full_pair"):
                add(f"seed{ps} n={n} {w} {f}", r[w][f], "-", ROBUST, "per projection seed", f"projection {ps}", C["step4"])
    for key, r in robust["f_seeds_heldout"]["held_out"].items():
        ps, n, tag = key.split("/")
        for f in ("half_sample_floor_test_mean", "half_sample_floor_train_mean", "floor_cap_at_N155"):
            add(f"heldout proj={ps} split={tag}({r['split_seed']}) n={n} {f}", r[f], "bits", ROBUST,
                "MCV floor on one half (N=155/156 enrolled syndromes)", f"projection {ps}; split {r['split_seed']}", C["step4"])
        for w in WGT:
            for f, v in r[w].items():
                add(f"heldout proj={ps} split={tag}({r['split_seed']}) n={n} {w} {f}", v, "-", ROBUST,
                    "400 subject half-splits: choose on train half, evaluate on test half",
                    f"projection {ps}; split {r['split_seed']}", C["step4"])
    for tag, r in robust["f_seeds_heldout"]["length_selection"].items():
        for k, f in r["selection_freq_pct"].items():
            add(f"heldout length selection split={tag}({r['split_seed']}) {k}", f, "% of splits", ROBUST,
                "train-half criterion over n=8..24", f"{proj}; split {r['split_seed']}; inner {r['split_seed'] + 1}", C["step4"])
        for k in ("n15_train_parity8_pct", "n15_test_residual_probe_mean", "n15_test_residual_probe_positive_pct",
                  "n15_test_residual_subject_mean", "n15_test_residual_subject_positive_pct"):
            add(f"heldout split={tag}({r['split_seed']}) {k}", r[k], "-", ROBUST, "400 subject half-splits", f"{proj}; split {r['split_seed']}", C["step4"])
    for n, r in robust["e_ratios"].items():
        for f in ("decode_ratio", "far_ratio_sampled", "far_ratio_full_pair", "far_ratio_full_comparison"):
            add(f"ratio n={n} {f}", r[f], "x", ROBUST, "empirical / independent-bits", proj, C["step4"])
    for n in al["lengths"]:
        for side in ("center_crop", "aligned"):
            for f, v in al[side][str(n)].items():
                add(f"alignment n={n} {side} {f}", v, "-", "pipeline/step6_alignment.py", "SCRFD det_10g + Umeyama to ArcFace template", proj, C["step6"])
    add("alignment corpus fallbacks", al["corpus_fallback_center_crop"], "images", "pipeline/step6_alignment.py", "no face detected", "-", C["step6"])
    if "lfw_pairs_aligned_accuracy_pct" in al:
        add("alignment LFW 6000-pair accuracy", al["lfw_pairs_aligned_accuracy_pct"], "%", "pipeline/step6_alignment.py", "aligned embeddings, 10-fold", "-", C["step6"])

    # ------------------------------------------------------------------ derived quantities the prose states
    from fuzzycommit.common import bch_parity, bch_true_t
    g = lambda n, k: LD[str(n)][k]
    W_ = lambda n, w, k: LD[str(n)]["weightings"][w][k]
    D = lambda q, v, units, how: add(q, v, units, THIS, how, proj, C["step8"])
    sw_pos = [n for n in MEASURED if W_(n, "subject", "residual") > 0 and W_(n, "subject", "k") > 0]
    run = 0
    for n in MEASURED:
        if n in sw_pos: run = n
        else: break
    D("subject-positive measured lengths", sw_pos, "list", "measured n with subject residual > 0")
    D("subject-positive contiguous from shortest, max n", run, "n", "longest prefix of measured lengths")
    D("Fig2 band edge", run + 0.5, "n", "subject-positive max n + 0.5")
    D("probe residual positive at every measured n<=24", all(W_(n, "probe", "residual") > 0 for n in MEASURED if n <= 24), "bool", "all()")
    D("probe residual positive at no measured n>24", all(W_(n, "probe", "residual") <= 0 for n in MEASURED if n > 24), "bool", "all()")
    bpos = [n for n in range(8, 25) if W_(n, "probe", "bound")["mean"] > 0]
    bpos_both = [n for n in range(8, 25) if W_(n, "probe", "bound")["mean"] > 0 and W_(n, "subject", "bound")["mean"] > 0]
    D("bound-positive lengths n<=24 (probe)", bpos, "list", "bound_mean > 0")
    D("bound-positive max n (probe)", max(bpos), "n", "max of list")
    D("bound-positive lengths are the 4-bit plateau", all(W_(n, "probe", "parity") == 4 for n in bpos) and bpos == [8, 9, 10, 11], "bool", "parity==4 for all and set == 8..11")
    D("bound-positive both weightings max n", max(bpos_both), "n", "probe and subject bound_mean > 0")
    D("bound-positive both weightings contiguous from 8", bpos_both == list(range(8, max(bpos_both) + 1)), "bool", "set check")
    D("floor 4-bit plateau 8..11 all exceed 4", all(g(n, "mcv")["floor_mean"] > 4 for n in (8, 9, 10, 11)), "bool", "floor > 4")
    D("floor cap after n=24 (k=1)", -math.log2(__import__("scipy.stats", fromlist=["beta"]).beta.ppf(0.99, 2, 311 - 1)), "bits", "-log2 CP99(k=1, N=311)")
    D("floor cap below every parity cost n>24", all(W_(n, "probe", "parity") > 5.6 for n in (32, 40, 48, 56, 64)), "bool", "parity > cap")
    D("parity step t2->t3 m=4", bch_parity(4, 3) - bch_parity(4, 2), "bits", "fuzzycommit.common.bch_parity")
    D("parity step t2->t3 m=5", bch_parity(5, 3) - bch_parity(5, 2), "bits", "fuzzycommit.common.bch_parity")
    for t in (8, 9, 10):
        D(f"parity m=6 t={t}", bch_parity(6, t), "bits", "fuzzycommit.common.bch_parity")
    D("parity m=5 t=4", bch_parity(5, 4), "bits", "fuzzycommit.common.bch_parity")
    D("delivered t m=5 designed t=4", bch_true_t(5, 4), "errors", "fuzzycommit.common.bch_true_t")
    D("mt equals exact parity at every plotted m<=5", all(g(n, "m") * W_(n, "probe", "t_designed") == W_(n, "probe", "parity") for n in MEASURED if g(n, "m") <= 5), "bool", "m*t == p")
    D("mt differs from exact parity at every plotted m>=6", all(g(n, "m") * W_(n, "probe", "t_designed") != W_(n, "probe", "parity") for n in MEASURED if g(n, "m") >= 6), "bool", "m*t != p")
    D("n=24 next-plateau residual (parity 20)", g(24, "neff") - 20, "bits", "N_eff - 20")
    br24 = {b["parity"]: b for b in LD["24"]["bootstrap"]["probe"]["branches"]}
    D("n=24 closure non-closing branch needs t>=4 (parity 20)", set(br24) == {15, 20} and br24[20]["pct_positive_within"] == 0, "bool", "mixture branches")
    D("cos needed for unselected projection (n=15 beta)", math.cos(math.pi * g(15, "beta")), "cosine", "cos(pi*beta), Charikar")
    D("n=15 same code under all three weightings", len({W_(15, w, "parity") for w in WGT}) == 1 and W_(15, "probe", "parity") == 8, "bool", "parity equality")
    D("n=15 min residual over weightings", min(W_(15, w, "residual") for w in WGT), "bits", "min")
    D("n=15 max residual over weightings", max(W_(15, w, "residual") for w in WGT), "bits", "max")
    D("n=15 positive in all three tertiles (probe)", all(t["weightings"]["probe"]["residual"] > 0 for t in LD["15"]["tertiles"]), "bool", "all()")
    D("n=15 positive in all tertiles under all weightings", all(t["weightings"][w]["residual"] > 0 for t in LD["15"]["tertiles"] for w in WGT), "bool", "all()")
    D("n=15 three-error residual over two-error residual", (g(15, "neff") - 10) / (g(15, "neff") - 8), "ratio", "(N_eff-10)/(N_eff-8)")
    D("n=15 decode fail pct", 100 * W_(15, "probe", "decode_fail"), "%", "x100")
    D("n=15 FAR sampled per thousand", g(15, "sampled")["accepts"] / 1e3, "per 1000", "accepts/1e3")
    D("n=11 index sets positive (probe)", W_(11, "probe", "bound")["index_sets_positive"], "of 311", "count")
    ab = sweep["ablation"]
    D("arm B / arm A beta ratio n=24", ab["B24"]["beta"] / ab["A24"]["beta"], "x", "ratio")
    D("arm A N_eff/n n=24", ab["A24"]["neff"] / 24, "ratio", "N_eff/24")
    D("arm B N_eff/n n=24", ab["B24"]["neff"] / 24, "ratio", "N_eff/24")
    D("arm B - arm A N_eff n=24", ab["B24"]["neff"] - ab["A24"]["neff"], "bits", "difference")
    D("arm gap grows with length 16<24<32", (ab["B16"]["neff"] - ab["A16"]["neff"]) < (ab["B24"]["neff"] - ab["A24"]["neff"]) < (ab["B32"]["neff"] - ab["A32"]["neff"]), "bool", "monotone")
    D("arm B closes at no length", all(ab[f"B{n}"]["residual"] < 0 for n in (16, 24, 32)), "bool", "all residual < 0")
    D("arm B max |t_required - t_binomial|", max(abs(ab[f"B{n}"]["t_required_designed"] - ab[f"B{n}"]["t_binomial"]) for n in (16, 24, 32)), "steps", "max")
    D("arm A min dispersion", min(ab[f"A{n}"]["dispersion"] for n in (16, 24, 32)), "x", "min")
    D("arm A max dispersion", max(ab[f"A{n}"]["dispersion"] for n in (16, 24, 32)), "x", "max")
    D("arm A min overshoot t_required - t_binomial", min(ab[f"A{n}"]["t_required_designed"] - ab[f"A{n}"]["t_binomial"] for n in (16, 24, 32)), "steps", "min")
    D("arm A max overshoot t_required - t_binomial", max(ab[f"A{n}"]["t_required_designed"] - ab[f"A{n}"]["t_binomial"] for n in (16, 24, 32)), "steps", "max")
    D("overshoot t_required - t_binomial n=64", g(64, "t_required_designed") - g(64, "t_binomial"), "steps", "difference")
    D("top-5 subjects share of probes", s1_top5(OUT_MAIN), "%", "sum of 5 largest probe counts / 3870")
    seeds = robust["f_seeds_heldout"]["in_sample"]
    P = (43, 101, 202, 303, 404)
    D("n=11 seeds needing parity 8 (probe)", sum(seeds[f"{p}/11"]["probe"]["parity"] == 8 for p in P), "of 5", "count")
    D("n=9 min bound over seeds and both weightings", min(seeds[f"{p}/9"][w]["bound_mean"] for p in P for w in ("probe", "subject")), "bits", "min")
    D("n=9 max bound over seeds and both weightings", max(seeds[f"{p}/9"][w]["bound_mean"] for p in P for w in ("probe", "subject")), "bits", "max")
    b15 = [seeds[f"{p}/15"]["beta"] for p in P]
    m15 = sum(b15) / 5
    D("n=15 beta CV across seeds", 100 * (sum((x - m15) ** 2 for x in b15) / 4) ** 0.5 / m15, "%", "stdev/mean")
    D("n=15 beta range across seeds over mean", 100 * (max(b15) - min(b15)) / m15, "%", "(max-min)/mean")
    D("n=15 same code all five seeds", all(seeds[f"{p}/15"]["probe"]["parity"] == 8 for p in P), "bool", "all parity 8")
    ho = robust["f_seeds_heldout"]["held_out"]
    D("held-out n=9 keep 4-bit code (probe)", ho["43/9/paper"]["probe"]["share_train_choice_eq_full_sample_pct"], "% of splits", "train-half choice")
    D("held-out n=11 need 8-bit code (probe)", ho["43/11/paper"]["probe"]["parity_freq"].get("8", 0.0), "% of splits", "train-half choice")
    D("n=16 N_eff/n", g(16, "neff_over_n"), "ratio", "N_eff/16")
    D("n=64 N_eff/n", g(64, "neff_over_n"), "ratio", "N_eff/64")
    D("deploy authorization time", 0.159, "s (assumption)", "stated assumption")
    D("deploy lane period", 2.86, "s (assumption, TCRP 165)", "stated assumption")
    D("deploy lanes", 3, "lanes (assumption)", "stated assumption")
    D("deploy required throughput", 62, "people/min (assumption)", "stated assumption")
    D("deploy authorization share of lane period", 100 * 0.159 / 2.86, "%", "0.159/2.86")
    D("deploy throughput", 3 * 60 / 2.86, "people/min", "3*60/2.86")
    D("deploy max decode failure", 100 * (1 - 62 / 63), "%", "1 - 62/63")
    D("deploy throughput with retries n=15", 63 * (1 - W_(15, "probe", "decode_fail")), "people/min", "63*(1-f15)")
    D("max decode ratio (n=24, t=5)", LD["24"]["decode_fail_by_t"]["5"]["empirical"] / LD["24"]["decode_fail_by_t"]["5"]["binomial"], "x", "ratio")
    a16 = robust["a_codes"]["16"]["families"]["shortened BCH + extended BCH"]["subject"]
    a32 = robust["a_codes"]["32"]["families"]["shortened BCH + extended BCH"]["probe"]
    a23 = robust["a_codes"]["23"]["families"]["shortened BCH + Golay"]
    D("extended BCH n=16 subject parity", a16["parity"], "bits", "robust a_codes")
    D("extended BCH n=32 probe parity", a32["parity"], "bits", "robust a_codes")
    D("extended BCH n=16 subject closure", a16["closure_pct"], "%", "robust a_codes, seed 558")
    D("extended BCH n=32 probe closure", a32["closure_pct"], "%", "robust a_codes, seed 574")
    D("Golay n=23 probe residual", a23["probe"]["residual"], "bits", "robust a_codes")
    D("Golay n=23 subject residual", a23["subject"]["residual"], "bits", "robust a_codes")
    D("measured lengths", len(MEASURED), "count", "list")
    D("Table 1 rows", 9, "count", "rows printed")
    pw_run = 0
    for n in MEASURED:
        if W_(n, "probe", "residual") > 0 and W_(n, "probe", "k") > 0: pw_run = n
        else: break
    D("probe-positive max measured n", pw_run, "n", "longest prefix of measured lengths with probe residual > 0")
    D("first plotted m where mt != exact parity",
      min(g(n, "m") for n in MEASURED if g(n, "m") * W_(n, "probe", "t_designed") != W_(n, "probe", "parity")), "m", "min")
    D("n=24 non-closing branch designed t", next(t for t in range(25) if bch_parity(5, t) == 20), "errors",
      "smallest designed t with parity 20 at m=5")
    from pipeline.bch_codes import bch_generator, min_distance, shortened_cyclic
    D("BCH(15,7) minimum distance", min_distance(shortened_cyclic(bch_generator(4, 2), 15), 15), "bits",
      "pipeline.bch_codes exhaustive enumeration")
    D("n=15 residual positive under all weightings", all(W_(15, w, "residual") > 0 for w in WGT), "bool", "all()")
    D("tertile count", len(LD["15"]["tertiles"]), "count", "fuzzycommit.weighting.run_tertiles")
    D("bound-positive min n (probe)", min(bpos), "n", "min of list")
    D("bound argmax n (probe)", max(range(8, 25), key=lambda n: W_(n, "probe", "bound")["mean"]), "n", "argmax bound_mean")
    D("n=11 subject bound negative", W_(11, "subject", "bound")["mean"] < 0, "bool", "bound_mean n=11 subject < 0")
    D("projection seeds", len(P), "count", "seeds 43,101,202,303,404")
    D("arm B parity exceeds n at n=24", ab["B24"]["parity"] > 24, "bool", "parity > n")
    t15 = LD["15"]["tertiles"]
    D("tertile1 n=15 has the highest beta", t15[0]["beta"]["probe"] == max(t["beta"]["probe"] for t in t15), "bool", "max")
    D("delivered t of the next code at n=15",
      bch_true_t(4, next(t for t in range(16) if bch_parity(4, t) > W_(15, "probe", "parity"))), "errors",
      "next parity plateau above 8 at m=4")
    modal = robust["f_seeds_heldout"]["length_selection"]["paper"]["modal_choice"]
    D("held-out criteria disagree on the winning length",
      len({v for k, v in modal.items() if k.startswith("C")}) > 1, "bool", "modal winners of C1..C5, split 20260903")
    D("noise control explains part, not all, of the FAR gap", 1 < nc["ratio"] < nc["lfw_ratio_pair"], "bool",
      "1 < noise ratio < LFW ratio (n=24)")
    D("max shortcut ratio orders of magnitude",
      math.floor(math.log10(max(LD["24"]["decode_fail_by_t"]["5"]["empirical"] / LD["24"]["decode_fail_by_t"]["5"]["binomial"],
                                LD["64"]["sampled"]["far_ratio"]))), "orders", "floor(log10(max ratio))")
    D("deploy authorization time ms", 159, "ms (assumption)", "stated assumption")
    bs = J("step5_bound_seeds.json")
    for key, row in bs["rows"].items():
        ps, n = key.split("/")
        for w, r in row.items():
            for f, v in r.items():
                add(f"plateau seed{ps} n={n} {w} {f}", v, "bits" if f != "parity" else "bits (parity)",
                    "pipeline/step5_bound_seeds.py", "MCV floor per projection seed", f"projection {ps}", bs["git"]["commit"])
    D("bound-positive both weightings all five seeds max n", bs["longest_positive_both_weightings_all_seeds"], "n",
      "step 5: longest n in 8..11 positive under probe and subject weighting for every seed")
    if ver:
        D("verify_results.py checks passed", ver["verify_py"]["pass"], "count", "checks/verify_results.py, run by step 7")
        D("verify_results.py checks failed", ver["verify_py"]["fail"], "count", "checks/verify_results.py, run by step 7")
        D("verify_short_lengths.py checks passed", ver["verify_canonical"]["pass"], "count", "checks/verify_short_lengths.py, run by step 7")
        D("verify_short_lengths.py checks failed", ver["verify_canonical"]["fail"], "count", "checks/verify_short_lengths.py, run by step 7")
    for q, v, u in (("NIST IR 8547 disallowed from year", 2035, "external (draft NIST IR 8547)"),
                    ("SP 800-90B sample requirement", 1000000, "external"),
                    ("ArcFace published LFW accuracy", 99.83, "external (InsightFace model zoo, buffalo_l)")):
        add(q, v, u, "-", "external source, not computed", "-", "-")

    # ------------------------------------------------------------------ write
    keys = {r["quantity"] for r in rows}
    dup = len(rows) - len(keys)
    assert dup == 0, f"{dup} duplicate quantity names"
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "units", "script", "function", "seeds", "commit"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"manifest.csv: {len(rows)} quantities")


def s1_top5(out):
    """Share of all probes held by the five subjects with the most probes."""
    import numpy as np
    from pipeline.provenance import EMB_NPZ
    e = np.load(EMB_NPZ, allow_pickle=True)["embeddings"].item()
    pc = np.sort([len(v) - 5 for v in e.values() if len(v) >= 6])[::-1]
    return 100 * float(pc[:5].sum() / pc.sum())


if __name__ == "__main__":
    main()
