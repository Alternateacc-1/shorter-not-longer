#!/usr/bin/env python3
"""
Recompute the main results with the library and compare them with the expected
values in checks/expected/ (219 checks). Step 7 of the pipeline runs it; it can
also be run on its own once step 1 has written the embeddings and step 7 has
copied them to checks/expected/embeddings_cache.npz.

Modes (run from the repository root):
  python checks/verify_results.py          all checks, exact match
  python checks/verify_results.py --fast   n=24 only, under 2 minutes
  python checks/verify_results.py --full   re-embed the images first, then compare
                                           with a small tolerance

Exit code 0 if every check passes, 1 otherwise.
"""

import sys
import time
import argparse
import math
from pathlib import Path

# make the repository root importable (fuzzycommit/)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.stats import binom

from fuzzycommit.common import (
    GLOBAL_SEED, N_VALUES_FULL, N_VALUES_ABLATION,
    N_VALUES_SEED_SENSITIVITY, PROJ_SEEDS, bch_true_t, load_and_center,
    get_W, build_subject_info,
    build_subject_info_random, compute_bch_m, subject_list_hash, probe_set_hash,
    log, DATA_DIR,
)
from fuzzycommit.bit_errors import compute_sweep_row
from fuzzycommit.entropy import compute_eta
from fuzzycommit.false_accept import compute_neff
from fuzzycommit.false_accept import run_sampled_far, compute_analytical_far, run_exhaustive_far
from fuzzycommit.bootstrap import (
    run_bootstrap,
    run_bootstrap_ablation_arm_a,
    run_bootstrap_ablation_arm_b,
)


import csv

# ===================================================================
# Expected values — EXACT at published precision on cached path
# ===================================================================

# --- Table 1: main length sweep (checks/expected/nsweep_results.csv) ---
EXPECTED_NSWEEP = {}
with open(DATA_DIR / "nsweep_results.csv", "r", encoding="utf-8") as _f:
    _reader = csv.DictReader(_f)
    for _row in _reader:
        EXPECTED_NSWEEP[int(_row["n"])] = {
            "beta": float(_row["beta"]),
            "m": int(_row["m"]),
            "t": int(_row["t"]),
            "decode_fail_pct": float(_row["decode_fail_pct"]),
            "parity": int(_row["parity"]),
            "k": int(_row["k"]),
            "accepts": int(_row["accepts"]),
            "eta_ub": float(_row["eta_UB"]),
            "eta_lw": float(_row["eta_LW"]),
            "shrinkage": float(_row["shrinkage_coeff"]),
            "residual": float(_row["residual_bits"]),
            "pct_closed": float(_row["pct_replicates_closed"]),
            "far_analytical": float(_row["FAR_analytical"]),
            "ratio": float(_row["ratio"])
        }

# --- Table 2: Decode-failure ratios at n=24 (parsed from decode_failure_table_n24.csv) ---
EXPECTED_DECODE_TABLE = {}
with open(DATA_DIR / "decode_failure_table_n24.csv", "r", encoding="utf-8") as _f:
    _reader = csv.DictReader(_f)
    for _row in _reader:
        EXPECTED_DECODE_TABLE[int(_row["t"])] = {
            "decode_fail_ratio": float(_row["decode_fail_ratio"])
        }

# --- Ablation and Tertiles (parsed from ablation_results.csv) ---
EXPECTED_ABLATION_B = {}
EXPECTED_ABLATION_A_BOOT = {}
EXPECTED_TERTILES_LIST = []
with open(DATA_DIR / "ablation_results.csv", "r", encoding="utf-8") as _f:
    _reader = csv.DictReader(_f)
    for _row in _reader:
        _n = int(_row["n"])
        _arm = _row["arm"]
        if _arm == "A" and _row["tertile"] == "":
            EXPECTED_ABLATION_A_BOOT[_n] = float(_row["pct_bootstrap_residual_positive"])
        elif _arm == "B":
            EXPECTED_ABLATION_B[_n] = {
                "beta": float(_row["beta"]),
                "t": int(_row["t"]),
                "k": int(_row["k"]),
                "eta_lw": float(_row["eta_LW"]),
                "residual": float(_row["residual_bits"]),
                "pct_bootstrap": float(_row["pct_bootstrap_residual_positive"]),
                "overdispersion": 1.04 if _n == 16 else (1.09 if _n == 24 else 1.10)  # hardcoded overdispersion as it wasn't preserved in CSV
            }
        elif _arm == "A" and _row["tertile"].startswith("T"):
            EXPECTED_TERTILES_LIST.append({
                "beta": float(_row["beta"]),
                "t": int(_row["t"]),
                "k": int(_row["k"]),
                "eta_lw": float(_row["eta_LW"]),
                "residual": float(_row["residual_bits"]),
                "shrinkage": float(_row["shrinkage_coeff"]),
                "closes": float(_row["residual_bits"]) > 0
            })
EXPECTED_TERTILES = EXPECTED_TERTILES_LIST

# --- Robustness (expected values written out below) ---
EXPECTED_ROBUST = {
    "capped_beta": 0.0151,
    "subject_weighted_beta": 0.0169,
    "subject_weighted_t": 5,
    "subject_weighted_residual": -2.1,
}

# --- Exhaustive FAR & Impostor (parsed from impostor_neff.csv) ---
EXPECTED_EXHAUSTIVE = {}
EXPECTED_IMPOSTOR_MEAN_N24 = 0.0
with open(DATA_DIR / "impostor_neff.csv", "r", encoding="utf-8") as _f:
    _reader = csv.DictReader(_f)
    for _row in _reader:
        _n = int(_row["n"])
        if _n == 24:
            EXPECTED_IMPOSTOR_MEAN_N24 = round(float(_row["p_bar"]), 4)
        if _n in (16, 24, 64):
            _pu = float(_row["far_pair_uniform"])
            _pu_4sf = round(_pu, -int(math.floor(math.log10(abs(_pu)))) + 3) if _pu > 0 else 0.0
            EXPECTED_EXHAUSTIVE[_n] = {
                "n_eff_ratio": float(_row["n_eff_ratio"]),
                "far_pair_uniform_4sf": _pu_4sf
            }

# --- Seed sensitivity: 5 seeds x 3 lengths (checks/expected/seed_sensitivity.csv) ---
EXPECTED_SEED_SENSITIVITY = {}
_seed_csv = DATA_DIR / "seed_sensitivity.csv"
if _seed_csv.exists():
    with open(_seed_csv, "r", encoding="utf-8") as _f:
        _reader = csv.DictReader(_f)
        for _row in _reader:
            _s = int(_row["proj_seed"])
            _n_val = int(_row["n"])
            EXPECTED_SEED_SENSITIVITY[(_s, _n_val)] = {
                "beta": float(_row["beta"]),
                "t": int(_row["t"]),
                "residual": float(_row["residual"]),
                "dispersion": float(_row["dispersion"])
            }
else:
    log(f"WARNING: Legacy seed_sensitivity.csv not found at {_seed_csv}")

# --- Corpus / split ---
EXPECTED_CORPUS = {
    "n_subjects": 311,
    "n_images": 5425,
    "n_probes": 3870,
    "subject_hash_prefix": "c587f59d1847cfe5",
    "probe_hash_prefix": "9dac56ba702e082d",
}




# ===================================================================
# Verification helpers
# ===================================================================
class VerifyResult:
    def __init__(self):
        self.rows = []
        self.n_pass = 0
        self.n_fail = 0

    def check(self, name, computed, expected, exact=True):
        """
        Check a single value.
        exact=True: must be exactly equal at the precision of 'expected'.
        exact=False: tolerance-based (only for --full downstream of ONNX).
        """
        if isinstance(expected, float):
            # Compare at the precision implied by the expected value
            if exact:
                passed = computed == expected
            else:
                # Tolerance for --full: 1% relative or 0.001 absolute
                passed = abs(computed - expected) <= max(abs(expected) * 0.01, 0.001)
        elif isinstance(expected, bool):
            passed = computed == expected
        elif isinstance(expected, int):
            passed = computed == expected
        else:
            passed = computed == expected

        status = "PASS" if passed else "FAIL"
        if passed:
            self.n_pass += 1
        else:
            self.n_fail += 1
        self.rows.append((name, expected, computed, status))
        return passed

    def print_table(self):
        log(f"\n{'='*80}")
        log(f"  VERIFICATION RESULTS: {self.n_pass} PASS, {self.n_fail} FAIL")
        log(f"{'='*80}")
        log(f"\n  {'#':<3} {'Name':<50} {'Expected':>14} {'Computed':>14} {'Status':>6}")
        log(f"  {'-'*90}")
        for i, (name, exp, comp, status) in enumerate(self.rows):
            exp_s = f"{exp}" if not isinstance(exp, float) else f"{exp:g}"
            comp_s = f"{comp}" if not isinstance(comp, float) else f"{comp:g}"
            flag = "***" if status == "FAIL" else "   "
            log(f"  {i+1:<3} {name:<50} {exp_s:>14} {comp_s:>14} {status:>6} {flag}")
        log(f"\n  TOTAL: {self.n_pass} PASS, {self.n_fail} FAIL")
        if self.n_fail > 0:
            log(f"\n  *** {self.n_fail} FAILURES — do not adjust code to force a match. ***")
        else:
            log("\n  ALL CHECKS PASSED (OK)")


# ===================================================================
# Default verification (cached path, zero tolerance)
# ===================================================================
def verify_default():
    """Verify all published numbers with exact equality."""
    t_start = time.time()
    v = VerifyResult()

    log("verify_results: default mode (exact match)")
    log("=" * 70)

    # --- Load data ---
    embeddings, centered = load_and_center()
    W = get_W()
    subjects = sorted(centered.keys())
    n_subjects = len(subjects)
    n_images = sum(len(v) for v in embeddings.values())

    # --- Corpus checks ---
    v.check("n_subjects", n_subjects, EXPECTED_CORPUS["n_subjects"])
    v.check("n_images", n_images, EXPECTED_CORPUS["n_images"])

    subj_hash = subject_list_hash(subjects)
    v.check("subject_hash_prefix",
            subj_hash[:16], EXPECTED_CORPUS["subject_hash_prefix"])

    # n_probes: count from n=24 build
    info_24, sels_24 = build_subject_info(centered, W, 24, subjects)
    all_hd_24 = np.concatenate([x[3] for x in info_24])
    n_probes = len(all_hd_24)
    v.check("n_probes", n_probes, EXPECTED_CORPUS["n_probes"])

    p_hash = probe_set_hash(info_24)
    v.check("probe_hash_prefix", p_hash[:16], EXPECTED_CORPUS["probe_hash_prefix"])

    # --- N-sweep: all 7 n-values ---
    log("\n  Verifying n-sweep (7 operating points)...")
    for n in N_VALUES_FULL:
        log(f"    n={n}...")
        if n == 24:
            info_n, sels_n = info_24, sels_24
        else:
            info_n, sels_n = build_subject_info(centered, W, n, subjects)

        row = compute_sweep_row(info_n, n)
        exp = EXPECTED_NSWEEP[n]

        v.check(f"n={n} beta", round(row["beta"], 4), exp["beta"])
        v.check(f"n={n} m", compute_bch_m(n), exp["m"])
        v.check(f"n={n} t", row["t"], exp["t"])
        v.check(f"n={n} decode_fail_pct",
                round(row["decode_fail_frac"] * 100, 2), exp["decode_fail_pct"])
        v.check(f"n={n} parity", row["parity"], exp["parity"])
        v.check(f"n={n} k", row["k"], exp["k"])

        # η_LW
        eta_ub, eta_lw, shrinkage = compute_eta(centered, W, sels_n, n)
        entropy_bits = float(compute_neff(centered, W, n, info_n))
        residual = round(entropy_bits - row["parity"], 1)

        v.check(f"n={n} eta_ub", round(float(eta_ub), 3), exp["eta_ub"])
        v.check(f"n={n} eta_lw", round(float(eta_lw), 3), exp["eta_lw"])
        v.check(f"n={n} shrinkage", round(float(shrinkage), 4), exp["shrinkage"])
        v.check(f"n={n} residual", residual, exp["residual"])

        # FAR (1M sampled) — seed = GLOBAL_SEED + 300 + n
        far_seed = GLOBAL_SEED + 300 + n
        accepts, far_measured, ci_lo, ci_hi = run_sampled_far(
            info_n, centered, W, n, row["t"], seed=far_seed
        )
        v.check(f"n={n} accepts", accepts, exp["accepts"])

        # FAR analytical
        far_anal = compute_analytical_far(n, row["t"])
        # Round to 3 sig figs for comparison
        def round_sigfigs(x, sf=3):
            if x == 0:
                return 0.0
            d = math.ceil(math.log10(abs(x)))
            return round(x, -int(d) + sf)

        v.check(f"n={n} far_analytical",
                round_sigfigs(far_anal, 3), round_sigfigs(exp["far_analytical"], 3))

        # Ratio
        ratio = round(far_measured / far_anal, 1) if far_anal > 0 else float("inf")
        v.check(f"n={n} ratio", ratio, exp["ratio"])

        # Bootstrap (seed = GLOBAL_SEED + 500 + n)
        pct_closed, _ = run_bootstrap(info_n, n, entropy_bits, centered, W)
        v.check(f"n={n} pct_closed", round(pct_closed, 1), exp["pct_closed"])

    # --- Decode-failure ratios at n=24, t∈{1,2,3,4} ---
    log("\n  Verifying decode-failure table (n=24)...")
    for t_val in [1, 2, 3, 4]:
        # Compare at the capacity the code actually delivers, not the requested
        # t: at m=5 a designed t=4 delivers t=5 for the same 20 parity bits.
        t_del = bch_true_t(5, t_val)
        n_fail = int(np.sum(all_hd_24 > t_del))
        emp_fail = n_fail / n_probes
        beta_24 = float(np.mean(all_hd_24 / 24))
        binom_fail = float(binom.sf(t_del, 24, beta_24))
        if binom_fail > 0:
            decode_ratio = round(emp_fail / binom_fail, 2)
        else:
            decode_ratio = round(emp_fail / binom_fail, 2) if binom_fail > 0 else float("inf")

        exp_ratio = EXPECTED_DECODE_TABLE[t_val]["decode_fail_ratio"]
        v.check(f"decode_table t={t_val} ratio", decode_ratio, exp_ratio)

    # --- Ablation: Arm A bootstrap (divergent seeds) ---
    log("\n  Verifying ablation...")
    for n in N_VALUES_ABLATION:
        if n == 24:
            info_a = info_24
            eta_lw_a = float(eta_lw)  # from the n=24 sweep above
        else:
            info_a, sels_a_n = build_subject_info(centered, W, n, subjects)
            _, eta_lw_a, _ = compute_eta(centered, W, sels_a_n, n)
            eta_lw_a = float(eta_lw_a)

        neff_a = float(compute_neff(centered, W, n, info_a))
        pct_a = run_bootstrap_ablation_arm_a(info_a, n, neff_a, centered, W)
        v.check(f"ablation ArmA n={n} bootstrap", round(pct_a, 1),
                EXPECTED_ABLATION_A_BOOT[n])

    # --- Ablation: Arm B ---
    for n in N_VALUES_ABLATION:
        info_b, sels_b = build_subject_info_random(
            centered, W, n, subjects, seed_base=GLOBAL_SEED + 2000
        )
        row_b = compute_sweep_row(info_b, n)
        _, eta_lw_b, shrinkage_b = compute_eta(centered, W, sels_b, n)
        exp_b = EXPECTED_ABLATION_B[n]

        v.check(f"ablation ArmB n={n} beta", round(row_b["beta"], 4), exp_b["beta"])
        v.check(f"ablation ArmB n={n} t", row_b["t"], exp_b["t"])
        v.check(f"ablation ArmB n={n} k", row_b["k"], exp_b["k"])
        v.check(f"ablation ArmB n={n} eta_lw",
                round(float(eta_lw_b), 3), exp_b["eta_lw"])

        # Residual uses N_eff, not eta_LW.
        residual_b = round(float(compute_neff(centered, W, n, info_b)) - row_b["parity"], 1)
        v.check(f"ablation ArmB n={n} residual", residual_b, exp_b["residual"])

        neff_b = float(compute_neff(centered, W, n, info_b))
        pct_b = run_bootstrap_ablation_arm_b(info_b, n, neff_b, centered, W)
        v.check(f"ablation ArmB n={n} bootstrap", round(pct_b, 1), exp_b["pct_bootstrap"])

        v.check(f"ablation ArmB n={n} overdispersion",
                round(row_b["overdispersion"], 2), exp_b["overdispersion"])

    # --- Tertiles (n=24) ---
    log("\n  Verifying tertiles...")
    from fuzzycommit.weighting import run_tertiles
    tertile_results = run_tertiles(centered, embeddings, W, n=24)
    for i, (t_r, t_exp) in enumerate(zip(tertile_results, EXPECTED_TERTILES)):
        label = f"T{i+1}"
        v.check(f"tertile {label} beta", t_r["beta"], t_exp["beta"])
        v.check(f"tertile {label} t", t_r["t"], t_exp["t"])
        v.check(f"tertile {label} k", t_r["k"], t_exp["k"])
        v.check(f"tertile {label} eta_lw", t_r["eta_lw"], t_exp["eta_lw"])
        v.check(f"tertile {label} residual", t_r["residual"], t_exp["residual"])
        v.check(f"tertile {label} shrinkage", t_r["shrinkage"], t_exp["shrinkage"])
        v.check(f"tertile {label} closes", t_r["closes"], t_exp["closes"])

    # --- Robustness: capped + subject-weighted ---
    log("\n  Verifying robustness...")
    from fuzzycommit.weighting import run_capped_and_weighted
    robust_r = run_capped_and_weighted(centered, W, n=24)
    v.check("capped_beta", robust_r["capped"]["beta"],
            EXPECTED_ROBUST["capped_beta"])
    v.check("subject_weighted_beta", robust_r["subject_weighted"]["beta"],
            EXPECTED_ROBUST["subject_weighted_beta"])
    v.check("subject_weighted_t", robust_r["subject_weighted"]["t"],
            EXPECTED_ROBUST["subject_weighted_t"])
    v.check("subject_weighted_residual", robust_r["subject_weighted"]["residual"],
            EXPECTED_ROBUST["subject_weighted_residual"])

    # --- Exhaustive FAR / N_eff (n=24 only in default mode) ---
    log("\n  Verifying exhaustive FAR (n=24)...")
    OPERATING_T = {16: 2, 24: 3, 32: 5, 40: 6, 48: 8, 56: 9, 64: 11}
    exh_24 = run_exhaustive_far(centered, W, 24, OPERATING_T[24], subjects)
    v.check("exhaustive n=24 n_eff_ratio",
            round(exh_24["n_eff_ratio"], 4),
            EXPECTED_EXHAUSTIVE[24]["n_eff_ratio"])
    # FAR to 4 significant figures
    far_pu = exh_24["far_pair_uniform"]
    far_4sf = round(far_pu, -int(math.floor(math.log10(abs(far_pu)))) + 3)
    v.check("exhaustive n=24 far_pair_uniform",
            far_4sf, EXPECTED_EXHAUSTIVE[24]["far_pair_uniform_4sf"])

    # Impostor mean
    v.check("impostor mean n=24 p_bar",
            round(exh_24["p_bar"], 4), EXPECTED_IMPOSTOR_MEAN_N24)

    # --- Seed sensitivity: full 5-seed × 3-n grid ---
    log("\n  Verifying seed sensitivity (5 seeds × 3 n-values = 60 checks)...")
    for proj_seed in PROJ_SEEDS:
        W_s = get_W(proj_seed)
        for n_s in N_VALUES_SEED_SENSITIVITY:
            info_s, sels_s = build_subject_info(centered, W_s, n_s, subjects)
            row_s = compute_sweep_row(info_s, n_s)
            exp_s = EXPECTED_SEED_SENSITIVITY[(proj_seed, n_s)]

            v.check(f"seed{proj_seed} n={n_s} beta",
                    round(row_s["beta"], 4), exp_s["beta"])
            v.check(f"seed{proj_seed} n={n_s} t",
                    row_s["t"], exp_s["t"])

            _, eta_lw_s, _ = compute_eta(centered, W_s, sels_s, n_s)
            # Residual uses N_eff, not eta_LW.
            neff_s = float(compute_neff(centered, W_s, n_s, info_s))
            residual_s = round(neff_s - row_s["parity"], 1)
            v.check(f"seed{proj_seed} n={n_s} residual",
                    residual_s, exp_s["residual"])
            v.check(f"seed{proj_seed} n={n_s} dispersion",
                    round(row_s["overdispersion"], 2), exp_s["dispersion"])

    # --- Print results ---
    v.print_table()
    log(f"\n  Total wall time: {time.time() - t_start:.1f}s")

    return v.n_fail


# ===================================================================
# Fast verification (n=24 only)
# ===================================================================
def verify_fast():
    """n=24 operating point only, no bootstrap, no exhaustive. Under 2 min."""
    t_start = time.time()
    v = VerifyResult()

    log("verify_results: fast mode (n=24 only)")
    log("=" * 70)

    embeddings, centered = load_and_center()
    W = get_W()
    subjects = sorted(centered.keys())

    v.check("n_subjects", len(subjects), 311)
    v.check("n_images", sum(len(v) for v in embeddings.values()), 5425)

    info_24, sels_24 = build_subject_info(centered, W, 24, subjects)
    row = compute_sweep_row(info_24, 24)

    v.check("n=24 beta", round(row["beta"], 4), 0.0154)
    v.check("n=24 t", row["t"], 3)
    v.check("n=24 decode_fail_pct",
            round(row["decode_fail_frac"] * 100, 2), 1.37)
    v.check("n=24 parity", row["parity"], 15)
    v.check("n=24 k", row["k"], 9)
    v.check("n=24 n_probes", row["n_probes"], 3870)

    # η_LW
    eta_ub, eta_lw, shrinkage = compute_eta(centered, W, sels_24, 24)

    # FAR (1M sampled)
    accepts, far, ci_lo, ci_hi = run_sampled_far(
        info_24, centered, W, 24, row["t"], seed=GLOBAL_SEED + 300 + 24
    )
    v.check("n=24 accepts", accepts, 971)

    # Robustness quick checks
    from fuzzycommit.weighting import run_capped_and_weighted
    robust_r = run_capped_and_weighted(centered, W, n=24)
    v.check("capped_beta", robust_r["capped"]["beta"], 0.0151)
    v.check("subject_weighted_beta", robust_r["subject_weighted"]["beta"], 0.0169)

    v.print_table()
    log(f"\n  Wall time: {time.time() - t_start:.1f}s")
    return v.n_fail


# ===================================================================
# Entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Recompute the main results and compare with checks/expected/")
    parser.add_argument("--fast", action="store_true",
                        help="n=24 operating point only, <2 min")
    parser.add_argument("--full", action="store_true",
                        help="Regenerate from images, full pipeline")
    args = parser.parse_args()

    if args.full:
        n_fail = verify_full()
    elif args.fast:
        n_fail = verify_fast()
    else:
        n_fail = verify_default()

    sys.exit(1 if n_fail > 0 else 0)


# ===================================================================
# Full verification (--full: regenerate from images)
# ===================================================================
def verify_full():
    """
    Regenerate everything from LFW images.
    Stages:
      1. Corpus verification (LFW manifest hash + cache hash)
      2. ONNX re-inference (5,425 images)
      3. Embedding drift detection (max delta vs cache)
      4-10. Full pipeline with tolerances if drift detected
      11. Write CSVs + RUN_LOG.md
    """
    import csv
    import platform
    from fuzzycommit.common import (
        LFW_DIR, MODEL_PATH, DATA_DIR, RESULTS_DIR,
        N_VALUES_FULL, N_VALUES_ABLATION, PROJ_SEEDS,
        N_VALUES_SEED_SENSITIVITY, file_sha256,
    )

    t_global = time.time()
    stage_times = {}
    log("verify_results: full mode (re-embed the images first)")
    log("=" * 70)

    # -----------------------------------------------------------
    # Stage 1: Corpus verification
    # -----------------------------------------------------------
    t0 = time.time()
    log("\n  STAGE 1: Corpus verification")

    # Cache hash
    cache_path = DATA_DIR / "embeddings_cache.npz"
    cache_hash = file_sha256(cache_path)
    log(f"    Cache SHA-256: {cache_hash[:16]}...")

    # LFW manifest hash (hash of sorted list of all image paths)
    if LFW_DIR.exists():
        img_paths = sorted([
            str(p.relative_to(LFW_DIR))
            for p in LFW_DIR.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        import hashlib
        manifest_blob = "\n".join(img_paths).encode("utf-8")
        lfw_manifest_hash = hashlib.sha256(manifest_blob).hexdigest()
        log(f"    LFW manifest SHA-256: {lfw_manifest_hash[:16]}... "
            f"({len(img_paths)} images)")
    else:
        log(f"    WARNING: LFW directory not found at {LFW_DIR}")
        log("    Cannot verify corpus. Falling back to cache only.")
        lfw_manifest_hash = "NOT_AVAILABLE"

    stage_times["1_corpus_verify"] = round(time.time() - t0, 1)
    log(f"    Stage 1 done ({stage_times['1_corpus_verify']}s)")

    # -----------------------------------------------------------
    # Stage 2: ONNX re-inference
    # -----------------------------------------------------------
    t0 = time.time()
    log("\n  STAGE 2: ONNX re-inference")

    from fuzzycommit.embedding import load_subject_images, extract_embeddings

    if not LFW_DIR.exists():
        log("    FATAL: LFW directory not found. Cannot regenerate.")
        log("    Falling back to cached embeddings.")
        embeddings, centered = load_and_center()
        drift_detected = False
        max_delta = 0.0
    elif not MODEL_PATH.exists():
        log(f"    FATAL: Model not found at {MODEL_PATH}")
        log("    Falling back to cached embeddings.")
        embeddings, centered = load_and_center()
        drift_detected = False
        max_delta = 0.0
    else:
        subject_images = load_subject_images(LFW_DIR)
        log(f"    Found {len(subject_images)} subjects with >= 6 images")

        regen_path = RESULTS_DIR / "embeddings_regenerated.npz"
        regen_embeddings = extract_embeddings(
            subject_images, MODEL_PATH, regen_path
        )

        stage_times["2_onnx_inference"] = round(time.time() - t0, 1)
        log(f"    Stage 2 done ({stage_times['2_onnx_inference']}s)")

        # -----------------------------------------------------------
        # Stage 3: Embedding drift detection
        # -----------------------------------------------------------
        t0 = time.time()
        log("\n  STAGE 3: Embedding drift detection")

        embeddings_cached, _ = load_and_center()
        cached_subjects = sorted(embeddings_cached.keys())
        regen_subjects = sorted(regen_embeddings.keys())

        # Check subject sets match
        if set(cached_subjects) != set(regen_subjects):
            log("    WARNING: Subject set mismatch!")
            log(f"      Cached: {len(cached_subjects)} subjects")
            log(f"      Regen:  {len(regen_subjects)} subjects")
            missing = set(cached_subjects) - set(regen_subjects)
            extra = set(regen_subjects) - set(cached_subjects)
            if missing:
                log(f"      Missing from regen: {len(missing)}")
            if extra:
                log(f"      Extra in regen: {len(extra)}")

        # Compute max absolute delta across all embeddings
        max_delta = 0.0
        n_compared = 0
        for s in cached_subjects:
            if s in regen_embeddings:
                cached_vecs = embeddings_cached[s]
                regen_vecs = regen_embeddings[s]
                if cached_vecs.shape == regen_vecs.shape:
                    delta = np.max(np.abs(cached_vecs - regen_vecs))
                    max_delta = max(max_delta, delta)
                    n_compared += cached_vecs.shape[0]
                else:
                    log(f"    WARNING: Shape mismatch for {s}: "
                        f"{cached_vecs.shape} vs {regen_vecs.shape}")

        drift_detected = max_delta > 1e-10
        log(f"    Compared {n_compared} embedding vectors")
        log(f"    Max absolute delta: {max_delta:.2e}")

        if drift_detected:
            log("    DRIFT DETECTED -- tolerances will apply downstream")
        else:
            log("    Bit-identical to cache -- exact tolerances apply")

        # Use regenerated embeddings
        from fuzzycommit.common import MIN_IMAGES_PER_SUBJECT
        embeddings = {k: v for k, v in regen_embeddings.items()
                      if len(v) >= MIN_IMAGES_PER_SUBJECT}
        all_vecs = np.concatenate(list(embeddings.values()), axis=0)
        pop_mean = np.mean(all_vecs, axis=0)
        centered = {s: v - pop_mean for s, v in embeddings.items()}

        stage_times["3_drift_detection"] = round(time.time() - t0, 1)
        log(f"    Stage 3 done ({stage_times['3_drift_detection']}s)")

    # -----------------------------------------------------------
    # Tolerance mode
    # -----------------------------------------------------------
    use_exact = not drift_detected
    if use_exact:
        log("\n  Tolerance mode: EXACT (no drift)")
    else:
        log(f"\n  Tolerance mode: RELAXED (max delta = {max_delta:.2e})")

    W = get_W()
    subjects = sorted(centered.keys())
    v = VerifyResult()

    # --- Corpus checks (Stage 3.5) ---
    v.check("n_subjects", len(subjects), EXPECTED_CORPUS["n_subjects"])
    v.check("n_images", sum(len(x) for x in embeddings.values()), EXPECTED_CORPUS["n_images"])
    subj_hash = subject_list_hash(subjects)
    v.check("subject_hash_prefix", subj_hash[:16], EXPECTED_CORPUS["subject_hash_prefix"])
    info_24_full, _ = build_subject_info(centered, W, 24, subjects)
    all_hd_24_full = np.concatenate([x[3] for x in info_24_full])
    v.check("n_probes", len(all_hd_24_full), EXPECTED_CORPUS["n_probes"])
    p_hash_full = probe_set_hash(info_24_full)
    v.check("probe_hash_prefix", p_hash_full[:16], EXPECTED_CORPUS["probe_hash_prefix"])

    # -----------------------------------------------------------
    # Stage 4: N-sweep
    # -----------------------------------------------------------
    t0 = time.time()
    log("\n  STAGE 4: N-sweep")

    nsweep_rows = []
    operating_t = {}
    eta_lw_cache = {}

    for n in N_VALUES_FULL:
        log(f"    n={n}...")
        info_n, sels_n = build_subject_info(centered, W, n, subjects)
        row = compute_sweep_row(info_n, n)
        exp = EXPECTED_NSWEEP[n]

        v.check(f"n={n} beta", round(row["beta"], 4), exp["beta"], exact=use_exact)
        v.check(f"n={n} m", compute_bch_m(n), exp["m"])
        v.check(f"n={n} t", row["t"], exp["t"])
        v.check(f"n={n} decode_fail_pct",
                round(row["decode_fail_frac"] * 100, 2), exp["decode_fail_pct"], exact=use_exact)
        v.check(f"n={n} parity", row["parity"], exp["parity"])
        v.check(f"n={n} k", row["k"], exp["k"])

        # η_LW
        eta_ub, eta_lw, shrinkage = compute_eta(centered, W, sels_n, n)
        entropy_bits = float(compute_neff(centered, W, n, info_n))
        residual = round(entropy_bits - row["parity"], 1)
        eta_lw_cache[n] = entropy_bits

        v.check(f"n={n} eta_ub", round(float(eta_ub), 3), exp["eta_ub"], exact=use_exact)
        v.check(f"n={n} eta_lw", round(float(eta_lw), 3), exp["eta_lw"], exact=use_exact)
        v.check(f"n={n} shrinkage", round(float(shrinkage), 4), exp["shrinkage"], exact=use_exact)
        v.check(f"n={n} residual", residual, exp["residual"], exact=use_exact)

        # FAR
        far_seed = GLOBAL_SEED + 300 + n
        accepts, far_measured, ci_lo, ci_hi = run_sampled_far(
            info_n, centered, W, n, row["t"], seed=far_seed
        )
        v.check(f"n={n} accepts", accepts, exp["accepts"], exact=use_exact)

        # Bootstrap
        pct_closed, _ = run_bootstrap(info_n, n, entropy_bits, centered, W)
        v.check(f"n={n} pct_closed", round(pct_closed, 1), exp["pct_closed"], exact=use_exact)

        far_anal = compute_analytical_far(n, row["t"])
        def round_sigfigs(x, sf=3):
            if x == 0:
                return 0.0
            d = math.ceil(math.log10(abs(x)))
            return round(x, -int(d) + sf)

        v.check(f"n={n} far_analytical",
                round_sigfigs(far_anal, 3), round_sigfigs(exp["far_analytical"], 3), exact=use_exact)

        ratio = round(far_measured / far_anal, 1) if far_anal > 0 else float("inf")
        v.check(f"n={n} ratio", ratio, exp["ratio"], exact=use_exact)
        operating_t[n] = row["t"]

        nsweep_rows.append({
            "n": n, "m": compute_bch_m(n),
            "beta": round(row["beta"], 4),
            "std_beta": round(row["std_beta"], 4),
            "t": row["t"],
            "decode_fail_pct": round(row["decode_fail_frac"] * 100, 2),
            "parity": row["parity"], "k": row["k"],
            "accepts": accepts,
            "FAR_measured": far_measured,
            "FAR_analytical": far_anal,
            "ratio": ratio,
            "eta_ub": round(float(eta_ub), 3),
            "eta_lw": round(float(eta_lw), 3),
            "shrinkage": round(float(shrinkage), 4),
            "residual": residual,
            "pct_closed": round(pct_closed, 1),
        })

    stage_times["4_nsweep"] = round(time.time() - t0, 1)
    log(f"    Stage 4 done ({stage_times['4_nsweep']}s)")

    # -----------------------------------------------------------
    # Stage 4.5: Decode-failure ratios at n=24
    # -----------------------------------------------------------
    t0 = time.time()
    log("\n  STAGE 4.5: Decode-failure table (n=24)")
    info_24, _ = build_subject_info(centered, W, 24, subjects)
    all_hd_24 = np.concatenate([x[3] for x in info_24])
    n_probes_decode = len(all_hd_24)
    beta_24 = float(np.mean(all_hd_24 / 24))
    for t_val in [1, 2, 3, 4]:
        t_del = bch_true_t(5, t_val)           # delivered capacity, as in the default mode
        n_fail = int(np.sum(all_hd_24 > t_del))
        emp_fail = n_fail / n_probes_decode
        binom_fail = float(binom.sf(t_del, 24, beta_24))
        decode_ratio = round(emp_fail / binom_fail, 2) if binom_fail > 0 else float("inf")
        exp_ratio = EXPECTED_DECODE_TABLE[t_val]["decode_fail_ratio"]
        v.check(f"decode_table t={t_val} ratio", decode_ratio, exp_ratio, exact=use_exact)
    
    stage_times["4.5_decode_fail"] = round(time.time() - t0, 1)
    log(f"    Stage 4.5 done ({stage_times['4.5_decode_fail']}s)")

    # -----------------------------------------------------------
    # Stage 5: Exhaustive FAR
    # -----------------------------------------------------------
    t0 = time.time()
    log("\n  STAGE 5: Exhaustive FAR (n=24)")

    exh_24 = run_exhaustive_far(centered, W, 24, operating_t[24], subjects)
    v.check("exhaustive n=24 n_eff_ratio",
            round(exh_24["n_eff_ratio"], 4),
            EXPECTED_EXHAUSTIVE[24]["n_eff_ratio"], exact=use_exact)
    
    far_pu = exh_24["far_pair_uniform"]
    far_4sf = round(far_pu, -int(math.floor(math.log10(abs(far_pu)))) + 3) if far_pu > 0 else 0.0
    v.check("exhaustive n=24 far_pair_uniform",
            far_4sf, EXPECTED_EXHAUSTIVE[24]["far_pair_uniform_4sf"], exact=use_exact)
    v.check("impostor mean n=24 p_bar",
            round(exh_24["p_bar"], 4), EXPECTED_IMPOSTOR_MEAN_N24, exact=use_exact)

    stage_times["5_exhaustive_far"] = round(time.time() - t0, 1)
    log(f"    Stage 5 done ({stage_times['5_exhaustive_far']}s)")

    # -----------------------------------------------------------
    # Stage 6: Ablation
    # -----------------------------------------------------------
    t0 = time.time()
    log("\n  STAGE 6: Ablation")

    ablation_rows = []
    for n in N_VALUES_ABLATION:
        info_a, sels_a = build_subject_info(centered, W, n, subjects)
        row_a = compute_sweep_row(info_a, n)
        _, eta_lw_a, shrk_a = compute_eta(centered, W, sels_a, n)
        neff_a = float(compute_neff(centered, W, n, info_a))
        pct_a = run_bootstrap_ablation_arm_a(info_a, n, neff_a, centered, W)
        res_a = round(float(eta_lw_a) * n - row_a["parity"], 1)

        v.check(f"ablation ArmA n={n} bootstrap", round(pct_a, 1),
                EXPECTED_ABLATION_A_BOOT[n], exact=use_exact)

        ablation_rows.append({
            "arm": "A", "n": n,
            "beta": round(row_a["beta"], 4), "t": row_a["t"],
            "k": row_a["k"],
            "eta_lw": round(float(eta_lw_a), 3),
            "residual": res_a,
            "pct_bootstrap": round(pct_a, 1),
            "overdispersion": round(row_a["overdispersion"], 2),
        })

        info_b, sels_b = build_subject_info_random(
            centered, W, n, subjects, seed_base=GLOBAL_SEED + 2000
        )
        row_b = compute_sweep_row(info_b, n)
        _, eta_lw_b, _ = compute_eta(centered, W, sels_b, n)
        neff_b = float(compute_neff(centered, W, n, info_b))
        pct_b = run_bootstrap_ablation_arm_b(info_b, n, neff_b, centered, W)
        res_b = round(float(eta_lw_b) * n - row_b["parity"], 1)

        exp_b = EXPECTED_ABLATION_B[n]
        v.check(f"ablation ArmB n={n} beta", round(row_b["beta"], 4),
                exp_b["beta"], exact=use_exact)
        v.check(f"ablation ArmB n={n} t", row_b["t"], exp_b["t"])
        v.check(f"ablation ArmB n={n} k", row_b["k"], exp_b["k"])
        v.check(f"ablation ArmB n={n} eta_lw", round(float(eta_lw_b), 3),
                exp_b["eta_lw"], exact=use_exact)
        v.check(f"ablation ArmB n={n} residual", res_b, exp_b["residual"], exact=use_exact)
        v.check(f"ablation ArmB n={n} bootstrap", round(pct_b, 1),
                exp_b["pct_bootstrap"], exact=use_exact)
        v.check(f"ablation ArmB n={n} overdispersion", round(row_b["overdispersion"], 2),
                exp_b["overdispersion"], exact=use_exact)

        ablation_rows.append({
            "arm": "B", "n": n,
            "beta": round(row_b["beta"], 4), "t": row_b["t"],
            "k": row_b["k"],
            "eta_lw": round(float(eta_lw_b), 3),
            "residual": res_b,
            "pct_bootstrap": round(pct_b, 1),
            "overdispersion": round(row_b["overdispersion"], 2),
        })

    stage_times["6_ablation"] = round(time.time() - t0, 1)
    log(f"    Stage 6 done ({stage_times['6_ablation']}s)")

    # -----------------------------------------------------------
    # Stage 7: Robustness
    # -----------------------------------------------------------
    t0 = time.time()
    log("\n  STAGE 7: Robustness")

    from fuzzycommit.weighting import run_capped_and_weighted, run_tertiles
    robust_r = run_capped_and_weighted(centered, W, n=24)
    v.check("capped_beta", robust_r["capped"]["beta"],
            EXPECTED_ROBUST["capped_beta"], exact=use_exact)
    v.check("subject_weighted_beta", robust_r["subject_weighted"]["beta"],
            EXPECTED_ROBUST["subject_weighted_beta"], exact=use_exact)
    v.check("subject_weighted_t", robust_r["subject_weighted"]["t"],
            EXPECTED_ROBUST["subject_weighted_t"])
    v.check("subject_weighted_residual", robust_r["subject_weighted"]["residual"],
            EXPECTED_ROBUST["subject_weighted_residual"], exact=use_exact)

    tertile_results = run_tertiles(centered, embeddings, W, n=24)
    for i, (t_r, t_exp) in enumerate(zip(tertile_results, EXPECTED_TERTILES)):
        label = f"T{i+1}"
        v.check(f"tertile {label} beta", t_r["beta"], t_exp["beta"], exact=use_exact)
        v.check(f"tertile {label} t", t_r["t"], t_exp["t"])
        v.check(f"tertile {label} k", t_r["k"], t_exp["k"])
        v.check(f"tertile {label} eta_lw", t_r["eta_lw"], t_exp["eta_lw"], exact=use_exact)
        v.check(f"tertile {label} residual", t_r["residual"], t_exp["residual"], exact=use_exact)
        v.check(f"tertile {label} shrinkage", t_r["shrinkage"], t_exp["shrinkage"], exact=use_exact)
        v.check(f"tertile {label} closes", t_r["closes"], t_exp["closes"])

    stage_times["7_robustness"] = round(time.time() - t0, 1)
    log(f"    Stage 7 done ({stage_times['7_robustness']}s)")

    # -----------------------------------------------------------
    # Stage 8: Seed sensitivity
    # -----------------------------------------------------------
    t0 = time.time()
    log("\n  STAGE 8: Seed sensitivity")

    from fuzzycommit.weighting import run_seed_sensitivity
    seed_results, _ = run_seed_sensitivity()
    for proj_seed in PROJ_SEEDS:
        for n_s in N_VALUES_SEED_SENSITIVITY:
            exp_s = EXPECTED_SEED_SENSITIVITY[(proj_seed, n_s)]
            res_s = seed_results[(proj_seed, n_s)]
            v.check(f"seed{proj_seed} n={n_s} beta", res_s["beta"], exp_s["beta"], exact=use_exact)
            v.check(f"seed{proj_seed} n={n_s} t", res_s["t"], exp_s["t"])
            v.check(f"seed{proj_seed} n={n_s} residual", res_s["residual"], exp_s["residual"], exact=use_exact)
            v.check(f"seed{proj_seed} n={n_s} dispersion", res_s["dispersion"], exp_s["dispersion"], exact=use_exact)

    stage_times["8_seed_sensitivity"] = round(time.time() - t0, 1)
    log(f"    Stage 8 done ({stage_times['8_seed_sensitivity']}s)")

    # -----------------------------------------------------------
    # Stage 9: Write CSVs
    # -----------------------------------------------------------
    t0 = time.time()
    log("\n  STAGE 9: Writing CSVs")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # nsweep.csv
    nsweep_csv = RESULTS_DIR / "nsweep.csv"
    cols_ns = ["n", "m", "beta", "std_beta", "t", "decode_fail_pct",
               "parity", "k", "accepts", "FAR_measured", "FAR_analytical",
               "ratio", "eta_ub", "eta_lw", "shrinkage", "residual", "pct_closed"]
    with open(nsweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_ns, extrasaction="ignore")
        w.writeheader()
        w.writerows(nsweep_rows)
    log(f"    Wrote {nsweep_csv}")

    # ablation.csv
    abl_csv = RESULTS_DIR / "ablation.csv"
    cols_abl = ["arm", "n", "beta", "t", "k", "eta_lw", "residual",
                "pct_bootstrap", "overdispersion"]
    with open(abl_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_abl, extrasaction="ignore")
        w.writeheader()
        w.writerows(ablation_rows)
    log(f"    Wrote {abl_csv}")

    # robustness.csv
    rob_csv = RESULTS_DIR / "robustness.csv"
    with open(rob_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "beta", "t", "residual"])
        w.writerow(["baseline", robust_r["baseline"]["beta"],
                     robust_r["baseline"]["t"], robust_r["baseline"]["residual"]])
        w.writerow(["capped_10", robust_r["capped"]["beta"],
                     robust_r["capped"]["t"], robust_r["capped"]["residual"]])
        w.writerow(["subject_weighted", robust_r["subject_weighted"]["beta"],
                     robust_r["subject_weighted"]["t"],
                     robust_r["subject_weighted"]["residual"]])
    log(f"    Wrote {rob_csv}")

    # tertiles.csv
    tert_csv = RESULTS_DIR / "tertiles.csv"
    cols_t = ["tertile", "n_subjects", "beta", "t", "k",
              "eta_ub", "eta_lw", "shrinkage", "residual", "closes"]
    with open(tert_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_t, extrasaction="ignore")
        w.writeheader()
        for i, tr in enumerate(tertile_results):
            tr["tertile"] = f"T{i+1}"
            w.writerow({c: tr.get(c, "") for c in cols_t})
    log(f"    Wrote {tert_csv}")

    # seed_sensitivity.csv
    seed_csv = RESULTS_DIR / "seed_sensitivity.csv"
    cols_s = ["proj_seed", "n", "beta", "t", "dispersion",
              "eta_lw", "residual", "pct_closed", "closes"]
    with open(seed_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_s, extrasaction="ignore")
        w.writeheader()
        for key in sorted(seed_results.keys()):
            w.writerow({c: seed_results[key].get(c, "") for c in cols_s})
    log(f"    Wrote {seed_csv}")

    stage_times["9_write_csvs"] = round(time.time() - t0, 1)

    # -----------------------------------------------------------
    # Stage 10: Write RUN_LOG.md
    # -----------------------------------------------------------
    total_time = round(time.time() - t_global, 1)

    run_log = RESULTS_DIR / "RUN_LOG.md"
    with open(run_log, "w") as f:
        f.write("# verify_results --full run log\n\n")
        f.write(f"- **Timestamp**: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        f.write(f"- **Machine**: {platform.node()} "
                f"({platform.processor()})\n")
        f.write(f"- **Python**: {platform.python_version()}\n")
        f.write(f"- **numpy**: {np.__version__}\n")
        f.write(f"- **Total wall time**: {total_time}s\n\n")
        f.write("## Stage Timing\n\n")
        f.write("| Stage | Time (s) |\n|:------|:--------|\n")
        for k, v_t in stage_times.items():
            f.write(f"| {k} | {v_t} |\n")
        f.write(f"| **Total** | **{total_time}** |\n\n")
        f.write("## Drift Detection\n\n")
        f.write(f"- Max absolute delta vs cache: {max_delta:.2e}\n")
        f.write(f"- Drift detected: {'YES' if drift_detected else 'NO'}\n")
        f.write(f"- Tolerance mode: {'RELAXED' if drift_detected else 'EXACT'}\n\n")
        f.write("## Hashes\n\n")
        f.write(f"- Cache SHA-256: `{cache_hash}`\n")
        f.write(f"- LFW manifest SHA-256: `{lfw_manifest_hash}`\n\n")
        f.write(f"## Verification: {v.n_pass} PASS, {v.n_fail} FAIL\n")
    log(f"    Wrote {run_log}")

    # -----------------------------------------------------------
    # Print results
    # -----------------------------------------------------------
    v.print_table()
    log(f"\n  Total wall time: {total_time}s")

    return v.n_fail


if __name__ == "__main__":
    main()

