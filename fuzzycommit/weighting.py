#!/usr/bin/env python3
"""
The three weightings (probe, capped at 10 probes per subject, subject),
probe-count tertiles, and the five-seed replication.
"""

import time

import numpy as np

from .common import (
    bch_parity, bch_true_t, select_capacity,
    ENROLLMENT_SIZE, DECODE_FAILURE_TARGET, PROJ_SEEDS,
    N_VALUES_SEED_SENSITIVITY, PROBE_CAP,
    load_and_center, get_W,
    build_subject_info, enroll_select, probe_hds,
    compute_bch_m, log,
)
from .bit_errors import compute_sweep_row
from .entropy import compute_eta, compute_eta_tertile
from .bootstrap import run_bootstrap_seed_sensitivity
from .false_accept import compute_neff


def run_capped_and_weighted(centered, W, n=24):
    """
    Capped-at-10 and subject-weighted β variants at n=24.

    The entropy budget is N_eff, computed at run time from this selection
    rather than read from a frozen constant.
    """
    subjects = sorted(centered.keys())
    m = compute_bch_m(n)

    # Build per-subject data
    info = []
    per_subject = []
    for s in subjects:
        vecs = centered[s]
        if len(vecs) < ENROLLMENT_SIZE + 1:
            continue
        sel, tmpl = enroll_select(vecs[:ENROLLMENT_SIZE], W, n)
        hds = probe_hds(vecs[ENROLLMENT_SIZE:], W, sel, tmpl)
        info.append((s, sel, tmpl, hds))
        per_subject.append((s, len(hds), hds))

    # Entropy budget, derived here rather than frozen into a constant.
    entropy_bits = float(compute_neff(centered, W, n, info))

    # Probe-uniform β (baseline, for reference)
    all_hd = np.concatenate([x[2] for x in per_subject])
    beta_baseline = float(np.mean(all_hd / n))
    _, parity_baseline, te_baseline = select_capacity(all_hd, n)

    # --- Capped at PROBE_CAP ---
    capped_hds = []
    for s, np_, hds in per_subject:
        if np_ > PROBE_CAP:
            capped_hds.append(hds[:PROBE_CAP])
        else:
            capped_hds.append(hds)
    all_hd_capped = np.concatenate(capped_hds)
    beta_cap = float(np.mean(all_hd_capped / n))
    _, parity_cap, te_cap = select_capacity(all_hd_capped, n)
    k_cap = n - parity_cap
    residual_cap = entropy_bits - parity_cap

    # --- Subject-weighted β ---
    per_subj_means = np.array([float(np.mean(x[2] / n)) for x in per_subject])
    beta_sw = float(np.mean(per_subj_means))

    # Subject-weighted t
    te_sw, parity_sw = bch_true_t(m, n), bch_parity(m, n)
    for t_try in range(n + 1):
        ta = bch_true_t(m, t_try)
        per_subj_fail = np.array([float(np.mean(x[2] > ta)) for x in per_subject])
        sw_fail_rate = float(np.mean(per_subj_fail))
        if sw_fail_rate < DECODE_FAILURE_TARGET:
            te_sw, parity_sw = ta, bch_parity(m, t_try)
            break
    k_sw = n - parity_sw
    residual_sw = entropy_bits - parity_sw

    return {
        "baseline": {
            "beta": round(beta_baseline, 4),
            "t": te_baseline,
            "parity": parity_baseline,
            "k": n - parity_baseline,
            "residual": round(entropy_bits - parity_baseline, 1),
        },
        "capped": {
            "beta": round(beta_cap, 4),
            "t": te_cap,
            "parity": parity_cap,
            "k": k_cap,
            "residual": round(residual_cap, 1),
        },
        "subject_weighted": {
            "beta": round(beta_sw, 4),
            "t": te_sw,
            "parity": parity_sw,
            "k": k_sw,
            "residual": round(residual_sw, 1),
        },
        "entropy_bits": entropy_bits,
        "per_subject_data": per_subject,
    }


def run_tertiles(centered, embeddings, W, n=24):
    """
    Per-tertile envelope metrics at n=24.
    η_LW recomputed WITHIN each tertile on that tertile's subjects only.
    """
    subjects = sorted(centered.keys())

    # Build per-subject info
    info, sels = build_subject_info(centered, W, n, subjects)

    # Compute full-population η for reference
    eta_ub_full, eta_lw_full, _, _ = compute_eta_tertile(centered, W, sels, n)

    # Sort subjects by image count
    subj_img_counts = sorted(
        [(x[0], len(embeddings[x[0]])) for x in info],
        key=lambda x: x[1]
    )
    ns_total = len(subj_img_counts)
    cuts = [ns_total // 3, 2 * ns_total // 3, ns_total]

    tertile_names_list = [
        [x[0] for x in subj_img_counts[:cuts[0]]],
        [x[0] for x in subj_img_counts[cuts[0]:cuts[1]]],
        [x[0] for x in subj_img_counts[cuts[1]:]],
    ]
    tertile_labels = [
        f"T1 (fewest: {subj_img_counts[0][1]}-{subj_img_counts[cuts[0]-1][1]} imgs)",
        f"T2 (middle: {subj_img_counts[cuts[0]][1]}-{subj_img_counts[cuts[1]-1][1]} imgs)",
        f"T3 (most: {subj_img_counts[cuts[1]][1]}-{subj_img_counts[-1][1]} imgs)",
    ]

    results = []
    for label, t_names in zip(tertile_labels, tertile_names_list):
        t_set = set(t_names)
        t_info = [x for x in info if x[0] in t_set]
        t_sels = [(x[0], x[1]) for x in t_info]
        t_hd = np.concatenate([x[3] for x in t_info])

        t_beta = float(np.mean(t_hd / n))
        _, t_parity, t_te = select_capacity(t_hd, n)
        t_k = n - t_parity

        # η_LW within tertile
        t_centered = {s: centered[s] for s in t_names}
        t_eta_ub, t_eta_lw, t_n_samples, t_shrinkage = compute_eta_tertile(
            t_centered, W, t_sels, n
        )
        t_neff = float(compute_neff(t_centered, W, n, t_info))
        t_residual = t_neff - t_parity
        t_closes = t_k > 0 and t_residual > 0

        results.append({
            "label": label,
            "n_subjects": len(t_info),
            "n_probes": len(t_hd),
            "beta": round(t_beta, 4),
            "t": t_te,
            "parity": t_parity,
            "k": t_k,
            "eta_ub": round(t_eta_ub, 3),
            "eta_lw": round(t_eta_lw, 3),
            "shrinkage": round(t_shrinkage, 4),
            "neff": round(t_neff, 3),
            "residual": round(t_residual, 1),
            "residual_eta": round(t_eta_lw * n - t_parity, 1),
            "closes": t_closes,
            "n_samples": t_n_samples,
        })

    return results


def run_seed_sensitivity(n_values=None, proj_seeds=None):
    """
    5-seed replication.

    Seeds: {43, 101, 202, 303, 404}
    Bootstrap seed: proj_seed * 1000 + n
    """
    if n_values is None:
        n_values = N_VALUES_SEED_SENSITIVITY
    if proj_seeds is None:
        proj_seeds = PROJ_SEEDS

    t_start = time.time()
    log("Seed sensitivity")
    log("=" * 70)

    embeddings, centered = load_and_center()
    subjects = sorted(centered.keys())
    log(f"  Seeds: {proj_seeds}")
    log(f"  n-values: {n_values}")

    results = {}

    for proj_seed in proj_seeds:
        W = get_W(proj_seed)

        for n in n_values:
            m = compute_bch_m(n)
            log(f"  seed={proj_seed}, n={n}")

            info, sels = build_subject_info(centered, W, n, subjects)
            row = compute_sweep_row(info, n)

            # η_LW
            eta_ub, eta_lw, shrinkage = compute_eta(centered, W, sels, n)
            neff = float(compute_neff(centered, W, n, info))
            residual = neff - row["parity"]

            # Bootstrap
            pct_closed, boot_seed = run_bootstrap_seed_sensitivity(
                info, n, neff, proj_seed, centered=centered, W=W
            )

            closes = row["k"] > 0 and residual > 0

            results[(proj_seed, n)] = {
                "proj_seed": proj_seed,
                "n": n,
                "m": m,
                "beta": round(row["beta"], 4),
                "std_beta": round(row["std_beta"], 4),
                "dispersion": round(row["overdispersion"], 2),
                "t": row["t"],
                "decode_fail_pct": round(row["decode_fail_frac"] * 100, 2),
                "parity": row["parity"],
                "k": row["k"],
                "eta_ub": round(float(eta_ub), 3),
                "eta_lw": round(float(eta_lw), 3),
                "shrinkage": round(float(shrinkage), 4),
                "neff": round(neff, 3),
                "residual": round(float(residual), 1),
                "residual_eta": round(float(eta_lw) * n - row["parity"], 1),
                "pct_closed": round(pct_closed, 1),
                "closes": closes,
                "boot_seed": boot_seed,
            }

    meta = {"wall_time": round(time.time() - t_start, 1)}
    return results, meta
