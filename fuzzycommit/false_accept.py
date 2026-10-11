#!/usr/bin/env python3
"""
False-accept rate: 10^6 sampled impostor trials, the full impostor set
(every enrollee against every image of the other 310 subjects, averaged per
ordered subject pair), the independent-bits rate, and N_eff from the variance
of the impostor Hamming distances.
"""

import time

import numpy as np
from scipy.special import comb

from .common import (
    GLOBAL_SEED, N_IMPOSTOR, ENROLLMENT_SIZE,
    enroll_select, clopper_pearson, log,
)


def run_sampled_far(info, centered, W, n, t, n_trials=None, seed=None):
    """
    Run N sampled impostor trials.
    Seed convention: GLOBAL_SEED + 300 + n

    Args:
        info: list of (name, sel, template, hd_array)
        centered: dict of centered vectors
        W: projection matrix
        n: bit-string length
        t: BCH correction capacity
        n_trials: number of trials (default: N_IMPOSTOR = 1e6)
        seed: RNG seed (default: GLOBAL_SEED + 300 + n)

    Returns:
        accepts: int, number of false accepts
        far: float, FAR = accepts / n_trials
        ci_lo: float, Clopper-Pearson lower bound
        ci_hi: float, Clopper-Pearson upper bound
    """
    if n_trials is None:
        n_trials = N_IMPOSTOR
    if seed is None:
        seed = GLOBAL_SEED + 300 + n

    rng = np.random.RandomState(seed)
    ns = len(info)
    accept = 0

    t_start = time.time()
    for trial in range(n_trials):
        ia, ib = rng.choice(ns, 2, replace=False)
        _, sel_a, tmpl_a, _ = info[ia]
        name_b = info[ib][0]
        vecs_b = centered[name_b]
        img_b = vecs_b[rng.randint(len(vecs_b))]
        bits_b = ((img_b @ W)[sel_a] > 0).astype(int)
        hd = int(np.sum(tmpl_a != bits_b))
        if hd <= t:
            accept += 1

        if (trial + 1) % 250_000 == 0:
            elapsed = time.time() - t_start
            log(f"    {trial+1:>9,} / {n_trials:,}  "
                f"accepts={accept}  ({elapsed:.1f}s)")

    far = accept / n_trials
    ci_lo, ci_hi = clopper_pearson(accept, n_trials)
    return accept, far, ci_lo, ci_hi


def compute_analytical_far(n, t):
    """Analytical FAR assuming independent bits: Σ_{i≤t} C(n,i) / 2^n."""
    return float(sum(comb(n, i, exact=True) for i in range(t + 1)) / 2**n)


def run_exhaustive_far(centered, W, n, t, subjects=None):
    """
    Exhaustive impostor enumeration over all ordered subject pairs.

    Args:
        centered: dict of centered vectors
        W: projection matrix
        n: bit-string length
        t: BCH correction capacity at this n
        subjects: sorted list of subject names

    Returns:
        dict with p_bar, p_std, n_eff, far_pair_uniform, far_comparison_uniform, etc.
    """
    if subjects is None:
        subjects = sorted(centered.keys())

    n_subjects = len(subjects)
    n_ordered_pairs = n_subjects * (n_subjects - 1)

    # Build flat arrays with subject-to-image mapping
    all_vecs_list = []
    subject_of = []
    subject_ranges = []
    idx = 0
    for si, s in enumerate(subjects):
        vecs = centered[s]
        n_imgs = len(vecs)
        all_vecs_list.append(vecs)
        subject_of.extend([si] * n_imgs)
        subject_ranges.append((idx, idx + n_imgs))
        idx += n_imgs

    all_vecs_flat = np.concatenate(all_vecs_list, axis=0)
    subject_of = np.array(subject_of, dtype=np.int32)
    imgs_per_subj = np.array([end - start for start, end in subject_ranges])

    # Precompute all sign-quantized bits
    all_bits = (all_vecs_flat @ W > 0).astype(np.int8)

    # Build per-subject templates
    templates = []
    for si, s in enumerate(subjects):
        enroll_vecs = centered[s][:ENROLLMENT_SIZE]
        sel, tmpl = enroll_select(enroll_vecs, W, n)
        templates.append((sel, tmpl))

    # Exhaustive impostor computation
    all_impostor_hds_list = []
    pair_far_sum = 0.0
    total_accepts = 0
    total_comparisons = 0

    t_lap = time.time()
    for si in range(n_subjects):
        sel_i, tmpl_i = templates[si]
        bits_sel = all_bits[:, sel_i]
        hd_all = np.sum(bits_sel != tmpl_i[np.newaxis, :], axis=1)

        mask_imp = subject_of != si
        impostor_hds_i = hd_all[mask_imp]
        all_impostor_hds_list.append(impostor_hds_i)

        accepts_vec = (hd_all <= t).astype(np.float64)
        accept_per_subj = np.bincount(
            subject_of, weights=accepts_vec, minlength=n_subjects
        )
        accept_per_subj[si] = 0.0
        pair_rates = accept_per_subj / imgs_per_subj
        pair_rates[si] = 0.0
        pair_far_sum += np.sum(pair_rates)

        total_accepts += int(np.sum(impostor_hds_i <= t))
        total_comparisons += len(impostor_hds_i)

        if (si + 1) % 100 == 0:
            log(f"    {si+1}/{n_subjects} template subjects "
                f"({time.time()-t_lap:.1f}s)")

    all_impostor_hds = np.concatenate(all_impostor_hds_list)

    # Distribution statistics
    frac_hds = all_impostor_hds.astype(np.float64) / n
    p_bar = float(np.mean(frac_hds))
    p_std = float(np.std(frac_hds))
    p_var = float(np.var(frac_hds))

    # N_eff
    n_eff = p_bar * (1 - p_bar) / p_var if p_var > 0 else float("inf")
    n_eff_ratio = n_eff / n
    n_eff_round = round(n_eff)

    # FAR
    far_pair_uniform = pair_far_sum / n_ordered_pairs
    far_comparison_uniform = total_accepts / total_comparisons

    # Analytical
    far_anal_n = compute_analytical_far(n, t)
    if n_eff_round > 0 and n_eff_round >= t:
        far_anal_neff = float(
            sum(comb(n_eff_round, i, exact=True)
                for i in range(t + 1)) / 2**n_eff_round
        )
    else:
        far_anal_neff = float("nan")

    return {
        "n": n,
        "t": t,
        "n_comparisons": total_comparisons,
        "total_accepts": total_accepts,
        "p_bar": round(p_bar, 6),
        "p_std": round(p_std, 6),
        "n_eff": round(n_eff, 4),
        "n_eff_ratio": round(n_eff_ratio, 4),
        "n_eff_round": n_eff_round,
        "far_pair_uniform": far_pair_uniform,
        "far_comparison_uniform": far_comparison_uniform,
        "far_anal_n": far_anal_n,
        "far_anal_neff": far_anal_neff,
    }


# ---------------------------------------------------------------
# Effective degrees of freedom (Daugman) -- the reported entropy figure
# ---------------------------------------------------------------
def impostor_pair_stats(centered, W, n, info):
    """Per ordered subject pair (i, j), i != j: the sum, sum of squares and
    count of NORMALISED impostor Hamming distances between subject i's
    template and each of subject j's images. Diagonals are zeroed.

    Returned as matrices so a bootstrap replicate can be aggregated as a
    quadratic form in the resampled subject counts, without re-enumerating.
    """
    names = [x[0] for x in info]
    sizes = np.array([len(centered[s]) for s in names])
    flat = np.concatenate([centered[s] for s in names], axis=0)
    owner = np.concatenate([[i] * sizes[i] for i in range(len(names))])
    bits = (flat @ W > 0).astype(np.int8)
    K = len(names)
    S1 = np.zeros((K, K)); S2 = np.zeros((K, K))
    CNT = np.tile(sizes.astype(float), (K, 1))
    for i in range(K):
        _, sel, tmpl, _ = info[i]
        hd = np.sum(bits[:, sel] != tmpl[np.newaxis, :], axis=1) / n
        S1[i] = np.bincount(owner, weights=hd, minlength=K)
        S2[i] = np.bincount(owner, weights=hd * hd, minlength=K)
    # Zeroing the diagonal IS the same-subject exclusion, and it is keyed to
    # ORIGINAL subject index, not to a position in any resample. A bootstrap
    # replicate enters as a multiplicity vector c indexed the same way, so the
    # quadratic form c @ M @ c contributes c_i * c_i * M[i,i] = 0 for every
    # subject drawn more than once. Excluding by resample position instead
    # would let two copies of one subject form an "impostor" pair at Hamming
    # distance ~beta*n, inflating sigma^2 and deflating N_eff by ~0.43 bits at
    # n=16 and ~0.85 bits at n=24, in every replicate.
    np.fill_diagonal(S1, 0.0); np.fill_diagonal(S2, 0.0); np.fill_diagonal(CNT, 0.0)
    return S1, S2, CNT


def neff_from_stats(S1, S2, CNT, counts):
    """N_eff = p(1-p)/sigma^2 over the impostor pairs selected by `counts`,
    a per-subject multiplicity vector (all ones for the full sample)."""
    c = np.asarray(counts, dtype=float)
    # Guard the same-subject exclusion against a future refactor.
    assert not (np.any(np.diag(S1)) or np.any(np.diag(S2)) or np.any(np.diag(CNT))), (
        "impostor pair statistics have a non-zero diagonal: same-subject pairs "
        "would be counted as impostors for any duplicated subject")
    tot = float(c @ CNT @ c)
    if tot <= 0:
        return float("nan")
    p = float(c @ S1 @ c) / tot
    var = float(c @ S2 @ c) / tot - p * p
    if var <= 0:
        return float("nan")
    return p * (1.0 - p) / var


def compute_neff(centered, W, n, info):
    """Effective degrees of freedom of the bit-string, from the variance of the
    impostor Hamming distribution over all ordered subject pairs.

    This replaces eta_LW * n as the reported entropy budget. Both are
    second-order statistics -- Var(HD) depends only on first and second
    moments -- but N_eff is estimated from 1,681,750 impostor comparisons
    rather than 311 templates, where Ledoit-Wolf shrinkage reaches 0.65 and
    the log-det penalty collapses to 0.005. It is effective degrees of
    freedom in Daugman's sense, NOT a min-entropy bound.
    """
    S1, S2, CNT = impostor_pair_stats(centered, W, n, info)
    return neff_from_stats(S1, S2, CNT, np.ones(len(info)))
