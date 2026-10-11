#!/usr/bin/env python3
"""
Subject-level cluster bootstrap (1,000 replicates): resample subjects with
replacement, choose the code again, recompute N_eff, and count the replicates
whose residual stays positive ("closure").
"""

import numpy as np

from .false_accept import impostor_pair_stats, neff_from_stats
from .common import (
    select_capacity, GLOBAL_SEED,
    N_BOOTSTRAP,
)


def run_bootstrap(info, n, entropy_bits, centered=None, W=None, n_reps=None, seed=None):
    """
    Subject-level bootstrap: resample subjects with replacement,
    re-compute t, k, residual, and check closure.

    Seed convention:
        seed = GLOBAL_SEED + 500 + n

    Args:
        info: list of (name, sel, template, hd_array)
        n: bit-string length
        entropy_bits: entropy budget in BITS (N_eff), not a per-bit rate
        n_reps: number of bootstrap replicates (default: N_BOOTSTRAP)
        seed: RNG seed (default: GLOBAL_SEED + 500 + n)

    Returns:
        pct_closed: float, percentage of replicates where k>0 and residual>0
        boot_results: dict with additional stats
    """
    if n_reps is None:
        n_reps = N_BOOTSTRAP
    if seed is None:
        seed = GLOBAL_SEED + 500 + n

    rng = np.random.RandomState(seed)
    ns = len(info)
    stats = (impostor_pair_stats(centered, W, n, info)
             if centered is not None and W is not None else None)

    boot_closed = []
    boot_t = []
    boot_residual = []
    for _ in range(n_reps):
        idx = rng.choice(ns, ns, replace=True)
        hd_pool = np.concatenate([info[i][3] for i in idx])
        _, par, te = select_capacity(hd_pool, n)
        k = n - par
        # The entropy budget is resampled too: N_eff comes from impostor
        # comparisons, so a subject-level resample moves it. Holding it fixed
        # (the previous behaviour) left t as the only random quantity and
        # excluded all entropy uncertainty from every reported interval.
        rep_entropy = (neff_from_stats(*stats, np.bincount(idx, minlength=ns))
                       if stats is not None else entropy_bits)
        res = rep_entropy - par
        boot_closed.append(k > 0 and res > 0)
        boot_t.append(te)
        boot_residual.append(res)

    pct_closed = float(np.mean(boot_closed) * 100)

    return pct_closed, {
        "pct_closed": pct_closed,
        "seed": seed,
        "n_reps": n_reps,
        "t_values": np.array(boot_t),
        "residuals": np.array(boot_residual, dtype=float),
    }


def run_bootstrap_ablation_arm_a(info, n, entropy_bits, centered=None, W=None, n_reps=None, seed=None):
    """
    Bootstrap for ablation Arm A.
    Uses its own seed, GLOBAL_SEED + 3000 + n, not the sweep's 500 + n.
    The sweep bootstrap gives the closure figures in the paper; this one is
    only for the ablation check (checks/expected/ablation_results.csv).
    """
    if n_reps is None:
        n_reps = N_BOOTSTRAP
    if seed is None:
        seed = GLOBAL_SEED + 3000 + n

    rng = np.random.RandomState(seed)
    ns = len(info)
    stats = (impostor_pair_stats(centered, W, n, info)
             if centered is not None and W is not None else None)

    boot_closed = []
    for _ in range(n_reps):
        idx = rng.choice(ns, ns, replace=True)
        hd_pool = np.concatenate([info[i][3] for i in idx])
        _, par, te = select_capacity(hd_pool, n)
        k = n - par
        # The entropy budget is resampled too: N_eff comes from impostor
        # comparisons, so a subject-level resample moves it. Holding it fixed
        # (the previous behaviour) left t as the only random quantity and
        # excluded all entropy uncertainty from every reported interval.
        rep_entropy = (neff_from_stats(*stats, np.bincount(idx, minlength=ns))
                       if stats is not None else entropy_bits)
        res = rep_entropy - par
        boot_closed.append(k > 0 and res > 0)

    return float(np.mean(boot_closed) * 100)


def run_bootstrap_ablation_arm_b(info, n, entropy_bits, centered=None, W=None, n_reps=None, seed=None):
    """
    Bootstrap for ablation Arm B.
    Uses seed: GLOBAL_SEED + 4000 + n
    """
    if n_reps is None:
        n_reps = N_BOOTSTRAP
    if seed is None:
        seed = GLOBAL_SEED + 4000 + n

    rng = np.random.RandomState(seed)
    ns = len(info)
    stats = (impostor_pair_stats(centered, W, n, info)
             if centered is not None and W is not None else None)

    boot_closed = []
    for _ in range(n_reps):
        idx = rng.choice(ns, ns, replace=True)
        hd_pool = np.concatenate([info[i][3] for i in idx])
        _, par, te = select_capacity(hd_pool, n)
        k = n - par
        # The entropy budget is resampled too: N_eff comes from impostor
        # comparisons, so a subject-level resample moves it. Holding it fixed
        # (the previous behaviour) left t as the only random quantity and
        # excluded all entropy uncertainty from every reported interval.
        rep_entropy = (neff_from_stats(*stats, np.bincount(idx, minlength=ns))
                       if stats is not None else entropy_bits)
        res = rep_entropy - par
        boot_closed.append(k > 0 and res > 0)

    return float(np.mean(boot_closed) * 100)


def run_bootstrap_seed_sensitivity(info, n, entropy_bits, proj_seed,
                                   n_reps=None, centered=None, W=None):
    """
    Bootstrap for seed sensitivity analysis.
    Seed convention: proj_seed * 1000 + n
    """
    if n_reps is None:
        n_reps = N_BOOTSTRAP

    boot_seed = proj_seed * 1000 + n
    rng = np.random.RandomState(boot_seed)
    ns = len(info)
    stats = (impostor_pair_stats(centered, W, n, info)
             if centered is not None and W is not None else None)

    boot_closed = []
    for _ in range(n_reps):
        idx = rng.choice(ns, ns, replace=True)
        hd_pool = np.concatenate([info[i][3] for i in idx])
        _, par, te = select_capacity(hd_pool, n)
        k = n - par
        # The entropy budget is resampled too: N_eff comes from impostor
        # comparisons, so a subject-level resample moves it. Holding it fixed
        # (the previous behaviour) left t as the only random quantity and
        # excluded all entropy uncertainty from every reported interval.
        rep_entropy = (neff_from_stats(*stats, np.bincount(idx, minlength=ns))
                       if stats is not None else entropy_bits)
        res = rep_entropy - par
        boot_closed.append(k > 0 and res > 0)

    return float(np.mean(boot_closed) * 100), boot_seed
