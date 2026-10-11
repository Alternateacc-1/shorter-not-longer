#!/usr/bin/env python3
"""
Bit-error statistics for one bit-string length: the probe Hamming distances,
the mean bit-error rate (beta), and the least-parity code that meets the
decode-failure target. The selection itself lives in common.py.
"""

import numpy as np

from .common import (
    select_capacity, compute_bch_m,
)


def compute_beta_and_hds(info, n):
    """
    Compute β (mean BER) and aggregate Hamming distances from subject info.

    Args:
        info: list of (name, sel, template, hd_array) from build_subject_info
        n: bit-string length

    Returns:
        all_hd: concatenated Hamming distance array
        beta_val: float, mean BER
        std_beta: float, std of BER
        n_probes: int, total probe count
    """
    all_hd = np.concatenate([x[3] for x in info])
    n_probes = len(all_hd)
    beta_val = float(np.mean(all_hd / n))
    std_beta = float(np.std(all_hd / n))
    return all_hd, beta_val, std_beta, n_probes


def compute_sweep_row(info, n):
    """
    Compute a single n-sweep row (β, t, decode failure, parity, k).

    Returns dict with all scalar quantities.
    """
    all_hd, beta_val, std_beta, n_probes = compute_beta_and_hds(info, n)
    m = compute_bch_m(n)

    # Least-parity selection: inside a parity plateau the designed t and the
    # delivered capacity differ, and the code is billed for the parity either way.
    t_designed, parity, te = select_capacity(all_hd, n)
    decode_fail = float(np.sum(all_hd > te) / n_probes)
    k = n - parity

    binom_std = float(np.sqrt(beta_val * (1 - beta_val) / n))
    overdispersion = std_beta / binom_std if binom_std > 0 else float("inf")

    return {
        "n": n,
        "m": m,
        "beta": beta_val,
        "std_beta": std_beta,
        "binom_std": binom_std,
        "overdispersion": overdispersion,
        "t": te,
        "t_designed": t_designed,
        "decode_fail_frac": decode_fail,
        "parity": parity,
        "k": k,
        "n_subjects": len(info),
        "n_probes": n_probes,
        "all_hd": all_hd,
    }


def per_subject_ber(info, n):
    """Per-subject mean BER array, shape (n_subjects,)."""
    return np.array([float(np.mean(hds / n)) for _, _, _, hds in info])
