#!/usr/bin/env python3
"""
Per-bit entropy estimates of the enrolled bit-strings (eta_UB and the
Ledoit-Wolf block estimate eta_LW). The paper reports N_eff instead; these
estimates are kept because the check scripts test them.
"""

import numpy as np
from sklearn.covariance import LedoitWolf

from .common import ENROLLMENT_SIZE

# A marginal this close to 0 or 1, or an empirical correlation matrix this
# close to singular, drives the log-det penalty to +inf and makes the
# estimator return a large negative "entropy". Unreachable on the LFW corpus
# (minimum per-bit entropy observed is 0.887) but a landmine on another.
EPS_MARGINAL = 1e-6
EPS_EIGENVALUE = 1e-8


class DegenerateSourceError(ValueError):
    """The bit source is degenerate, so the log-det entropy correction is not
    defined. Raised instead of returning a negative entropy."""


def _check_not_degenerate(bs, where):
    """Reject constant bits and exactly-dependent bits BEFORE shrinkage."""
    p = np.mean(bs, axis=0)
    bad = np.where((p <= EPS_MARGINAL) | (p >= 1.0 - EPS_MARGINAL))[0]
    if len(bad):
        raise DegenerateSourceError(
            f"{where}: {len(bad)} bit(s) are constant across the sample "
            f"(marginal within {EPS_MARGINAL} of 0 or 1) at column index "
            f"{bad[:5].tolist()}. Min-entropy is 0 for those bits and the "
            f"log-det correction is undefined; the estimator would return a "
            f"large negative value."
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(bs, rowvar=False)
    if not np.all(np.isfinite(R)):
        raise DegenerateSourceError(f"{where}: empirical correlation matrix is not finite.")
    lam = float(np.min(np.linalg.eigvalsh(R)))
    if lam < EPS_EIGENVALUE:
        raise DegenerateSourceError(
            f"{where}: empirical correlation matrix is singular before "
            f"shrinkage (min eigenvalue {lam:.3e} < {EPS_EIGENVALUE:g}); bits "
            f"are exactly dependent. The log-det correction diverges and the "
            f"estimator would return a large negative value."
        )



def enrolled_bitstrings(centered, W):
    """One enrolled bit-string per subject (311 rows), all 512 candidate bits.

    A commitment publishes one bit-string per enrollee, so entropy is estimated
    over the 311 enrolled templates, not over all 5,425 images: those are 311
    correlated clusters, and one subject with 530 images would dominate.
    """
    subjects = sorted(centered.keys())
    rows = [
        (centered[s][:ENROLLMENT_SIZE].mean(axis=0) @ W > 0).astype(float)
        for s in subjects
        if len(centered[s]) >= ENROLLMENT_SIZE + 1
    ]
    return np.array(rows)


def compute_eta(centered, W, sels, n):
    """
    Compute η_UB, η_LW, and shrinkage coefficient.

    Args:
        centered: dict {subject: centered vectors}
        W: (512, N_PROJECTIONS) projection matrix
        sels: list of (subject_name, selected_indices) tuples
        n: bit-string length

    Returns:
        eta_ub: float, mean upper-bound entropy
        eta_lw: float, mean Ledoit-Wolf corrected entropy
        shrinkage: float, mean shrinkage coefficient
    """
    all_bits = enrolled_bitstrings(centered, W)

    ub_list, lw_list, shrinkage_list = [], [], []
    for _, sel in sels:
        bs = all_bits[:, sel]
        _check_not_degenerate(bs, "compute_eta")

        # η_UB: mean per-bit min-entropy
        p = np.mean(bs, axis=0)
        eta_ub = np.mean(-np.log2(np.maximum(p, 1 - p)))
        ub_list.append(eta_ub)

        # Ledoit-Wolf shrinkage
        lw = LedoitWolf(assume_centered=False)
        lw.fit(bs)
        C = lw.covariance_
        d = np.sqrt(np.diag(C))
        d[d < 1e-12] = 1e-12
        shrunk_corr = C / np.outer(d, d)
        shrinkage_coef = lw.shrinkage_

        eigvals = np.linalg.eigvalsh(shrunk_corr)
        eigvals = np.maximum(eigvals, 1e-15)
        log_det_penalty = -0.5 * np.sum(np.log2(eigvals)) / n
        eta_lw = eta_ub - log_det_penalty

        lw_list.append(eta_lw)
        shrinkage_list.append(shrinkage_coef)

    return np.mean(ub_list), np.mean(lw_list), np.mean(shrinkage_list)


def compute_eta_tertile(centered_tertile, W, sels, n):
    """
    η_UB and η_LW recomputed WITHIN a tertile (tertile's own images only).

    Args:
        centered_tertile: dict of {subject: centered vectors} restricted to tertile
        W: projection matrix
        sels: list of (subject_name, selected_indices) for tertile subjects
        n: bit-string length

    Returns:
        eta_ub, eta_lw, n_samples, shrinkage
    """
    all_bits = enrolled_bitstrings(centered_tertile, W)
    n_samples = all_bits.shape[0]

    ub_list, lw_list, shrinkage_list = [], [], []
    for _, sel in sels:
        bs = all_bits[:, sel]
        _check_not_degenerate(bs, "compute_eta_tertile")
        p = np.mean(bs, axis=0)
        eta_ub = np.mean(-np.log2(np.maximum(p, 1 - p)))
        ub_list.append(eta_ub)

        lw = LedoitWolf(assume_centered=False)
        lw.fit(bs)
        C = lw.covariance_
        d = np.sqrt(np.diag(C))
        d[d < 1e-12] = 1e-12
        R = C / np.outer(d, d)
        shrinkage_list.append(lw.shrinkage_)

        ev = np.maximum(np.linalg.eigvalsh(R), 1e-15)
        penalty = -0.5 * np.sum(np.log2(ev)) / n
        lw_list.append(eta_ub - penalty)

    avg_ub = float(np.mean(ub_list))
    avg_lw = float(np.mean(lw_list))
    avg_shrinkage = float(np.mean(shrinkage_list)) if shrinkage_list else float("nan")
    return avg_ub, avg_lw, n_samples, avg_shrinkage
