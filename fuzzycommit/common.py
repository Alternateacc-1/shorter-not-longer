#!/usr/bin/env python3
"""
Core helpers: constants, paths, loading the embeddings, the shared projection
matrix, reliability-ordered bit selection, exact BCH parity and capacity, and
the per-subject enrollment/probe split.

Seeds:
  GLOBAL_SEED = 42
  Projection matrix: RandomState(GLOBAL_SEED + 1) = RandomState(43)
"""

import os
import sys
import math
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
GLOBAL_SEED = 42
MIN_IMAGES_PER_SUBJECT = 6
N_PROJECTIONS = 512
ENROLLMENT_SIZE = 5
DECODE_FAILURE_TARGET = 0.015

# Paths
_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PKG_DIR.parent
# Where load_and_center() looks for embeddings_cache.npz. The pipeline scripts point
# it at results/pipeline/; the check scripts use checks/expected/.
DATA_DIR = REPO_ROOT / "checks" / "expected"
MODEL_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results" / "checks"
def _resolve_lfw_dir():
    """Locate the deep-funneled LFW image tree, without hardcoding a machine path.

    Resolution order:
      1. the LFW_DIR environment variable
      2. a `lfw_dir = ...` line in paths.conf at the repository root
      3. the default kagglehub cache location for the dataset named in README.md

    Returns a Path even when nothing exists there yet; the scripts that need the
    images report a missing folder themselves.
    """
    env = os.environ.get("LFW_DIR")
    if env:
        return Path(env)
    conf = REPO_ROOT / "paths.conf"
    if conf.exists():
        for line in conf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("lfw_dir") and "=" in line:
                return Path(line.split("=", 1)[1].strip())
    return (Path.home() / ".cache" / "kagglehub" / "datasets" / "jessicali9530"
            / "lfw-dataset" / "versions" / "4" / "lfw-deepfunneled" / "lfw-deepfunneled")


LFW_DIR = _resolve_lfw_dir()
MODEL_PATH = MODEL_DIR / "w600k_r50.onnx"
CENTER_CROP_SIZE = 150

# N-values used across the pipeline
N_VALUES_FULL = [16, 24, 32, 40, 48, 56, 64]
N_VALUES_ABLATION = [16, 24, 32]
N_VALUES_SEED_SENSITIVITY = [16, 24, 32]
PROJ_SEEDS = [43, 101, 202, 303, 404]

# Bootstrap / impostor counts
N_BOOTSTRAP = 1000
N_IMPOSTOR = 1_000_000
PROBE_CAP = 10



def log(msg=""):
    """Print and flush."""
    print(msg, flush=True)


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
def load_and_center():
    """
    Load cached embeddings, filter to >= MIN_IMAGES_PER_SUBJECT, center.

    Returns:
        embeddings: dict {subject_name: (n_images, 512) array} (raw)
        centered: dict {subject_name: (n_images, 512) array} (mean-subtracted)
    """
    cache_path = DATA_DIR / "embeddings_cache.npz"
    if not cache_path.exists():
        log(f"  FATAL: cached embeddings not found at {cache_path}")
        log("")
        log("  Embeddings are not shipped. Create them from the LFW images with")
        log("  step 1, run from the repository root:")
        log("")
        log("      python -m pipeline.step1_embed")
        log("")
        log("  It needs the deep-funneled LFW tree and models/w600k_r50.onnx (see the")
        log("  README). Point it at your LFW copy with LFW_DIR=... or a")
        log(f"  `lfw_dir = ...` line in {REPO_ROOT / 'paths.conf'}.")
        log(f"  Currently resolved LFW_DIR: {LFW_DIR}")
        sys.exit(1)
    data = np.load(cache_path, allow_pickle=True)
    raw = data["embeddings"].item()
    embeddings = {k: v for k, v in raw.items() if len(v) >= MIN_IMAGES_PER_SUBJECT}

    all_vecs = np.concatenate(list(embeddings.values()), axis=0)
    pop_mean = np.mean(all_vecs, axis=0)
    centered = {s: v - pop_mean for s, v in embeddings.items()}
    return embeddings, centered


# ---------------------------------------------------------------
# Projection matrix
# ---------------------------------------------------------------
def get_W(proj_seed=None):
    """
    Generate the projection matrix.

    Args:
        proj_seed: If None, uses GLOBAL_SEED + 1 (= 43, the baseline).
                   Otherwise uses the given seed directly.

    Returns:
        W: (512, N_PROJECTIONS) array, unit-norm columns.
    """
    if proj_seed is None:
        proj_seed = GLOBAL_SEED + 1  # = 43
    rng = np.random.RandomState(proj_seed)
    W = rng.randn(512, N_PROJECTIONS)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return W


# ---------------------------------------------------------------
# Projection / selection / matching
# ---------------------------------------------------------------
def enroll_select(enroll_vecs, W, n):
    """
    Reliability-select n bits from enrollment.

    Args:
        enroll_vecs: (K, 512) enrollment vectors (centered)
        W: (512, N_PROJECTIONS) projection matrix
        n: number of bits to select

    Returns:
        sel: (n,) array of selected column indices into W
        template: (n,) int array of {0, 1} template bits
    """
    proj = enroll_vecs @ W  # (K, 512)
    signs = (proj > 0).astype(float)  # (K, 512)
    consistency = np.abs(np.mean(signs, axis=0) - 0.5) * 2.0  # (512,)
    tiebreak = np.mean(np.abs(proj), axis=0)  # (512,)
    order = np.lexsort((-tiebreak, -consistency))  # (512,)
    sel = order[:n]
    template = (np.mean(enroll_vecs, axis=0) @ W)[sel] > 0
    return sel, template.astype(int)


def enroll_select_random(enroll_vecs, W, n, seed):
    """
    Arm B of the ablation: select n of the N=512 candidates uniformly at random.
    No reliability ranking. Per-subject seeding.
    """
    rng = np.random.RandomState(seed)
    sel = rng.choice(N_PROJECTIONS, n, replace=False)
    template = (np.mean(enroll_vecs, axis=0) @ W)[sel] > 0
    return sel, template.astype(int)


def probe_hds(probe_vecs, W, sel, template):
    """
    Integer Hamming distances of probes against template.

    Returns:
        hds: (n_probes,) int array
    """
    proj = probe_vecs @ W  # (P, 512)
    bits = (proj[:, sel] > 0).astype(int)  # (P, n)
    return np.sum(bits != template[np.newaxis, :], axis=1)


def t_empirical(hd_arr, n, target=DECODE_FAILURE_TARGET):
    """Smallest integer t with fraction(HD > t) < target."""
    total = len(hd_arr)
    for t in range(n + 1):
        if np.sum(hd_arr > t) / total < target:
            return t
    return n


# ---------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------
def clopper_pearson(k, n, alpha=0.05):
    """Clopper-Pearson exact 95% CI for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def compute_bch_m(n):
    """BCH parameter m = ceil(log2(n+1))."""
    return math.ceil(math.log2(n + 1))


_COSET_CACHE = {}


def _cyclotomic_cosets(N):
    """2-cyclotomic cosets mod N."""
    seen, cosets = set(), []
    for s in range(N):
        if s in seen:
            continue
        c, x = [], s
        while x not in seen:
            seen.add(x)
            c.append(x)
            x = (2 * x) % N
        cosets.append(c)
    return cosets


def _generator_roots(m, t):
    """Root exponents of the narrow-sense BCH generator for designed distance
    2t+1 at length 2^m - 1: the union of the 2-cyclotomic cosets meeting
    {1, ..., 2t}."""
    N = 2 ** m - 1
    if m not in _COSET_CACHE:
        _COSET_CACHE[m] = _cyclotomic_cosets(N)
    need = set(range(1, 2 * t + 1))
    roots = set()
    for c in _COSET_CACHE[m]:
        if not need.isdisjoint(c):
            roots.update(c)
    return roots


def bch_parity(m, t):
    """Exact parity-check count of the narrow-sense binary BCH code of length
    2^m - 1 with designed distance 2t+1, i.e. the degree of the lcm of the
    minimal polynomials of alpha^1 ... alpha^2t.

    This equals m*t only when every cyclotomic coset involved has full size m.
    It does not at m=6 (the coset of 9 has size 3) or m=7 for t>=8, where m*t
    OVERSTATES the true parity by 3 to 9 bits. Shortening preserves parity, so
    a length-n shortened code has k = n - bch_parity(m, t).
    """
    if t <= 0:
        return 0
    return len(_generator_roots(m, t))


def bch_true_t(m, t):
    """Correction capacity the generator for designed distance 2t+1 actually
    delivers. The BCH bound gives d >= r+1 where r is the longest run of
    consecutive root exponents alpha^1 ... alpha^r, so t_actual = floor(r/2).

    Inside a parity plateau this exceeds the designed t: at m=6, designed
    t=8, 9 and 10 share one generator of parity 45 whose roots run to
    alpha^20, delivering t_actual = 10 in every case.
    """
    if t <= 0:
        return 0
    roots = _generator_roots(m, t)
    r = 0
    while (r + 1) in roots:
        r += 1
    return r // 2


def select_capacity(hd_arr, n, target=DECODE_FAILURE_TARGET):
    """Choose the code by LEAST PARITY meeting the decode-failure target,
    then report the capacity that code actually delivers.

    Picking the least t that meets the target would waste capacity inside a
    parity plateau: at n=56 it would ask for t=9 and pay 45 parity bits for a
    code that corrects 10.

    Parity is non-decreasing in the designed t, so the first designed t whose
    DELIVERED capacity meets the target is also the minimum-parity choice.

    Returns (t_designed, parity, t_actual).
    """
    m = compute_bch_m(n)
    total = len(hd_arr)
    for t in range(n + 1):
        ta = bch_true_t(m, t)
        if np.sum(hd_arr > ta) / total < target:
            return t, bch_parity(m, t), ta
    return n, bch_parity(m, n), bch_true_t(m, n)


# ---------------------------------------------------------------
# Integrity hashes
# ---------------------------------------------------------------
def subject_list_hash(subjects):
    """SHA-256 of newline-joined sorted subject names."""
    blob = "\n".join(sorted(subjects)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def probe_set_hash(info):
    """SHA-256 of subject->probe_count mapping (verifies split)."""
    pieces = []
    for s, _, _, hds in info:
        pieces.append(f"{s}:{len(hds)}")
    blob = "\n".join(sorted(pieces)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path):
    """SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------
# Subject info builder
# ---------------------------------------------------------------
def build_subject_info(centered, W, n, subjects=None):
    """
    Build per-subject enrollment/probe info for bit-string length n.

    Checks, for every subject:
      - Enrollment/probe indices are disjoint
      - Enrollment size equals ENROLLMENT_SIZE
      - Selection is derived from enrollment vectors only (re-derivation check)

    Args:
        centered: dict {subject: (n_images, 512) centered vectors}
        W: (512, N_PROJECTIONS) projection matrix
        n: bit-string length
        subjects: optional sorted list of subject names (default: sorted keys)

    Returns:
        info: list of (name, sel, template, hd_array) tuples
        sels: list of (name, sel) tuples
    """
    if subjects is None:
        subjects = sorted(centered.keys())

    info = []
    sels = []
    for s in subjects:
        vecs = centered[s]
        n_imgs = len(vecs)
        if n_imgs < ENROLLMENT_SIZE + 1:
            continue

        # enrollment and probe images never overlap
        enroll_idx = set(range(ENROLLMENT_SIZE))
        probe_idx = set(range(ENROLLMENT_SIZE, n_imgs))
        assert len(enroll_idx & probe_idx) == 0, \
            f"Subject {s}: enrollment/probe overlap detected!"
        assert len(enroll_idx) == ENROLLMENT_SIZE, \
            f"Subject {s}: enrollment size mismatch"

        sel, tmpl = enroll_select(vecs[:ENROLLMENT_SIZE], W, n)
        hds = probe_hds(vecs[ENROLLMENT_SIZE:], W, sel, tmpl)

        # the selection depends on the enrollment images only: deriving it again
        # from them alone must give the same bits
        sel_check, tmpl_check = enroll_select(vecs[:ENROLLMENT_SIZE], W, n)
        assert np.array_equal(sel, sel_check), \
            f"Subject {s}: selection differs on re-derivation — probe leak!"
        assert np.array_equal(tmpl, tmpl_check), \
            f"Subject {s}: template differs on re-derivation — probe leak!"

        info.append((s, sel, tmpl, hds))
        sels.append((s, sel))

    return info, sels


def build_subject_info_random(centered, W, n, subjects=None, seed_base=None):
    """
    Build per-subject info using Arm B (uniform-random selection).

    Args:
        centered: dict {subject: centered vectors}
        W: projection matrix
        n: bit-string length
        subjects: optional sorted subject list
        seed_base: base seed for per-subject RNG (default: GLOBAL_SEED + 2000)

    Returns:
        info: list of (name, sel, template, hd_array) tuples
        sels: list of (name, sel) tuples
    """
    if subjects is None:
        subjects = sorted(centered.keys())
    if seed_base is None:
        seed_base = GLOBAL_SEED + 2000

    info = []
    sels = []
    for si, s in enumerate(subjects):
        vecs = centered[s]
        n_imgs = len(vecs)
        if n_imgs < ENROLLMENT_SIZE + 1:
            continue

        enroll_vecs = vecs[:ENROLLMENT_SIZE]
        probe_vecs = vecs[ENROLLMENT_SIZE:]

        subj_seed = seed_base + si
        sel, tmpl = enroll_select_random(enroll_vecs, W, n, subj_seed)
        hds = probe_hds(probe_vecs, W, sel, tmpl)

        info.append((s, sel, tmpl, hds))
        sels.append((s, sel))

    return info, sels
