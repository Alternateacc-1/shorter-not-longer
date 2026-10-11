"""A second implementation of the main quantities, used by the analyses.

Written from the paper's definitions: exact BCH parity from 2-cyclotomic cosets,
least-parity code choice under each weighting, N_eff, the subject bootstrap, the
full impostor set and the Clopper-Pearson bound. From the library it imports
only the embedding loader and the projection matrix (fuzzycommit.common), so
where an analysis compares its numbers with the pipeline's, the two come from
separate code. The pipeline's manifest and step JSONs are read as values under
test.
"""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as betadist

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTV = REPO / "results" / "alignment"          # output folder of the alignment analysis
OUTV.mkdir(parents=True, exist_ok=True)
UNDER_TEST = REPO / "results" / "pipeline"     # outputs of steps 1-8
EMB = UNDER_TEST / "embeddings_cache.npz"


# The embeddings and W hashes depend on the platform (numpy and ONNX Runtime round
# the last bit of a few floats differently on Windows and Linux/macOS), so they are
# checked against this run's own step 1 and step 2 records, not fixed values.
def _run_hash(name, *keys):
    d = json.load(open(UNDER_TEST / name, encoding="utf-8"))
    for k in keys:
        d = d[k]
    return d
TARGET = 0.015
CAP = 10
N_ENROL = 5
sys.path.insert(0, str(REPO))
import fuzzycommit.common as _common                      # noqa: E402
from fuzzycommit.common import load_and_center, get_W    # noqa: E402  (the only library imports)


# ------------------------------------------------------------------ provenance / io
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_state():
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.rstrip()
    dirty = [l for l in run("status", "--porcelain").splitlines()
             if "analyses/" in l and l.endswith(".py")]
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_analysis_code": dirty}


def manifest():
    M = {}
    for r in csv.DictReader(open(UNDER_TEST / "manifest.csv", encoding="utf-8")):
        v = r["value"]
        try:
            M[r["quantity"]] = float(v)
        except ValueError:
            M[r["quantity"]] = v
    return M


def under_test(name):
    return json.load(open(UNDER_TEST / name, encoding="utf-8"))


class Checks:
    """Rows of (check id, quantity, manifest value, independent value, PASS/FAIL)."""
    def __init__(self, name):
        self.name, self.rows = name, []

    def add(self, cid, q, mval, ival, ok, note=""):
        self.rows.append({"check": cid, "quantity": q, "manifest": _fmt(mval), "independent": _fmt(ival),
                          "status": "PASS" if ok else "FAIL", "note": note})
        print(f"[{'PASS' if ok else 'FAIL'}] {cid} | {q} | manifest {_fmt(mval)} | independent {_fmt(ival)} {note}",
              flush=True)
        return ok

    def num(self, cid, q, mval, ival, integer=False, rel=1e-6, abs_=0.0, note=""):
        if mval is None or ival is None:
            return self.add(cid, q, mval, ival, False, note or "missing value")
        if integer:
            ok = float(mval) == float(ival) and float(ival).is_integer()
        else:
            ok = abs(float(mval) - float(ival)) <= max(rel * abs(float(mval)), abs_)
        return self.add(cid, q, mval, ival, ok, note)

    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]

    def save(self, extra=None):
        json.dump({"git": git_state(), "rows": self.rows, "extra": extra or {}},
                  open(OUTV / f"{self.name}.json", "w", encoding="utf-8"), indent=1, default=_jsonable)


def _fmt(v):
    if isinstance(v, float):
        return repr(v)
    return v if isinstance(v, (str, int, bool)) or v is None else str(v)


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


# ------------------------------------------------------------------ data
def load():
    assert sha256(EMB) == _run_hash("step1_embed.json", "embeddings_npz_sha256"), "embeddings are not step 1's output"
    _common.DATA_DIR = UNDER_TEST
    emb, cen = load_and_center()
    return emb, cen


def load_npz_own(path):
    """Own loader (for the aligned set): filter >= 6 images, centre on the pooled mean."""
    raw = np.load(path, allow_pickle=True)["embeddings"].item()
    e = {k: v for k, v in raw.items() if len(v) >= 6}
    mu = np.concatenate([e[s] for s in sorted(e)]).mean(axis=0)
    return e, {s: v - mu for s, v in e.items()}


def W_of(seed=43):
    W = get_W(seed)
    h = hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()
    if seed == 43:
        w43 = _run_hash("step2_sweep.json", "shared_projection", "W_sha256")
        assert h == w43, f"get_W(43) hash {h[:8]} != step 2 record {w43[:8]}"
        assert np.array_equal(get_W(), W)
    return W, h


# ------------------------------------------------------------------ BCH from 2-cyclotomic cosets
def _cosets(m):
    N = 2 ** m - 1
    seen, out = set(), []
    for s in range(1, N):
        if s in seen:
            continue
        c, x = [], s
        while x not in c:
            c.append(x); x = (2 * x) % N
        seen.update(c); out.append(frozenset(c))
    return out


def bch(m, t):
    """(parity, delivered t) of the narrow-sense length 2^m-1 BCH code of designed distance 2t+1."""
    if t <= 0:
        return 0, 0
    need = set(range(1, 2 * t + 1))
    roots = set().union(*[c for c in _cosets(m) if c & need])
    r = 0
    while r + 1 in roots:
        r += 1
    return len(roots), r // 2


def m_of(n):
    return math.ceil(math.log2(n + 1))


def bch_menu(n):
    """Shortened-BCH options at length n: [(parity, delivered t)], one per distinct generator."""
    m = m_of(n); out = {}
    for t in range(0, n + 1):
        p, td = bch(m, t)
        if p not in out:
            out[p] = td
    return sorted(out.items())


def least_parity(menu, fail):
    """First code (least parity) whose delivered capacity meets the 1.5% target."""
    for p, td in menu:
        if fail(td) < TARGET:
            return p, td
    return menu[-1]


# ------------------------------------------------------------------ extractor
def order_positions(proj5):
    """Reliability order of the 512 positions from the enrollment projections (5 x 512):
    sign agreement (integer, 3..5) descending, ties by mean |projection| descending, then index."""
    pos = (proj5 > 0).sum(axis=0)
    agree = np.maximum(pos, proj5.shape[0] - pos)
    mag = np.abs(proj5).mean(axis=0)
    return np.array(sorted(range(proj5.shape[1]), key=lambda j: (-int(agree[j]), -float(mag[j]), j)))


class Setup:
    """Projection-level quantities for one embedding set and one projection matrix."""
    def __init__(self, cen, W):
        self.subs = sorted(cen); self.NS = len(self.subs)
        self.sizes = np.array([len(cen[s]) for s in self.subs])
        self.nprobe = self.sizes - N_ENROL
        self.owner = np.repeat(np.arange(self.NS), self.sizes)
        self.start = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        allv = np.concatenate([cen[s] for s in self.subs])
        self.bits = allv @ W > 0                                            # every image, 512 signs
        self.enrol_proj = [cen[s][:N_ENROL] @ W for s in self.subs]
        self.order = [order_positions(p) for p in self.enrol_proj]
        self.tmpl_full = np.array([(cen[s][:N_ENROL].mean(axis=0) @ W) > 0 for s in self.subs])
        self.probe_bits = [cen[s][N_ENROL:] @ W > 0 for s in self.subs]
        # consistency of the two ways of computing a probe's signs (one big product vs per subject)
        self.bits_consistent = all(np.array_equal(self.bits[self.start[i] + N_ENROL:self.start[i] + self.sizes[i]],
                                                  self.probe_bits[i]) for i in range(self.NS))


class Length:
    """Everything at one bit-string length n."""
    def __init__(self, S, n):
        self.S, self.n = S, n
        NS = S.NS
        self.sel = [o[:n] for o in S.order]
        self.tmpl = [S.tmpl_full[i, self.sel[i]] for i in range(NS)]
        self.hd = [(S.probe_bits[i][:, self.sel[i]] != self.tmpl[i]).sum(axis=1) for i in range(NS)]
        H = np.zeros((NS, n + 1)); Hc = np.zeros((NS, n + 1))
        for i, h in enumerate(self.hd):
            H[i] = np.bincount(h, minlength=n + 1); Hc[i] = np.bincount(h[:CAP], minlength=n + 1)
        self.H, self.Hc = H, Hc
        # all impostor Hamming distances: template i vs every image of every other subject
        self.IH = np.array([(S.bits[:, self.sel[i]] != self.tmpl[i]).sum(axis=1) for i in range(NS)], dtype=np.int16)
        hn = self.IH / n
        self.S1 = np.zeros((NS, NS)); self.S2 = np.zeros((NS, NS))
        for i in range(NS):
            self.S1[i] = np.bincount(S.owner, hn[i], NS); self.S2[i] = np.bincount(S.owner, hn[i] ** 2, NS)
        self.CNT = np.tile(S.sizes.astype(float), (NS, 1))
        for M in (self.S1, self.S2, self.CNT):
            np.fill_diagonal(M, 0.0)                     # same-subject pairs excluded by original identity
        self.menu = bch_menu(n)

    # failure fraction at radius t under a weighting, for subject multiplicities c
    def fail(self, t, w, c=None):
        c = np.ones(self.S.NS) if c is None else c
        if w == "probe":
            return float(c @ self.H[:, t + 1:].sum(axis=1)) / float(c @ self.H.sum(axis=1))
        if w == "capped":
            return float(c @ self.Hc[:, t + 1:].sum(axis=1)) / float(c @ self.Hc.sum(axis=1))
        per = self.H[:, t + 1:].sum(axis=1) / self.H.sum(axis=1)
        return float(c @ per) / float(c.sum())

    def beta(self, w, c=None):
        c = np.ones(self.S.NS) if c is None else c
        k = np.arange(self.n + 1)
        if w == "probe":
            return float(c @ (self.H @ k)) / float(c @ self.H.sum(axis=1)) / self.n
        if w == "capped":
            return float(c @ (self.Hc @ k)) / float(c @ self.Hc.sum(axis=1)) / self.n
        return float(c @ ((self.H @ k) / self.H.sum(axis=1))) / float(c.sum()) / self.n

    def choose(self, w, c=None, menu=None):
        return least_parity(menu or self.menu, lambda t: self.fail(t, w, c))

    def neff(self, c=None):
        c = np.ones(self.S.NS) if c is None else c
        tot = c @ self.CNT @ c
        p = (c @ self.S1 @ c) / tot
        var = (c @ self.S2 @ c) / tot - p * p
        return float(p * (1 - p) / var)

    def neff_pooled(self):
        """Direct: every one of the impostor comparisons, pooled (no quadratic forms)."""
        mask = self.S.owner[None, :] != np.arange(self.S.NS)[:, None]
        x = self.IH[mask].astype(np.float64) / self.n
        p = x.mean()
        return float(p * (1 - p) / x.var()), int(mask.sum())

    def far_full(self, t):
        mask = self.S.owner[None, :] != np.arange(self.S.NS)[:, None]
        acc = (self.IH <= t) & mask
        comp = float(acc.sum()) / float(mask.sum())
        per_pair = np.zeros((self.S.NS, self.S.NS))
        for i in range(self.S.NS):
            per_pair[i] = np.bincount(self.S.owner, acc[i].astype(float), self.S.NS) / self.S.sizes
        off = ~np.eye(self.S.NS, dtype=bool)
        return float(per_pair[off].mean()), comp, int(acc.sum()), int(mask.sum())

    def sampled_hist(self, seed, trials=1_000_000):
        """The paper's sampled protocol: two distinct subjects uniformly, then one image of the
        second uniformly; same random calls in the same order as fuzzycommit.false_accept."""
        rng = np.random.RandomState(seed)
        hist = np.zeros(self.n + 1, dtype=np.int64)
        NS, st, sz, IH = self.S.NS, self.S.start, self.S.sizes, self.IH
        for _ in range(trials):
            a, b = rng.choice(NS, 2, replace=False)
            hist[IH[a, st[b] + rng.randint(sz[b])]] += 1
        return hist

    def mcv_floors(self, conf=0.99, idx=None, subjects=None):
        """One floor per index set i: -log2 of the one-sided Clopper-Pearson upper bound on the
        probability of the most common value among the enrolled syndromes of `subjects`
        (default all 311), each restricted to subject i's selected positions."""
        subjects = np.arange(self.S.NS) if subjects is None else np.asarray(subjects)
        idx = range(self.S.NS) if idx is None else idx
        out = []
        for i in idx:
            sym = self.S.tmpl_full[np.ix_(subjects, self.sel[i])]
            _, cnt = np.unique(sym, axis=0, return_counts=True)
            out.append(mcv_floor(int(cnt.max()), len(subjects), conf))
        return np.array(out)


def cp_upper(k, N, conf):
    return 1.0 if k >= N else float(betadist.ppf(conf, k + 1, N - k))


def mcv_floor(k, N, conf=0.99):
    return -math.log2(cp_upper(k, N, conf))


def analytical_far(n, t):
    return sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n


def bootstrap_draws(n, NS, reps=1000):
    rng = np.random.RandomState(42 + 500 + n)
    return [np.bincount(rng.choice(NS, NS, replace=True), minlength=NS).astype(float) for _ in range(reps)]
