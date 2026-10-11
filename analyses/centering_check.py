"""Centering check (paper Sect. 5.2, "Centering").

The pipeline removes the mean of all 5,425 corpus embeddings (probes included)
before projection (fuzzycommit.common.load_and_center). This script repeats the
selection rule with two means that use no probe image:
  E  the mean of the 1,555 enrollment images (first 5 per subject);
  X  the mean of 800 images of LFW subjects outside the corpus (fewer than
     6 images each), drawn with random.Random(7), embedded exactly as in step 1.
P (the paper's corpus mean) is recomputed as a control; it must reproduce the
manifest (n=15: residual +4.12, closure 100/100/96.1%).

For every shortened-BCH length n = 8..24 and 32 it computes, per weighting
(probe / capped at 10 per subject / subject), the least-parity code and the
N_eff residual, the subject-cluster bootstrap closure (1,000 replicates,
seed 542+n, N_eff resampled; as in step 2), and the full-set pair-uniform
false-accept rate at the largest delivered t over the weightings. The rule:
among lengths positive in >=95% of replicates under every weighting, the
lowest false-accept rate. (The paper's rule uses the sampled false-accept
rate and also scans other code families; this check covers shortened BCH.)

Run after step 1 (it needs results/pipeline/embeddings_cache.npz, the LFW tree and
models/w600k_r50.onnx):
    python -m analyses.centering_check
Writes results/centering/centering_check.json (about 15 min on 2 cores).
"""
import hashlib
import json
import random

import numpy as np

from analyses.helpers import PIPELINE_OUT, out_dir, require_committed
from fuzzycommit.common import (LFW_DIR, ENROLLMENT_SIZE, GLOBAL_SEED, N_BOOTSTRAP, PROBE_CAP,
                                DECODE_FAILURE_TARGET as TARGET, get_W, build_subject_info,
                                bch_parity, bch_true_t, select_capacity, compute_bch_m)
from fuzzycommit.false_accept import impostor_pair_stats, neff_from_stats, run_exhaustive_far

EMB = PIPELINE_OUT / "embeddings_cache.npz"
OUT = out_dir("centering")
LENGTHS = list(range(8, 25)) + [32]
N_EXTERNAL, EXTERNAL_SEED = 800, 7


def least_parity(m, n, failfn):
    for t in range(n + 1):
        ta = bch_true_t(m, t)
        if failfn(ta) < TARGET:
            return t, bch_parity(m, t), ta
    return n, bch_parity(m, n), bch_true_t(m, n)


def outside_pool():
    """Every image of the LFW subjects with 1-5 images (none of them in the corpus), in name
    order. Names are compared as plain strings so the order is the same on every operating
    system; sorting Path objects is case-insensitive on Windows only."""
    pool = []
    for d in sorted((d for d in LFW_DIR.iterdir() if d.is_dir()), key=lambda d: d.name):
        imgs = sorted((p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                      key=lambda p: p.name)
        if 0 < len(imgs) < 6:
            pool += imgs
    return pool


def draw_indices(pool_size, seed=EXTERNAL_SEED):
    """Positions of the 800 drawn images: the pool shuffled with random.Random(seed)."""
    idx = list(range(pool_size))
    random.Random(seed).shuffle(idx)
    return idx[:N_EXTERNAL]


def external_images(seed=EXTERNAL_SEED):
    """The 800 outside images of one draw (the paper's draw is seed 7)."""
    pool = outside_pool()
    return [pool[i] for i in draw_indices(len(pool), seed)]


def external_list_hash(images):
    """SHA-256 of the drawn images as subject/file names, one per line."""
    return hashlib.sha256("\n".join(f"{p.parent.name}/{p.name}" for p in images).encode("utf-8")).hexdigest()


def external_mean(images):
    cache = OUT / f"external_embeddings_{external_list_hash(images)[:12]}.npz"
    if not cache.exists():
        from fuzzycommit.embedding import extract_embeddings
        extract_embeddings({"external": images}, cache_path=cache)
    ext = np.load(cache, allow_pickle=True)["embeddings"].item()["external"]
    assert len(ext) == N_EXTERNAL
    return ext.mean(axis=0)


def setup_length(cen, subs, W, n):
    """Everything about one length that does not depend on the bootstrap seed: the per-subject
    probe distances, the least-parity code under each weighting, the impostor statistics behind
    N_eff, and the full-set false-accept rate at the largest delivered t."""
    info, _ = build_subject_info(cen, W, n, subs)
    m = compute_bch_m(n)
    hds = [x[3] for x in info]
    cap = [h[:PROBE_CAP] for h in hds]
    PF = np.array([[float(np.mean(h > ta)) for ta in range(n + 1)] for h in hds])
    sel = {"probe": select_capacity(np.concatenate(hds), n),
           "capped": select_capacity(np.concatenate(cap), n),
           "subject": least_parity(m, n, lambda ta: float(np.mean(PF[:, ta])))}
    S1, S2, CNT = impostor_pair_stats(cen, W, n, info)
    ne = neff_from_stats(S1, S2, CNT, np.ones(len(subs)))
    tmax = max(s[2] for s in sel.values())
    far = run_exhaustive_far(cen, W, n, tmax, subs)["far_pair_uniform"]
    return {"n": n, "m": m, "hds": hds, "cap": cap, "PF": PF, "sel": sel, "stats": (S1, S2, CNT),
            "neff": ne, "far": far}


def closure(L, seed):
    """Subject-cluster bootstrap (1,000 replicates, N_eff resampled): the share of replicates
    in which the least-parity code exists and leaves a positive residual, per weighting."""
    n, m, hds, cap, PF = L["n"], L["m"], L["hds"], L["cap"], L["PF"]
    S1, S2, CNT = L["stats"]
    NS = len(hds)
    rng = np.random.RandomState(seed)
    par = {w: [] for w in L["sel"]}
    nes = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(NS, NS, replace=True)
        par["probe"].append(select_capacity(np.concatenate([hds[i] for i in idx]), n)[1])
        par["capped"].append(select_capacity(np.concatenate([cap[i] for i in idx]), n)[1])
        par["subject"].append(least_parity(m, n, lambda ta: float(np.mean(PF[idx, ta])))[1])
        nes.append(neff_from_stats(S1, S2, CNT, np.bincount(idx, minlength=NS)))
    nes = np.array(nes)
    return {w: 100 * float(np.mean((n - np.array(par[w]) > 0) & (nes - np.array(par[w]) > 0)))
            for w in par}


def row(L, clo):
    sel, ne = L["sel"], L["neff"]
    return {"neff": ne, "t": {w: int(sel[w][2]) for w in sel},
            "parity": {w: int(sel[w][1]) for w in sel},
            "residual": {w: ne - sel[w][1] for w in sel},
            "closure_pct": clo, "far_full_pair_at_tmax": L["far"]}


def apply_rule(rows):
    """Among lengths positive in >= 95% of replicates under every weighting, the lowest
    false-accept rate. Returns (eligible lengths, chosen length or None)."""
    eligible = [n for n in sorted(rows) if min(rows[n]["closure_pct"].values()) >= 95.0]
    pick = min(eligible, key=lambda n: rows[n]["far_full_pair_at_tmax"]) if eligible else None
    return eligible, pick


def evaluate(emb, subs, mu, W, lengths=LENGTHS, label=None):
    """The rule on embeddings `emb` centred on `mu` and projected with `W`; bootstrap seed 542+n
    as in step 2. Returns (rows, eligible, pick)."""
    rows = {}
    cen = {s: emb[s] - mu for s in subs}
    for n in lengths:
        L = setup_length(cen, subs, W, n)
        rows[n] = row(L, closure(L, GLOBAL_SEED + 500 + n))
        if label:
            r = rows[n]
            print(label, n, {w: round(v, 2) for w, v in r["residual"].items()}, r["closure_pct"],
                  f"{r['far_full_pair_at_tmax']:.5f}", flush=True)
    eligible, pick = apply_rule(rows)
    return rows, eligible, pick


def load_corpus():
    emb = np.load(EMB, allow_pickle=True)["embeddings"].item()
    emb = {k: v for k, v in emb.items() if len(v) >= 6}
    subs = sorted(emb)
    assert len(subs) == 311
    return emb, subs


def main():
    st = require_committed()
    ext_imgs = external_images()
    emb, subs = load_corpus()
    means = {"P_corpus_mean": np.concatenate([emb[s] for s in subs]).mean(axis=0),
             "E_enrollment_mean": np.concatenate([emb[s][:ENROLLMENT_SIZE] for s in subs]).mean(axis=0),
             "X_external_mean": external_mean(ext_imgs)}
    W = get_W()
    res = {"git": st, "lengths": LENGTHS, "external_images": N_EXTERNAL, "external_seed": EXTERNAL_SEED,
           "external_list_sha256": external_list_hash(ext_imgs), "variants": {}}
    for name, mu in means.items():
        rows, eligible, pick = evaluate(emb, subs, mu, W, label=name)
        res["variants"][name] = {"rows": rows, "eligible": eligible, "rule_selects": pick}
        print(name, "rule selects", pick, "eligible", eligible, flush=True)
    p15 = res["variants"]["P_corpus_mean"]["rows"][15]
    assert round(p15["residual"]["probe"], 2) == 4.12 and p15["closure_pct"]["subject"] == 96.1, \
        "control does not reproduce the manifest"
    json.dump(res, open(OUT / "centering_check.json", "w"), indent=1, default=float)
    print("wrote", OUT / "centering_check.json")


if __name__ == "__main__":
    main()
