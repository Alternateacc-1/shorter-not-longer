"""Sampled false-accept worker, one process per bit-string length.

Runs fuzzycommit.false_accept.run_sampled_far at the probe-weighted delivered
capacity t (seed GLOBAL_SEED + 300 + n, 10^6 trials), then replays the same
random stream with precomputed bits to record the whole impostor Hamming
distance histogram. The false-accept rate at any other radius (other weightings,
other codes) then comes from the same 10^6 trials. The replay must reproduce
the accept count at t exactly.
"""
import numpy as np


def _load():
    import fuzzycommit.common as common
    from pipeline.provenance import OUT_MAIN
    common.DATA_DIR = OUT_MAIN          # embeddings written by step 1
    emb, cen = common.load_and_center()
    return common, emb, cen


def worker(args):
    n, proj_seed = args
    common, emb, cen = _load()
    from fuzzycommit.bit_errors import compute_sweep_row
    from fuzzycommit.false_accept import run_sampled_far
    W = common.get_W(proj_seed)
    subs = sorted(cen)
    info, _ = common.build_subject_info(cen, W, n, subs)
    t = int(compute_sweep_row(info, n)["t"])
    seed = common.GLOBAL_SEED + 300 + n
    acc, far, lo, hi = run_sampled_far(info, cen, W, n, t, seed=seed)

    # replay: same RNG calls in the same order, bits precomputed
    ns = len(info)
    sizes = np.array([len(cen[s]) for s in subs])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    bits = np.concatenate([cen[s] for s in subs], axis=0) @ W > 0
    hd_all = np.array([(bits[:, sel] != tmpl.astype(bool)).sum(axis=1)
                       for _, sel, tmpl, _ in info], dtype=np.int16)
    rng = np.random.RandomState(seed)
    hist = np.zeros(n + 1, dtype=np.int64)
    for _ in range(common.N_IMPOSTOR):
        ia, ib = rng.choice(ns, 2, replace=False)
        img = rng.randint(sizes[ib])
        hist[hd_all[ia, starts[ib] + img]] += 1
    replay_acc = int(hist[: t + 1].sum())
    assert replay_acc == acc, f"n={n}: replay {replay_acc} != run_sampled_far {acc}"
    return {"n": n, "proj_seed": proj_seed, "t": t, "seed": seed, "trials": common.N_IMPOSTOR,
            "accepts": int(acc), "far": float(far), "ci_lo": float(lo), "ci_hi": float(hi),
            "hd_hist": hist.tolist()}
