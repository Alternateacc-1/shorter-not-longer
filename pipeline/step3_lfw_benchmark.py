"""Step 3: the standard LFW 6,000-pair verification accuracy of our embedding
path (same centre crop and model as step 1), to compare with the published
figure for this model. 10 folds of 600 consecutive pairs; the threshold of each
fold is chosen on the other nine over their unique cosine scores.

Needs pairs.csv, two folders above the image tree (as in the Kaggle copy).

    python -m pipeline.step3_lfw_benchmark
"""
import csv
import time

from pipeline.provenance import OUT, Log, dump, require_committed_code, sha256_file


def main():
    st = require_committed_code()
    log = Log("step3_lfw_benchmark")
    import numpy as np
    from fuzzycommit.common import LFW_DIR, MODEL_PATH
    from fuzzycommit.embedding import extract_embeddings

    pairs_csv = LFW_DIR.parent.parent / "pairs.csv"
    img = lambda name, num: LFW_DIR / name / f"{name}_{int(num):04d}.jpg"
    pairs = []
    for r in list(csv.reader(open(pairs_csv)))[1:]:
        r = (r + ["", "", "", ""])[:4]
        if r[3].strip() == "":
            pairs.append((img(r[0], r[1]), img(r[0], r[2]), 1))
        else:
            pairs.append((img(r[0], r[1]), img(r[2], r[3]), 0))
    assert len(pairs) == 6000 and sum(p[2] for p in pairs) == 3000
    uniq = sorted({p for pr in pairs for p in pr[:2]})
    assert all(p.exists() for p in uniq)
    log(f"pairs {len(pairs)}, unique images {len(uniq)}, pairs.csv sha256 {sha256_file(pairs_csv)[:16]}")
    t0 = time.time()
    emb = extract_embeddings({"ALL": uniq}, model_path=MODEL_PATH,
                             cache_path=OUT / "lfw_pairs_embeddings.npz")["ALL"]
    secs = time.time() - t0
    idx = {p: i for i, p in enumerate(uniq)}
    A = emb[[idx[a] for a, _, _ in pairs]]; B = emb[[idx[b] for _, b, _ in pairs]]
    A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True)
    cos = np.sum(A * B, axis=1); y = np.array([l for _, _, l in pairs])
    accs, ths = [], []
    for f in range(10):
        te = np.zeros(len(y), bool); te[f * 600:(f + 1) * 600] = True; tr = ~te
        best = (0, 0)
        for th in np.unique(cos[tr]):
            a = np.mean((cos[tr] >= th).astype(int) == y[tr])
            if a > best[0]:
                best = (a, th)
        ths.append(best[1]); accs.append(np.mean((cos[te] >= best[1]).astype(int) == y[te]))
    accs = np.array(accs)
    res = {"git": st, "pairs": 6000, "unique_images": len(uniq), "embedding_seconds": round(secs, 1),
           "accuracy_mean_pct": 100 * float(accs.mean()), "accuracy_sd_pct": 100 * float(accs.std()),
           "per_fold_pct": (100 * accs).tolist(), "threshold_mean": float(np.mean(ths))}
    dump(res, "step3_lfw_benchmark.json")
    log(f"LFW 10-fold accuracy {res['accuracy_mean_pct']:.3f}% +/- {res['accuracy_sd_pct']:.3f}")


if __name__ == "__main__":
    main()
