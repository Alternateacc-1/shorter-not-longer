"""Landmark alignment, part 2: checks step 6 against the second implementation and
writes results/alignment/manifest.csv with what the paper needs beyond step 6
(aligned closure with its mixture, sampled and full-set false-accept rates at
n = 15, 24, 32, probe and subject weighting). The pipeline manifest is not touched.

    python -m analyses.alignment_results     (after analyses.realign_faces)
"""
import csv
import hashlib
import json
import re

import numpy as np

from analyses import crosscheck as V

ALIGNED = V.UNDER_TEST / "aligned_embeddings.npz"
LFW = V._common.LFW_DIR   # the configured LFW tree (LFW_DIR, paths.conf, or the kagglehub default)
SUBJ_SHA = "c587f59d1847cfe5371fa2d2e426d669ee3a99db34dfc3906063337c742521c2"


def tenfold(cos, y):
    accs = []
    for f in range(10):
        te = np.zeros(len(y), bool); te[600 * f:600 * (f + 1)] = True; tr = ~te
        best_acc, best_th = -1.0, None
        for th in np.unique(cos[tr]):                         # threshold from the nine training folds only
            a = np.mean((cos[tr] >= th) == (y[tr] == 1))
            if a > best_acc:
                best_acc, best_th = a, th
        accs.append(np.mean((cos[te] >= best_th) == (y[te] == 1)))
    return 100 * float(np.mean(accs)), 100 * float(np.std(accs))


def pairs_list():
    D = LFW.parent.parent; out = []
    for r in list(csv.reader(open(D / "pairs.csv")))[1:]:
        r = (r + ["", "", "", ""])[:4]
        f = lambda nm, k: LFW / nm / f"{nm}_{int(k):04d}.jpg"
        out.append((f(r[0], r[1]), f(r[0], r[2]), 1) if r[3].strip() == "" else (f(r[0], r[1]), f(r[2], r[3]), 0))
    return out


def envelope(cen, ns=(9, 11, 15, 16, 24, 32)):
    W, _ = V.W_of(43); S = V.Setup(cen, W); out = {}
    for n in ns:
        L = V.Length(S, n); ne = L.neff(); pp, tp = L.choose("probe"); psw, tsw = L.choose("subject")
        out[n] = {"beta_probe": L.beta("probe"), "neff": ne, "parity_probe": pp, "parity_subject": psw,
                  "residual_probe": ne - pp, "residual_subject": ne - psw, "mcv_floor_mean": float(L.mcv_floors().mean())}
    return out, S


def main():
    C = V.Checks("alignment_results"); M = V.manifest(); rl = json.load(open(V.OUTV / "realign_faces.json"))
    al_sha = V.sha256(ALIGNED)
    raw_al = np.load(ALIGNED, allow_pickle=True)["embeddings"].item()
    raw_cc = np.load(V.EMB, allow_pickle=True)["embeddings"].item()
    # ---- same images, same subject list, same split
    same = sorted(raw_al) == sorted(raw_cc) and all(raw_al[s].shape == raw_cc[s].shape for s in raw_cc)
    C.add("align.same", "aligned set: same 311 subjects, same image count per subject (5,425)", "5425",
          sum(len(v) for v in raw_al.values()), same and sum(len(v) for v in raw_al.values()) == 5425)
    h = hashlib.sha256("\n".join(sorted(raw_al)).encode()).hexdigest()
    C.add("align.same", "subject list hash", SUBJ_SHA[:16], h[:16], h == SUBJ_SHA)
    step6_src = (V.REPO / "pipeline" / "step6_alignment.py").read_text(encoding="utf-8").splitlines()
    pats = [r"for s in sorted\(subject_images\):", r"aligned\[s\] = np.array\(\[aligned_embedding\(p\) for p in subject_images\[s\]\]\)",
            r"subs = sorted\(cen\); W = get_W\(\)", r"info, sels = build_subject_info\(cen, W, n, subs\)",
            r"ne = float\(compute_neff\(cen, W, n, info\)\)", r"r = detect\(img\)", r"img = Image.open\(path\).convert\(\"RGB\"\)"]
    q = [f"step6_alignment.py:{i + 1}: {l.strip()}" for i, l in enumerate(step6_src) if any(re.search(p, l) for p in pats)]
    ok = all(any(re.search(p, x) for x in q) for p in pats)
    C.add("align.same", "only the crop differs: same image list, first-5 enrollment, get_W(), build_subject_info, compute_neff on "
          "aligned impostor distances; detection reads only the image", "as stated", "as stated" if ok else "differs", ok,
          note=" || ".join(q))
    # ---- independent re-alignment agrees with the stored aligned set
    cr = rl["corpus_recheck"]
    agree_det = all(r["detected_here"] != r["stored_is_fallback"] for r in cr)
    mincos = min(r["cos"] for r in cr); maxd = max(r["max_abs_diff"] for r in cr)
    C.add("align.realign", f"independent re-alignment of {len(cr)} corpus images (300 random + every fallback): same detect/fallback decision",
          "all agree", "all agree" if agree_det else "differs", agree_det)
    C.add("align.realign", "independent aligned embedding vs stored: min cosine / max |diff|", ">= 0.999",
          f"{mincos:.6f} / {maxd:.2e}", mincos >= 0.999, note=f"bit-identical: {sum(r['max_abs_diff'] == 0 for r in cr)}/{len(cr)}")
    # ---- the fallbacks
    fb = rl["fallbacks_stored"]
    C.num("align.fallback", "centre-crop fallbacks in the aligned corpus (stored vector == centre-crop vector)",
          M["alignment corpus fallbacks"], len(fb), integer=True, note="; ".join(f"{x['file']}" for x in fb))
    # ---- aligned envelope recomputed, then with the fallbacks excluded
    e_al, c_al = V.load_npz_own(ALIGNED)
    env, S = envelope(c_al)
    for n, r in env.items():
        for k in ("beta_probe", "neff", "parity_probe", "parity_subject", "residual_probe", "residual_subject", "mcv_floor_mean"):
            C.num("align.env", f"alignment n={n} aligned {k}", M[f"alignment n={n} aligned {k}"], r[k],
                  integer=k.startswith("parity"))
    drop = {(x["subject"], x["index"]) for x in fb}
    e_img = {s: np.array([v for i, v in enumerate(e_al[s]) if (s, i) not in drop]) for s in e_al}
    e_img = {s: v for s, v in e_img.items() if len(v) >= 6}
    mu = np.concatenate([e_img[s] for s in sorted(e_img)]).mean(0)
    env_img, _ = envelope({s: v - mu for s, v in e_img.items()})
    e_sub = {s: v for s, v in e_al.items() if not any(d[0] == s for d in drop)}
    mu = np.concatenate([e_sub[s] for s in sorted(e_sub)]).mean(0)
    env_sub, _ = envelope({s: v - mu for s, v in e_sub.items()})
    for n in env:
        for k in ("residual_probe", "residual_subject"):
            sgn = lambda x: (x > 0) - (x < 0)
            C.add("align.exclude", f"n={n} {k}: sign unchanged without the {len(fb)} fallback images / without their {len({d[0] for d in drop})} subjects",
                  f"{env[n][k]:+.2f}", f"{env_img[n][k]:+.2f} / {env_sub[n][k]:+.2f}",
                  sgn(env[n][k]) == sgn(env_img[n][k]) == sgn(env_sub[n][k]))
    # ---- LFW 6,000 pairs: centre crop (stored embeddings) and aligned (independent re-alignment)
    pairs = pairs_list(); y = np.array([l for _, _, l in pairs])
    uniq = sorted({p for a, b, _ in pairs for p in (a, b)})
    cc = np.load(V.UNDER_TEST / "lfw_pairs_embeddings.npz", allow_pickle=True)["embeddings"].item()["ALL"]
    ix = {p: i for i, p in enumerate(uniq)}
    A = cc[[ix[a] for a, _, _ in pairs]]; B = cc[[ix[b] for _, b, _ in pairs]]
    cos = np.sum(A / np.linalg.norm(A, axis=1, keepdims=True) * (B / np.linalg.norm(B, axis=1, keepdims=True)), 1)
    acc, sd = tenfold(cos, y)
    C.num("align.lfw", "LFW 6000-pair accuracy, centre crop (stored embeddings, own 10-fold code)", M["LFW 6000-pair accuracy"], acc)
    C.num("align.lfw", "LFW 6000-pair accuracy SD, centre crop", M["LFW 6000-pair accuracy SD"], sd)
    z = np.load(V.OUTV / "aligned_pairs_independent.npz")
    emb = dict(zip(z["keys"], z["emb"]))
    A = np.array([emb[str(a)] for a, _, _ in pairs]); B = np.array([emb[str(b)] for _, b, _ in pairs])
    acc_al, sd_al = tenfold(np.sum(A * B, 1), y)
    C.num("align.lfw", "LFW 6000-pair accuracy, aligned (independent re-alignment)", M["alignment LFW 6000-pair accuracy"], acc_al,
          rel=0, abs_=0.02, note=f"SD {sd_al:.3f}; tolerance one pair (1/6000 = 0.017%), since the alignment is re-implemented")
    for f in ("step3_lfw_benchmark.py", "step6_alignment.py"):
        src = (V.REPO / "pipeline" / f).read_text(encoding="utf-8").splitlines()
        q = [f"{f}:{i + 1}: {l.strip()}" for i, l in enumerate(src) if re.search(r"te\[(f \* 600:\(f \+ 1\) \* 600|600 \* f:600 \* \(f \+ 1\))\] = True|np.unique\(cos\[tr\]\)", l)]
        C.add("align.lfw", f"{f}: 10 folds of 600 consecutive pairs, threshold over the training folds' scores only", "as stated",
              "as stated" if len(q) == 2 else "differs", len(q) == 2, note=" || ".join(q))
    # ---- addendum: aligned closure (mixture), sampled and full-set FAR, n = 15, 24, 32
    commit = V.git_state()["commit"]
    rows = []
    add = lambda q_, v, u, f_, s_: rows.append({"quantity": q_, "value": repr(v) if isinstance(v, float) else v, "units": u,
                                                 "script": "analyses/alignment_results.py", "function": f_,
                                                 "seeds": s_, "commit": commit[:12]})
    add("aligned embeddings sha256", al_sha, "sha256", "results/pipeline/aligned_embeddings.npz (step 6)", "-")
    for n in (15, 24, 32):
        L = V.Length(S, n); ne = L.neff(); draws = V.bootstrap_draws(n, S.NS)
        hist = L.sampled_hist(42 + 300 + n)
        for w in ("probe", "subject"):
            p, t = L.choose(w)
            par = np.array([L.choose(w, c)[0] for c in draws], float)
            nes = np.array([L.neff(c) for c in draws]); res = nes - par; okk = (n - par > 0) & (res > 0)
            seeds = f"projection 43; boot {542 + n}"
            add(f"aligned n={n} {w} parity", int(p), "bits", "least parity meeting 1.5%", "projection 43")
            add(f"aligned n={n} {w} t", int(t), "errors", "delivered capacity", "projection 43")
            add(f"aligned n={n} {w} residual", ne - p, "bits", "N_eff (aligned impostor HDs) - parity", "projection 43")
            add(f"aligned n={n} {w} closure_pct", 100 * float(okk.mean()), "%", "subject-level cluster bootstrap, N_eff per replicate", seeds)
            add(f"aligned n={n} {w} closure_mixture", json.dumps([{"parity": int(q_), "share": round(100 * float((par == q_).mean()), 2),
                                                                     "pos": round(100 * float(okk[par == q_].mean()), 2),
                                                                     "res": round(float(res[par == q_].mean()), 4)}
                                                                    for q_ in sorted(set(par))], separators=(",", ":")),
                "json", "closure = sum_p share_p * P(res>0 | p)", seeds)
            add(f"aligned n={n} {w} residual_ci95", json.dumps([round(float(np.percentile(res, 2.5)), 4),
                                                                round(float(np.percentile(res, 97.5)), 4)]), "bits", "percentile", seeds)
            pair, comp, nacc, ncmp = L.far_full(t)
            add(f"aligned n={n} {w} FAR_sampled", float(hist[:t + 1].sum()) / 1e6, "rate", "10^6 sampled trials", f"far {342 + n}")
            add(f"aligned n={n} {w} FAR_sampled_accepts", int(hist[:t + 1].sum()), "count", "10^6 sampled trials", f"far {342 + n}")
            add(f"aligned n={n} {w} FAR_full_pair", pair, "rate", f"all {ncmp:,} impostor comparisons, pair-uniform", "-")
            add(f"aligned n={n} {w} FAR_full_comparison", comp, "rate", f"all {ncmp:,} comparisons, comparison-uniform", "-")
            add(f"aligned n={n} {w} FAR_independent", V.analytical_far(n, t), "rate", "sum_{i<=t} C(n,i)/2^n", "-")
        add(f"aligned n={n} N_eff", ne, "bits", "p(1-p)/var over aligned impostor HDs", "projection 43")
    with open(V.OUTV / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    C.save({"env_full": env, "env_without_fallback_images": env_img, "env_without_fallback_subjects": env_sub,
            "aligned_sha256": al_sha})
    print(f"alignment: {sum(r['status'] == 'PASS' for r in C.rows)} PASS, {len(C.failed())} FAIL; manifest rows {len(rows)}")


if __name__ == "__main__":
    main()
