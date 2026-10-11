"""Quadratic-residue codes QR(17,9,5), QR(31,16,7) and QR(47,24,11), evaluated at their
own lengths like the Golay rows of step 4, and the selection rule applied over every code.

1. Codes: qr_construction builds each code from its idempotent (g = gcd(e, x^p - 1)), checks
   the QR symmetry and enumerates all 2^k codewords. As a second check, g is rebuilt as the
   product of (X - gamma^r) over the residues r, with gamma of order p in GF(2^m) (irreducible
   f found by Rabin's test, no tables), and the minimum weight is found by a second enumeration.
2. Lengths 31 and 47 are not in step 2, so sweep_length(n) repeats step 2's per-length
   computation here. It is first run at n = 17 and 23 and must reproduce step2_sweep.json
   there exactly. Sampled false-accept rate: pipeline.sampled_far.worker (seed 342+n).
3. Code rows (as in step 4): least parity meeting 1.5% among {shortened BCH} U {the QR code}
   under probe / capped-10 / subject weighting; residual; closure over the same 1,000
   replicates as Table 1 (seed 542+n); false-accept rate sampled, full set and independent-bits.
4. Every new number is recomputed with analyses/crosscheck and compared.
5. The selection rule: among candidates whose residual is positive in >= 95% of replicates
   under every weighting, the lowest false-accept rate. Candidates: every shortened-BCH length
   (n = 8..24, 31, 32, 40, 47, 48, 56, 64), extended BCH (16, 24, 32), Golay (23), extended
   Golay (24) and the QR codes. A candidate's false-accept rate is that of the code it would
   deploy, at t = the largest delivered t over the weightings.

    python -m analyses.qr_codes
Writes results/codes_lookalike/qr_codes.json and part_qr_codes.csv."""
import json
import time

from analyses.helpers import PIPELINE_OUT as OUT_MAIN, out_dir, require_committed, write_manifest
from analyses import qr_construction as qr

OUT = out_dir("codes_lookalike")

PS = (17, 31, 47)
WGT = ("probe", "capped", "subject")
RULE_CLOSURE = 95.0
SOFT = ("Soft-decision decoding needs per-bit reliabilities: the gate can compute the probe's own (|w_j . x'| at the n "
        "selected positions, nothing stored), but enrolment-side reliabilities would require the card to store n quantised "
        "projection magnitudes |w_j . x_enrol| (n*q bits for q-bit levels) beside C, template-dependent side information "
        "outside bound (1).")


# ------------------------------------------------------------------ independent field construction
def _pmod(a, b):
    db = b.bit_length() - 1
    while a and a.bit_length() - 1 >= db:
        a ^= b << (a.bit_length() - 1 - db)
    return a


def _pmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1; b >>= 1
    return r


def _gcd(a, b):
    while b:
        a, b = b, _pmod(a, b)
    return a


def _xpow2k(k, f):                                   # x^(2^k) mod f
    r = 0b10
    for _ in range(k):
        r = _pmod(_pmul(r, r), f)
    return r


def irreducible(m):
    """Least irreducible binary polynomial of degree m (Rabin's test)."""
    primes = [q for q in range(2, m + 1) if m % q == 0 and all(q % s for s in range(2, q))]
    for f in range((1 << m) | 1, 1 << (m + 1), 2):
        if _xpow2k(m, f) != 0b10:
            continue
        if all(_gcd(_xpow2k(m // q, f) ^ 0b10, f) == 1 for q in primes):
            return f


def field_generator(p):
    m = next(k for k in range(1, p) if pow(2, k, p) == 1)
    f = irreducible(m)

    def fmul(a, b):
        return _pmod(_pmul(a, b), f)

    def fpow(a, e):
        r = 1
        while e:
            if e & 1:
                r = fmul(r, a)
            a = fmul(a, a); e >>= 1
        return r
    assert (2 ** m - 1) % p == 0
    gamma = next(g for g in (fpow(a, (2 ** m - 1) // p) for a in range(2, 1 << m)) if g != 1)
    assert fpow(gamma, p) == 1                                     # gamma != 1, gamma^p = 1: order p (p prime)
    poly = [1]                                                     # coefficients in GF(2^m), low first
    for r in qr.residues(p):
        root = fpow(gamma, r); new = [0] * (len(poly) + 1)
        for j, c in enumerate(poly):
            new[j + 1] ^= c; new[j] ^= fmul(c, root)
        poly = new
    assert all(c in (0, 1) for c in poly)
    return sum(c << j for j, c in enumerate(poly)), m, f


def main():
    st = require_committed()
    T0 = time.time()
    import numpy as np
    from scipy.stats import binom, beta as bdist
    from pipeline.provenance import EMB_NPZ, sha256_file
    from analyses import crosscheck as V

    env = json.load(open(OUT_MAIN / "step1_embed.json", encoding="utf-8"))
    assert sha256_file(EMB_NPZ) == env["embeddings_npz_sha256"]
    S1R = json.load(open(OUT_MAIN / "step2_sweep.json", encoding="utf-8"))
    S3R = json.load(open(OUT_MAIN / "step4_robustness.json", encoding="utf-8"))
    LD = S1R["lengths_detail"]
    R = {"git": st, "soft_decision": SOFT}

    # ============================================================== 1. codes
    C = {}
    for p in (17, 23, 31, 47):
        c = qr.qr_code(p)
        gf, m, f = field_generator(p)
        g_other = qr.pgcd((1 << p) | 1, sum(1 << r for r in sorted(set(range(1, p)) - set(qr.residues(p)))) ^ (1 if p % 8 == 1 else 0))
        dmin, total = qr.min_weight(c["rows"])
        c["independent"] = {"field_m": m, "field_poly": hex(f), "field_generator": hex(gf),
                            "field_generator_is_Q_or_N_code": gf in (c["generator_poly"], g_other),
                            "second_min_weight": dmin, "second_codewords": total}
        assert c["independent"]["field_generator_is_Q_or_N_code"] and dmin == c["d"] and total == 2 ** c["k"]
        assert c["invariant_under_all_Q_multipliers"] and not c["invariant_under_some_N_multiplier"]
        if p in (17, 31):                                       # third check: pipeline.bch_codes enumerator (k <= 22)
            from pipeline.bch_codes import min_distance
            c["independent"]["repo_min_distance"] = min_distance(c["rows"], p)
            assert c["independent"]["repo_min_distance"] == c["d"]
        C[p] = c
        print(f"QR({p},{c['k']},{c['d']}) t={c['t']} parity={c['parity']} A_d={c['min_weight_codewords']} "
              f"g={c['generator_hex']} field g={hex(gf)} (GF(2^{m}), f={hex(f)}) second d={dmin}", flush=True)
    from pipeline.bch_codes import GOLAY_G, shortened_cyclic           # p=23 is the Golay code already in the paper
    golay_A = qr.weight_distribution(shortened_cyclic(GOLAY_G, 23), 23)
    assert {int(w): int(a) for w, a in enumerate(golay_A) if a} == C[23]["weight_distribution"] and C[23]["d"] == 7
    expect = {17: (9, 5), 31: (16, 7), 47: (24, 11)}
    for p in PS:
        assert (C[p]["k"], C[p]["d"]) == expect[p], (p, C[p]["k"], C[p]["d"])
    R["codes"] = {p: {k: v for k, v in C[p].items() if k not in ("rows", "generator_poly")} for p in C}

    # ============================================================== 2. step 2's per-length computation
    import fuzzycommit.common as common
    common.DATA_DIR = OUT_MAIN
    from fuzzycommit.common import (GLOBAL_SEED, DECODE_FAILURE_TARGET as TARGET, PROBE_CAP, N_BOOTSTRAP, N_IMPOSTOR,
                            load_and_center, get_W, build_subject_info, select_capacity, bch_parity, bch_true_t,
                            t_empirical, probe_set_hash)
    from fuzzycommit.bit_errors import compute_sweep_row
    from fuzzycommit.false_accept import impostor_pair_stats, neff_from_stats, compute_neff, run_exhaustive_far, compute_analytical_far
    from fuzzycommit.bootstrap import run_bootstrap
    from fuzzycommit.weighting import run_capped_and_weighted, run_tertiles
    from fuzzycommit.entropy import enrolled_bitstrings
    emb, cen = load_and_center()
    subs = sorted(cen); NS = len(subs)
    W = get_W()
    sizes = np.array([len(cen[s]) for s in subs])
    owner = np.repeat(np.arange(NS), sizes)
    allbits = np.concatenate([cen[s] for s in subs], axis=0) @ W > 0
    TB = enrolled_bitstrings(cen, W).astype(np.uint8)
    ONES = np.ones(NS)

    def least_parity(m, n, failfn):
        for t in range(n + 1):
            ta = bch_true_t(m, t)
            if failfn(ta) < TARGET:
                return t, bch_parity(m, t), ta
        return n, bch_parity(m, n), bch_true_t(m, n)

    def mixture(n, par, neff):
        par = np.asarray(par); neff = np.asarray(neff)
        res = neff - par
        ok = (n - par > 0) & (res > 0)
        br = []
        for p in sorted(set(par.tolist())):
            sel = par == p
            br.append({"parity": int(p), "share_pct": 100 * float(sel.mean()),
                       "residual_mean": float(res[sel].mean()), "residual_sd": float(res[sel].std()),
                       "pct_positive_within": 100 * float(ok[sel].mean()),
                       "contribution_pct": 100 * float((ok & sel).mean())})
        clo = 100 * float(ok.mean())
        assert abs(sum(b["contribution_pct"] for b in br) - clo) < 1e-9
        return {"closure_pct": clo, "residual_median": float(np.median(res)),
                "residual_ci95": [float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))],
                "branches": br}

    def frr_summary(v):
        v = np.asarray(v, float)
        return {"subjects": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)),
                "max": float(v.max()), "n_gt_5pct": int((v > 0.05).sum()),
                "n_gt_10pct": int((v > 0.10).sum()), "n_zero": int((v == 0).sum())}

    def sweep_length(n):
        """Step 2's per-length loop body (pipeline/step2_sweep.py), unchanged but for the return."""
        info, sels = build_subject_info(cen, W, n, subs)
        assert [x[0] for x in info] == subs
        row = compute_sweep_row(info, n)
        m = row["m"]
        hds = [x[3] for x in info]
        all_hd = row["all_hd"]
        d = {"n": n, "m": m, "beta": row["beta"], "std_beta": row["std_beta"],
             "dispersion": row["overdispersion"], "n_probes": row["n_probes"],
             "probe_set_sha256": probe_set_hash(info)}
        cap = [h[:PROBE_CAP] if len(h) > PROBE_CAP else h for h in hds]
        all_cap = np.concatenate(cap)
        PF = np.array([[float(np.mean(h > ta)) for ta in range(n + 1)] for h in hds])
        sw_fail = lambda ta: float(np.mean(np.array([float(np.mean(h > ta)) for h in hds])))
        sel_w = {"probe": (row["t_designed"], row["parity"], row["t"]),
                 "capped": select_capacity(all_cap, n),
                 "subject": least_parity(m, n, sw_fail)}
        beta_w = {"probe": row["beta"], "capped": float(np.mean(all_cap / n)),
                  "subject": float(np.mean([float(np.mean(h / n)) for h in hds]))}
        fail_w = {"probe": row["decode_fail_frac"],
                  "capped": float(np.mean(all_cap > sel_w["capped"][2])),
                  "subject": sw_fail(sel_w["subject"][2])}
        rw = run_capped_and_weighted(cen, W, n)
        assert (rw["capped"]["parity"], rw["capped"]["t"]) == sel_w["capped"][1:]
        assert (rw["subject_weighted"]["parity"], rw["subject_weighted"]["t"]) == sel_w["subject"][1:]
        assert rw["capped"]["beta"] == round(beta_w["capped"], 4)
        assert rw["subject_weighted"]["beta"] == round(beta_w["subject"], 4)
        S1, S2, CNT = impostor_pair_stats(cen, W, n, info)
        ne = neff_from_stats(S1, S2, CNT, ONES)
        assert ne == float(compute_neff(cen, W, n, info)) == rw["entropy_bits"]
        d["neff"] = ne
        d["neff_over_n"] = ne / n
        d["weightings"] = {w: {"t_designed": int(sel_w[w][0]), "parity": int(sel_w[w][1]),
                               "t": int(sel_w[w][2]), "k": int(n - sel_w[w][1]),
                               "beta": beta_w[w], "decode_fail": fail_w[w],
                               "residual": ne - sel_w[w][1]} for w in sel_w}
        t_pw = int(row["t"])
        d["t_required_designed"] = int(t_empirical(all_hd, n))
        d["t_binomial"] = int(next(t for t in range(n + 1) if binom.sf(t, n, row["beta"]) < TARGET))
        d["decode_fail_by_t"] = {t: {"empirical": float(np.mean(all_hd > t)),
                                     "binomial": float(binom.sf(t, n, row["beta"]))}
                                 for t in range(min(n, 16) + 1)}
        exh = run_exhaustive_far(cen, W, n, t_pw, subs)
        A = np.zeros((NS, NS, n + 1), np.int32)
        for i, (_, sel, tmpl, _) in enumerate(info):
            h = (allbits[:, sel] != tmpl.astype(bool)).sum(axis=1)
            cnt = np.bincount(owner * (n + 1) + h, minlength=NS * (n + 1)).reshape(NS, n + 1)
            A[i] = np.cumsum(cnt, axis=1)
        eye = np.eye(NS, dtype=bool)
        A[eye] = 0
        off = ~eye
        comps = int(np.broadcast_to(sizes, (NS, NS))[off].sum())
        far_pair = [float((A[:, :, t] / sizes[None, :])[off].mean()) for t in range(n + 1)]
        far_comp = [float(A[:, :, t][off].sum() / comps) for t in range(n + 1)]
        assert comps == exh["n_comparisons"]
        assert far_comp[t_pw] == exh["far_comparison_uniform"]
        assert abs(far_pair[t_pw] - exh["far_pair_uniform"]) < 1e-12
        assert abs(round(ne, 4) - exh["n_eff"]) <= 1.01e-4
        d["exhaustive"] = {"comparisons": comps, "ordered_pairs": NS * (NS - 1),
                           "far_pair_uniform_by_t": far_pair, "far_comparison_uniform_by_t": far_comp,
                           "accepts_by_t": [int(A[:, :, t][off].sum()) for t in range(n + 1)],
                           "repo_far_pair_uniform": exh["far_pair_uniform"],
                           "repo_far_comparison_uniform": exh["far_comparison_uniform"],
                           "impostor_mean": exh["p_bar"], "repo_n_eff_4dp": exh["n_eff"]}
        d["far_analytical_by_t"] = [compute_analytical_far(n, t) for t in range(n + 1)]
        ks = np.array([np.unique(TB[:, sel], axis=0, return_counts=True)[1].max() for _, sel in sels])
        fl = -np.log2(bdist.ppf(0.99, ks + 1, NS - ks))
        flb = -np.log2(bdist.ppf(1 - 0.01 / NS, ks + 1, NS - ks))
        d["mcv"] = {"max_multiplicity_mean": float(ks.mean()), "max_multiplicity_max": int(ks.max()),
                    "floor_mean": float(fl.mean()), "floor_min": float(fl.min()),
                    "floor_bonf_mean": float(flb.mean()), "floor_bonf_min": float(flb.min()),
                    "bonferroni_alpha": 0.01 / NS}
        for w in sel_w:
            p = sel_w[w][1]
            d["weightings"][w]["bound"] = {
                "mean": float(fl.mean() - p), "min": float(fl.min() - p),
                "min_bonferroni": float(flb.min() - p), "mean_bonferroni": float(flb.mean() - p),
                "index_sets_positive": int((fl > p).sum()),
                "index_sets_positive_bonferroni": int((flb > p).sum())}
        pct_repo, boot_repo = run_bootstrap(info, n, ne, cen, W)
        rng = np.random.RandomState(GLOBAL_SEED + 500 + n)
        par = {"probe": [], "capped": [], "subject": []}
        nes = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(NS, NS, replace=True)
            par["probe"].append(select_capacity(np.concatenate([hds[i] for i in idx]), n)[1])
            par["capped"].append(select_capacity(np.concatenate([cap[i] for i in idx]), n)[1])
            par["subject"].append(least_parity(m, n, lambda ta: float(np.mean(PF[idx, ta])))[1])
            nes.append(neff_from_stats(S1, S2, CNT, np.bincount(idx, minlength=NS)))
        nes = np.array(nes)
        assert np.array_equal(nes - np.array(par["probe"]), boot_repo["residuals"])
        d["bootstrap"] = {"seed": GLOBAL_SEED + 500 + n, "replicates": N_BOOTSTRAP,
                          "neff_sd": float(nes.std()), "neff_mean": float(nes.mean()),
                          "repo_closure_pct": pct_repo}
        for w in par:
            d["bootstrap"][w] = mixture(n, par[w], nes)
        assert d["bootstrap"]["probe"]["closure_pct"] == pct_repo
        ter_repo = run_tertiles(cen, emb, W, n)
        cnts = sorted([(x[0], len(emb[x[0]])) for x in info], key=lambda x: x[1])
        cuts = [NS // 3, 2 * NS // 3, NS]
        groups = [cnts[:cuts[0]], cnts[cuts[0]:cuts[1]], cnts[cuts[1]:]]
        d["tertiles"] = []
        for g, rep in zip(groups, ter_repo):
            names = {x[0] for x in g}
            ids = [i for i, s in enumerate(subs) if s in names]
            c = np.zeros(NS); c[ids] = 1
            th = [hds[i] for i in ids]
            tcap = [cap[i] for i in ids]
            ne_t = neff_from_stats(S1, S2, CNT, c)
            selt = {"probe": select_capacity(np.concatenate(th), n),
                    "capped": select_capacity(np.concatenate(tcap), n),
                    "subject": least_parity(m, n, lambda ta: float(np.mean(PF[ids, ta])))}
            assert (rep["parity"], rep["t"]) == selt["probe"][1:]
            assert rep["neff"] == round(ne_t, 3) and rep["n_subjects"] == len(ids)
            d["tertiles"].append({
                "label": rep["label"], "subjects": len(ids), "probes": int(sum(len(h) for h in th)),
                "images_min": int(g[0][1]), "images_max": int(g[-1][1]), "neff": ne_t,
                "beta": {"probe": float(np.mean(np.concatenate(th) / n)),
                         "subject": float(np.mean([np.mean(h / n) for h in th]))},
                "weightings": {w: {"parity": int(selt[w][1]), "t": int(selt[w][2]),
                                   "residual": ne_t - selt[w][1]} for w in selt},
                "frr_at_global_t": frr_summary(PF[ids, t_pw])})
        d["per_subject_frr"] = {"t": t_pw, "summary": frr_summary(PF[:, t_pw]),
                                "values": PF[:, t_pw].tolist()}
        return d, {"hds": hds, "S": (S1, S2, CNT), "m": m}

    norm = lambda x: json.loads(json.dumps(x))
    S1N = {}
    for n in (17, 23):                                          # must reproduce step 2 where it has values
        d, aux = sweep_length(n)
        ref = {k: v for k, v in LD[str(n)].items() if k != "sampled"}
        assert norm(d) == norm(ref), f"sweep_length differs from step2_sweep.json at n={n}"
        S1N[n] = (norm(d), aux, LD[str(n)]["sampled"])
        print(f"sweep_length reproduces step2_sweep.json at n={n} exactly", flush=True)
    from pipeline.sampled_far import worker
    for n in (31, 47):
        t1 = time.time()
        d, aux = sweep_length(n)
        fr = worker((n, 43))
        fr["far_ratio"] = fr["accepts"] / N_IMPOSTOR / compute_analytical_far(n, fr["t"])
        assert fr["t"] == d["weightings"]["probe"]["t"]
        S1N[n] = (norm(d), aux, fr)
        print(f"n={n}: beta {d['beta']:.4f} N_eff {d['neff']:.3f} " + " | ".join(
            f"{w[:3]} t={d['weightings'][w]['t']} p={d['weightings'][w]['parity']} res={d['weightings'][w]['residual']:+.2f} "
            f"clo={d['bootstrap'][w]['closure_pct']:.1f}" for w in WGT) + f" FAR {fr['far']:.3g} ({time.time() - t1:.0f}s)", flush=True)
    R["lengths"] = {n: dict(S1N[n][0], sampled=S1N[n][2]) for n in (31, 47)}

    # ============================================================== 3. code rows (as in step 4)
    def clean(menu):
        best, out = -1, []
        for p, td in sorted(menu):
            if td > best:
                out.append((p, td)); best = td
        return out

    QROWS = {}
    for p in PS:
        n = p; d, aux, sam = S1N[n]
        hds = aux["hds"]; m = aux["m"]; S1, S2, CNT = aux["S"]
        H = np.array([np.bincount(h, minlength=n + 1) for h in hds], float)
        Hc = np.array([np.bincount(h[:PROBE_CAP], minlength=n + 1) for h in hds], float)
        TAIL = np.concatenate([np.flip(np.flip(H, 1).cumsum(1), 1)[:, 1:], np.zeros((NS, 1))], 1)
        TAILC = np.concatenate([np.flip(np.flip(Hc, 1).cumsum(1), 1)[:, 1:], np.zeros((NS, 1))], 1)
        NPR = (sizes - 5).astype(float); NCAP = np.minimum(sizes - 5, PROBE_CAP).astype(float)
        PF = TAIL / NPR[:, None]

        def fails(c, w):
            if w == "probe":
                return (c @ TAIL) / (c @ NPR)
            if w == "capped":
                return (c @ TAILC) / (c @ NCAP)
            return (c @ PF) / c.sum()

        def choose(c, w, menu):
            f = fails(c, w)
            for pp, td in menu:
                if f[td] < TARGET:
                    return pp, td
            return menu[-1]
        seen = {}
        for t in range(n + 1):
            pp = bch_parity(m, t)
            if pp not in seen:
                seen[pp] = bch_true_t(m, t)
        bmenu = sorted(seen.items())
        fams = {"shortened BCH (paper)": bmenu,
                "shortened BCH + QR": clean(bmenu + [(C[p]["parity"], C[p]["t"])])}
        hist = np.array(sam["hd_hist"]); exh = d["exhaustive"]
        rng = np.random.RandomState(GLOBAL_SEED + 500 + n)
        draws = [np.bincount(rng.choice(NS, NS, replace=True), minlength=NS).astype(float) for _ in range(N_BOOTSTRAP)]
        nes = np.array([neff_from_stats(S1, S2, CNT, c) for c in draws])
        ne = d["neff"]
        out = {"neff": ne, "bch_menu": bmenu, "qr": (C[p]["parity"], C[p]["t"]), "families": {}}
        for fam, menu in fams.items():
            rows = {}
            for w in WGT:
                pp, td = choose(ONES, w, menu)
                bp = np.array([choose(c, w, menu)[0] for c in draws], float)
                res = nes - bp; ok = (n - bp > 0) & (res > 0)
                br = [{"parity": int(q), "share_pct": 100 * float((bp == q).mean()),
                       "pct_positive_within": 100 * float(ok[bp == q].mean()),
                       "residual_mean": float(res[bp == q].mean())} for q in sorted(set(bp))]
                rows[w] = {"parity": int(pp), "t": int(td), "k": int(n - pp), "residual": ne - pp,
                           "decode_fail": float(fails(ONES, w)[td]), "closure_pct": 100 * float(ok.mean()), "mixture": br,
                           "far_sampled": float(hist[: td + 1].sum() / N_IMPOSTOR),
                           "far_full_pair": exh["far_pair_uniform_by_t"][td],
                           "far_full_comparison": exh["far_comparison_uniform_by_t"][td],
                           "far_independent": compute_analytical_far(n, td),
                           "code_used": "QR" if (pp, td) == (C[p]["parity"], C[p]["t"]) and (pp, td) not in bmenu else "shortened BCH"}
            out["families"][fam] = rows
        for w in WGT:                                           # same replicates as step 2
            assert out["families"]["shortened BCH (paper)"][w]["closure_pct"] == d["bootstrap"][w]["closure_pct"], (n, w)
            assert out["families"]["shortened BCH (paper)"][w]["parity"] == d["weightings"][w]["parity"], (n, w)
        QROWS[p] = out
        for fam, rows in out["families"].items():
            print(f"n={n} {fam:<24} " + " | ".join(f"{w[:3]} p={r['parity']} t={r['t']} {r['code_used'][:3]} "
                                                    f"res={r['residual']:+.2f} clo={r['closure_pct']:.1f} FAR={r['far_sampled']:.3g}"
                                                    for w, r in rows.items()), flush=True)
    R["qr_rows"] = QROWS

    # ============================================================== 4. independent library
    VX = []

    def vx(q, a, b, tol=1e-12):
        ok = (a == b) if isinstance(a, (int, str, tuple, list)) else abs(a - b) <= tol * max(1.0, abs(a))
        VX.append({"quantity": q, "pipeline": a, "crosscheck": b, "ok": bool(ok)})
        if not ok:
            print(f"CROSSCHECK MISMATCH {q}: {a} vs {b}", flush=True)
    emb_v, cen_v = V.load(); Wv, _ = V.W_of(43); SV = V.Setup(cen_v, Wv)
    for p in PS:
        n = p; L = V.Length(SV, n); d = S1N[n][0]
        vx(f"n={n} N_eff", d["neff"], L.neff(), 1e-9)
        vx(f"n={n} beta (probe)", d["beta"], L.beta("probe"), 1e-12)
        for w in WGT:
            vx(f"n={n} {w} beta", d["weightings"][w]["beta"], L.beta(w), 1e-12)
        draws = V.bootstrap_draws(n, SV.NS)
        nes_v = np.array([L.neff(c) for c in draws])
        hist_v = L.sampled_hist(42 + 300 + n)
        vx(f"n={n} sampled HD histogram (seed {342 + n})", [int(x) for x in S1N[n][2]["hd_hist"]], [int(x) for x in hist_v])
        bm = V.bch_menu(n)
        vx(f"n={n} shortened-BCH menu", [list(x) for x in QROWS[p]["bch_menu"]], [list(x) for x in bm])
        for fam, menu in (("shortened BCH (paper)", bm), ("shortened BCH + QR", sorted(set(bm) | {(C[p]["parity"], C[p]["t"])}))):
            for w in WGT:
                r = QROWS[p]["families"][fam][w]
                pp, td = L.choose(w, None, menu)
                vx(f"n={n} [{fam}] {w} parity,t", (r["parity"], r["t"]), (pp, td))
                vx(f"n={n} [{fam}] {w} residual", r["residual"], L.neff() - pp, 1e-9)
                vx(f"n={n} [{fam}] {w} decode_fail", r["decode_fail"], L.fail(td, w), 1e-12)
                bp = np.array([L.choose(w, c, menu)[0] for c in draws], float)
                clo = 100 * float(((n - bp > 0) & (nes_v - bp > 0)).mean())
                vx(f"n={n} [{fam}] {w} closure_pct", r["closure_pct"], clo, 1e-12)
                pair, comp, acc, ncmp = L.far_full(td)
                vx(f"n={n} [{fam}] {w} far_full_pair", r["far_full_pair"], pair, 1e-12)
                vx(f"n={n} [{fam}] {w} far_full_comparison", r["far_full_comparison"], comp, 1e-12)
                vx(f"n={n} [{fam}] {w} far_sampled", r["far_sampled"], float(hist_v[: td + 1].sum()) / 1e6, 1e-12)
                vx(f"n={n} [{fam}] {w} far_independent", r["far_independent"], V.analytical_far(n, td), 1e-12)
        print(f"crosscheck n={n}: {sum(x['ok'] for x in VX)}/{len(VX)} so far", flush=True)
    R["crosscheck"] = {"checks": len(VX), "agree": sum(x["ok"] for x in VX), "rows": VX}

    # ============================================================== 5. the rule over the union
    def bch_row(n, d, sam):
        hist = np.array(sam["hd_hist"]); exh = d["exhaustive"]; out = {}
        for w in WGT:
            t = d["weightings"][w]["t"]
            out[w] = {"parity": d["weightings"][w]["parity"], "t": t, "k": d["weightings"][w]["k"],
                      "residual": d["weightings"][w]["residual"], "closure_pct": d["bootstrap"][w]["closure_pct"],
                      "code_used": "shortened BCH"}
        return out, hist, exh

    cands = []

    def add_cand(cid, n, family, rows, hist, exh):
        tmax = max(r["t"] for r in rows.values())
        clo = min(r["closure_pct"] for r in rows.values())
        cands.append({"id": cid, "n": n, "family": family, "t_by_weighting": {w: rows[w]["t"] for w in WGT},
                      "parity_by_weighting": {w: rows[w]["parity"] for w in WGT},
                      "code_used": sorted({rows[w]["code_used"] for w in WGT}),
                      "residual_min": min(r["residual"] for r in rows.values()),
                      "k_min": min(r["k"] for r in rows.values()), "closure_min": clo,
                      "closure_by_weighting": {w: rows[w]["closure_pct"] for w in WGT}, "t_deploy": tmax,
                      "far_sampled": float(hist[: tmax + 1].sum()) / N_IMPOSTOR,
                      "far_full_pair": exh["far_pair_uniform_by_t"][tmax],
                      "far_full_comparison": exh["far_comparison_uniform_by_t"][tmax],
                      "eligible": clo >= RULE_CLOSURE})
    for k in sorted(LD, key=int):
        rows, hist, exh = bch_row(int(k), LD[k], LD[k]["sampled"])
        add_cand(f"n={k} shortened BCH", int(k), "shortened BCH", rows, hist, exh)
    for n in (31, 47):
        rows, hist, exh = bch_row(n, S1N[n][0], S1N[n][2])
        add_cand(f"n={n} shortened BCH", n, "shortened BCH", rows, hist, exh)
    for k, o in S3R["a_codes"].items():
        mk = LD[k]["m"]
        bset = {(bch_parity(mk, t), bch_true_t(mk, t)) for t in range(int(k) + 1)}
        for fam, rows in o["families"].items():
            if fam == "shortened BCH (paper)":
                continue
            alts = {(a["parity"], a["t"]) for a in o["codes"] if a["k"] > 0 and
                    (fam.endswith("all alternatives") or fam == f"shortened BCH + {a['family']}")}
            assert alts, (k, fam)
            rr = {w: dict(r, code_used="alternative" if (r["parity"], r["t"]) in alts - bset else "shortened BCH")
                  for w, r in rows.items()}
            hist = np.array(LD[k]["sampled"]["hd_hist"]); exh = LD[k]["exhaustive"]
            add_cand(f"n={k} {fam}", int(k), fam, rr, hist, exh)
    for p in PS:
        rows = QROWS[p]["families"]["shortened BCH + QR"]
        add_cand(f"n={p} shortened BCH + QR", p, "shortened BCH + QR", rows, np.array(S1N[p][2]["hd_hist"]), S1N[p][0]["exhaustive"])
    elig = [c for c in cands if c["eligible"]]
    by = {}
    for metric in ("far_sampled", "far_full_pair", "far_full_comparison"):
        best = min(elig, key=lambda c: (c[metric], c["n"]))
        by[metric] = best["id"]
    c15 = next(c for c in cands if c["id"] == "n=15 shortened BCH")
    nonbch = [c for c in cands if c["family"] != "shortened BCH" and any(u != "shortened BCH" for u in c["code_used"])]
    beat = [c["id"] for c in nonbch if c["eligible"] and c["far_sampled"] < c15["far_sampled"]]
    bch_only = min([c for c in elig if c["family"] == "shortened BCH"], key=lambda c: (c["far_sampled"], c["n"]))
    near = sorted(nonbch, key=lambda c: -c["closure_min"])[:4]
    R["rule"] = {"closure_threshold_pct": RULE_CLOSURE, "candidates": cands, "eligible": [c["id"] for c in elig],
                 "choice_by_metric": by, "bch_only_choice": bch_only["id"], "n15": c15,
                 "non_bch_rows_that_use_their_code": [c["id"] for c in nonbch],
                 "non_bch_beating_n15": beat, "triggered": bool(beat),
                 "nearest_non_bch": [{k: c[k] for k in ("id", "closure_min", "closure_by_weighting", "far_sampled", "t_deploy")} for c in near]}
    print(f"rule: eligible {R['rule']['eligible']}", flush=True)
    print(f"rule: choice {by}; BCH-only {bch_only['id']}; non-BCH beating n=15: {beat or 'none'}", flush=True)
    print("nearest non-BCH:", [(c["id"], round(c["closure_min"], 1)) for c in near], flush=True)
    R["wall_seconds"] = round(time.time() - T0, 1)
    json.dump(R, open(OUT / "qr_codes.json", "w", encoding="utf-8"), indent=1, default=V._jsonable)

    # ============================================================== manifest rows
    rows_m = []

    def add(q, v, u, f, seeds="projection 43"):
        rows_m.append({"quantity": q, "value": json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else
                       (repr(float(v)) if isinstance(v, float) else v), "units": u, "script": "analyses/qr_codes.py",
                       "function": f, "seeds": seeds, "commit": st["commit"][:12]})
    for p in PS:
        c = C[p]
        for k, u in (("k", "bits"), ("d", "-"), ("t", "errors"), ("parity", "bits"), ("min_weight_codewords", "codewords")):
            add(f"QR({p}) {k}", int(c[k]), u, "qr_construction.qr_code: idempotent, exhaustive enumeration of 2^k codewords", "-")
        add(f"QR({p}) generator", c["generator_hex"], "poly (bit j = x^j)", "gcd(idempotent, x^p - 1)", "-")
        add(f"QR({p}) weight distribution", c["weight_distribution"], "A_w", "exhaustive enumeration", "-")
    for n in (31, 47):
        d, _, fr = S1N[n]
        add(f"m n={n}", d["m"], "-", "compute_sweep_row")
        add(f"N_eff n={n}", d["neff"], "bits", "impostor_pair_stats / neff_from_stats")
        for w in WGT:
            sfx = "" if w == "probe" else f" {w}"
            ww = d["weightings"][w]
            for k, u in (("beta", "bit-error rate"), ("t", "errors (delivered)"), ("parity", "bits"), ("k", "bits"),
                         ("decode_fail", "fraction"), ("residual", "bits")):
                add(f"{k} n={n}{sfx}", ww[k], u, "step 2's per-length computation (qr_codes.sweep_length)")
            b = d["bootstrap"][w]
            add(f"closure_pct n={n}{sfx}", b["closure_pct"], "%", "subject bootstrap, 1,000 replicates", f"projection 43; boot {542 + n}")
            add(f"closure_mixture n={n}{sfx}", [{"parity": x["parity"], "share": round(x["share_pct"], 1),
                                                 "pos": round(x["pct_positive_within"], 2), "res": round(x["residual_mean"], 4)}
                                                for x in b["branches"]], "json", "closure = sum_p share_p * P(res>0 | parity p)",
                f"projection 43; boot {542 + n}")
        t = d["weightings"]["probe"]["t"]
        add(f"FAR_sampled n={n}", fr["far"], "rate", "fuzzycommit.false_accept.run_sampled_far (10^6 trials)", f"far {342 + n}")
        add(f"FAR_sampled_accepts n={n}", fr["accepts"], "of 10^6", "fuzzycommit.false_accept.run_sampled_far", f"far {342 + n}")
        add(f"FAR_full_pair n={n}", d["exhaustive"]["far_pair_uniform_by_t"][t], "rate", "every ordered subject pair")
        add(f"FAR_full_comparison n={n}", d["exhaustive"]["far_comparison_uniform_by_t"][t], "rate", "every comparison")
        add(f"FAR_full_accepts n={n}", d["exhaustive"]["accepts_by_t"][t], "of 1681750", "fuzzycommit.false_accept.run_exhaustive_far")
        add(f"FAR_independent n={n}", d["far_analytical_by_t"][t], "rate", "compute_analytical_far")
    for p in PS:
        for fam, rows in QROWS[p]["families"].items():
            for w, r in rows.items():
                for k in ("parity", "t", "k", "residual", "decode_fail", "closure_pct", "far_sampled", "far_full_pair",
                          "far_full_comparison", "far_independent", "code_used"):
                    add(f"code n={p} [{fam}] {w} {k}", r[k], "-", "least parity under weighting (as in step 4)",
                        f"projection 43; boot {542 + p}; far {342 + p}")
                add(f"code n={p} [{fam}] {w} closure_mixture", [{"parity": x["parity"], "share": round(x["share_pct"], 1),
                                                                   "pos": round(x["pct_positive_within"], 2),
                                                                   "res": round(x["residual_mean"], 4)} for x in r["mixture"]],
                    "json", "closure = sum_p share_p * P(res>0 | parity p)", f"projection 43; boot {542 + p}")
    add("w2 rule closure threshold", RULE_CLOSURE, "%", "minimum over the three weightings", "-")
    add("w2 rule eligible candidates", R["rule"]["eligible"], "list", "closure_min >= 95 over the union", "-")
    add("w2 rule choice (sampled FAR)", by["far_sampled"], "id", "lowest FAR at t = max delivered t", "-")
    add("w2 rule choice (full pair FAR)", by["far_full_pair"], "id", "lowest FAR at t = max delivered t", "-")
    add("w2 rule non-BCH code beating n=15", beat, "list", "eligible, uses its code, FAR below n=15", "-")
    add("w2 rule triggered", bool(beat), "bool", "selection rule over all codes", "-")
    for c in near:
        add(f"w2 rule {c['id']} closure min", c["closure_min"], "%", "min over weightings", "-")
    add("crosscheck checks agree (codes)", f"{R['crosscheck']['agree']}/{R['crosscheck']['checks']}", "count", "analyses/crosscheck recomputation", "-")
    write_manifest(OUT / "part_qr_codes.csv", rows_m)
    print(f"{len(rows_m)} manifest rows; crosscheck {R['crosscheck']['agree']}/{R['crosscheck']['checks']}; {R['wall_seconds']}s")


if __name__ == "__main__":
    main()
