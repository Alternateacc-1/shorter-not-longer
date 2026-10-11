"""Values the paper's tables need that the pipeline manifest does not hold.

  centre-crop FAR at the SUBJECT-weighted delivered t, n = 15, 24, 32 (sampled, seed 342+n;
  full set pair- and comparison-uniform), cross-checked against manifest.csv where it has them;
  throughput for each decode-failure target: the deployed code is the subject-weighted choice
  at the band-edge length; f is its per-attempt (probe-weighted) decode failure; 63(1 - f).

    python -m analyses.table_values
Writes results/supplementary/part_table_values.csv."""
from analyses.helpers import out_dir, require_committed, write_manifest
from analyses import crosscheck as V

OUT = out_dir("supplementary")


def main():
    st = require_committed()
    M = V.manifest(); emb, cen = V.load(); W, _ = V.W_of(43); S = V.Setup(cen, W)
    rows = []

    def add(q, v, u, f, seeds="projection 43"):
        rows.append({"quantity": q, "value": repr(v) if isinstance(v, float) else v, "units": u,
                     "script": "analyses/table_values.py", "function": f, "seeds": seeds,
                     "commit": st["commit"][:12]})
    L = {n: V.Length(S, n) for n in (15, 24, 32)}
    for n, Ln in L.items():
        p, t = Ln.choose("subject")
        pair, comp, acc, ncmp = Ln.far_full(t)
        hist = Ln.sampled_hist(42 + 300 + n)
        add(f"w0 centre n={n} subject t", int(t), "errors", "least parity meeting 1.5%, subject weighting")
        add(f"w0 centre n={n} subject FAR_full_pair", pair, "rate", f"all {ncmp:,} impostor comparisons, pair-uniform")
        add(f"w0 centre n={n} subject FAR_full_comparison", comp, "rate", "comparison-uniform")
        add(f"w0 centre n={n} subject FAR_sampled", float(hist[:t + 1].sum()) / 1e6, "rate", "10^6 sampled trials", f"far {342 + n}")
        ref = (M.get(f"code n={n} [shortened BCH (paper)] subject far_full_pair") if n != 15 else M["FAR_full_pair n=15"])
        assert abs(ref - pair) <= 1e-12 * max(1, ref), (n, ref, pair)     # same quantity where the manifest has it
    for tg in ("0.01", "0.015", "0.02", "0.03"):
        n = int(M[f"throughput target={tg} chosen n (longest subject-positive)"]); t = int(M[f"throughput target={tg} chosen t"])
        Ln = L[n] if n in L else V.Length(S, n)
        f = Ln.fail(t, "probe")
        add(f"w0 throughput target={tg} deployed probe decode_fail", f, "fraction",
            f"probe-weighted failure of the subject-weighted code (n={n}, t={t})")
        add(f"w0 throughput target={tg} ppm", 63 * (1 - f), "people/min", "63 x (1 - f), three lanes, serial model")
        add(f"w0 throughput target={tg} floor ppm", 63 * (1 - float(tg)), "people/min", "63 x (1 - target)")
    import json
    import re
    lengths = sorted(int(m.group(1)) for k in M for m in [re.fullmatch(r"m n=(\d+)", k)] if m)
    rows.append({"quantity": "w0 studied lengths", "value": json.dumps(lengths, separators=(",", ":")), "units": "list",
                 "script": "analyses/table_values.py", "function": "every n with an 'm n=' row in manifest.csv",
                 "seeds": "-", "commit": st["commit"][:12]})
    seeds = sorted(int(m.group(1)) for k in M for m in [re.fullmatch(r"seed(\d+) n=15 beta", k)] if m)
    rows.append({"quantity": "w0 projection seeds", "value": json.dumps(seeds, separators=(",", ":")), "units": "list",
                 "script": "analyses/table_values.py", "function": "every seed with a 'seedNNN n=15 beta' row in manifest.csv",
                 "seeds": "-", "commit": st["commit"][:12]})
    write_manifest(OUT / "part_table_values.csv", rows)
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
