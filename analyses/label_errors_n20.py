"""Label errors at n = 20, where removing them matters: subject-weighted decode failure
at t=3 (parity 15) with all probes and without the mislabelled probes, the subjects whose
failure moves, and the n=20 closure (1,000 replicates, seed 562) for each variant.

    python -m analyses.label_errors_n20      (after analyses.label_errors)
"""
import json

import numpy as np

from analyses.helpers import out_dir, require_committed, write_manifest
from analyses import crosscheck as V

OUT = out_dir("supplementary")
from analyses.label_errors import DROP, variant


def main():
    st = require_committed()
    emb, cen = V.load(); W, _ = V.W_of(43)
    out = {"git": st}; rows = []
    for name, c in [("base", cen)] + [(k, variant(emb, cen, d, fixed_mean=False)) for k, d in DROP.items()]:
        S = V.Setup(cen if name == "base" else c, W); L = V.Length(S, 20)
        per = np.array([(h > 3).mean() for h in L.hd])
        draws = V.bootstrap_draws(20, S.NS); nes = np.array([L.neff(x) for x in draws])
        par = np.array([L.choose("subject", x)[0] for x in draws], float)
        clo = 100 * float(((20 - par > 0) & (nes - par > 0)).mean())
        shares = {int(q): round(100 * float((par == q).mean()), 1) for q in sorted(set(par))}
        top = sorted(((S.subs[i], float(per[i]), int(S.nprobe[i])) for i in range(S.NS) if per[i] > 0), key=lambda x: -x[1])[:8]
        out[name] = {"subject_fail_t3": L.fail(3, "subject"), "chosen": L.choose("subject"), "neff": L.neff(),
                     "closure_subject_pct": clo, "parity_shares_pct": shares, "top_subjects_failing_at_t3": top,
                     "schumacher_abbas": {s: float(per[S.subs.index(s)]) for s in ("Michael_Schumacher", "Mahmoud_Abbas")}}
        print(name, json.dumps(out[name], default=V._jsonable)[:600], flush=True)
        for k in ("subject_fail_t3", "closure_subject_pct"):
            rows.append({"quantity": f"w0 label {name} n=20 {k}", "value": repr(out[name][k]), "units": "-",
                         "script": "analyses/label_errors_n20.py", "function": "crosscheck.Length, n=20", "seeds": "projection 43; boot 562",
                         "commit": st["commit"][:12]})
    json.dump(out, open(OUT / "label_errors_n20.json", "w"), indent=1, default=V._jsonable)
    write_manifest(OUT / "part_label_errors_n20.csv", rows)


if __name__ == "__main__":
    main()
