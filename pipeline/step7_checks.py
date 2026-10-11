"""Step 7: run the two check scripts on this run's embeddings.

checks/verify_results.py (219 assertions) and checks/verify_short_lengths.py
(290 assertions) recompute results with the library and compare them with the
expected values in checks/expected/. Those files are restored from git first, so
the checks always test against the committed values. The embeddings from step 1
are copied to checks/expected/embeddings_cache.npz (not tracked) after their
SHA-256 is checked against step1_embed.json.

    python -m pipeline.step7_checks
"""
import json
import re
import shutil
import subprocess
import sys
import time

from pipeline.provenance import EMB_NPZ, OUT, OUT_MAIN, REPO, Log, dump, require_committed_code, sha256_file

EXPECTED = ["checks/expected/" + f for f in (
    "ablation_results.csv", "decode_failure_table_n24.csv", "impostor_neff.csv", "nsweep_results.csv",
    "seed_sensitivity.csv", "regen_results.json", "short_sweep_canonical.json")]


def main():
    st = require_committed_code()
    log = Log("step7_checks")
    env = json.load(open(OUT_MAIN / "step1_embed.json", encoding="utf-8"))
    assert sha256_file(EMB_NPZ) == env["embeddings_npz_sha256"]
    target = REPO / "checks" / "expected" / "embeddings_cache.npz"
    shutil.copyfile(EMB_NPZ, target)
    log(f"embeddings copied to {target} (sha256 {sha256_file(target)[:16]})")
    subprocess.run(["git", "-C", str(REPO), "checkout", "HEAD", "--", *EXPECTED], check=True)
    log("expected-value files restored from HEAD:", EXPECTED)
    R = {"git": st, "embeddings_sha256": sha256_file(target), "expected_value_files": EXPECTED}
    for name, script in (("verify_canonical", "checks/verify_short_lengths.py"), ("verify_py", "checks/verify_results.py")):
        t0 = time.time()
        p = subprocess.run([sys.executable, script], cwd=str(REPO), capture_output=True, text=True)
        (OUT / f"{name}.log").write_text(p.stdout + p.stderr, encoding="utf-8")
        m = re.findall(r"(\d+) PASS[, ]+(\d+) FAIL", p.stdout)
        R[name] = {"returncode": p.returncode, "pass": int(m[-1][0]) if m else None,
                   "fail": int(m[-1][1]) if m else None, "seconds": round(time.time() - t0, 1)}
        log(name, R[name])
    dump(R, "step7_checks.json")


if __name__ == "__main__":
    main()
