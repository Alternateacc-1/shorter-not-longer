"""Step 1: record the environment and compute the face embeddings.

Checks the model and the LFW folder by hash, then embeds the 5,425 images of the
311 subjects with at least six images (ArcFace R50, centre crop) and writes
results/pipeline/embeddings_cache.npz. It refuses to start while any embedding
file already exists, so every later step reads embeddings made in this run.

    python -m pipeline.step1_embed
"""
import hashlib
import time

from pipeline.provenance import (OUT, Log, dump, require_committed_code, sha256_file, versions)

T0 = time.time()
st = require_committed_code()
log = Log("step1_embed")

from fuzzycommit.common import (LFW_DIR, MODEL_PATH, DATA_DIR, RESULTS_DIR, GLOBAL_SEED,
                                MIN_IMAGES_PER_SUBJECT, get_W, build_subject_info,
                                subject_list_hash, probe_set_hash)
from fuzzycommit.embedding import load_subject_images, extract_embeddings
import numpy as np

# ---- no embedding file may exist anywhere a later step could read one
caches = [DATA_DIR / "embeddings_cache.npz", RESULTS_DIR / "embeddings_regenerated.npz"]
caches += list(OUT.glob("*.npz"))
present = [str(p) for p in caches if p.exists()]
assert not present, f"cache present, refusing to run: {present}"
log("no embedding cache present:", [str(p) for p in caches])

env = {"git": st, "versions": versions(), "lfw_dir": str(LFW_DIR), "model_path": str(MODEL_PATH)}
env["model_sha256"] = sha256_file(MODEL_PATH)
assert env["model_sha256"] == "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43", \
    "w600k_r50.onnx is not the expected file (see README)"
log("git", st["commit"], st["branch"], f"{len(st['deleted_derived'])} derived files deleted")
log("versions", env["versions"])

# ---- corpus hashes
t = time.time()
all_imgs = sorted(str(p.relative_to(LFW_DIR)).replace("\\", "/") for p in LFW_DIR.rglob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
env["lfw_tree_images"] = len(all_imgs)
env["lfw_manifest_sha256"] = hashlib.sha256("\n".join(all_imgs).encode("utf-8")).hexdigest()
subject_images = load_subject_images(LFW_DIR)
h = hashlib.sha256()
n_img = 0
for s in sorted(subject_images):
    for p in subject_images[s]:
        h.update(str(p.relative_to(LFW_DIR)).replace("\\", "/").encode("utf-8") + b"\0")
        h.update(bytes.fromhex(sha256_file(p)))
        n_img += 1
env["corpus_subjects"] = len(subject_images)
env["corpus_images"] = n_img
env["corpus_content_sha256"] = h.hexdigest()
env["subject_list_sha256"] = subject_list_hash(list(subject_images))
log(f"corpus: {env['lfw_tree_images']} images in tree, {len(subject_images)} subjects >= "
    f"{MIN_IMAGES_PER_SUBJECT} images, {n_img} images used ({time.time() - t:.1f}s)")
log("lfw_manifest_sha256", env["lfw_manifest_sha256"])
log("corpus_content_sha256", env["corpus_content_sha256"])
log("subject_list_sha256", env["subject_list_sha256"])

# ---- ONNX inference
t = time.time()
emb = extract_embeddings(subject_images, model_path=MODEL_PATH,
                         cache_path=OUT / "embeddings_cache.npz")
env["onnx_inference_seconds"] = round(time.time() - t, 1)
env["embeddings_npz_sha256"] = sha256_file(OUT / "embeddings_cache.npz")
log(f"ONNX inference {env['onnx_inference_seconds']}s -> {OUT / 'embeddings_cache.npz'}")

# ---- probe-set hash (n=24, seed-43 projection)
embf = {k: v for k, v in emb.items() if len(v) >= MIN_IMAGES_PER_SUBJECT}
mu = np.mean(np.concatenate(list(embf.values()), axis=0), axis=0)
cen = {s: v - mu for s, v in embf.items()}
info24, _ = build_subject_info(cen, get_W(), 24, sorted(cen))
env["probe_set_sha256"] = probe_set_hash(info24)
env["n_probes"] = int(sum(len(x[3]) for x in info24))
env["seeds"] = {"GLOBAL_SEED": GLOBAL_SEED, "projection": GLOBAL_SEED + 1}
env["wall_seconds"] = round(time.time() - T0, 1)
log("probe_set_sha256", env["probe_set_sha256"], "probes", env["n_probes"])
dump(env, "step1_embed.json")
log("done", env["wall_seconds"], "s")
