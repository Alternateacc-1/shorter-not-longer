"""Landmark alignment, part 1: a second implementation of the alignment, written from
the InsightFace SCRFD / ArcFace conventions without importing pipeline/step6_alignment.py.

 a. a seeded sample of 300 corpus images, plus every image where step 6 found no face,
    re-aligned here and compared with results/pipeline/aligned_embeddings.npz;
 b. all 7,701 images of the LFW 6,000-pair protocol, aligned here (step 6 does not
    save them), for the 10-fold accuracy check in analyses.alignment_results.
Detection sees only the image file: no label, pair or identity information.

    python -m analyses.realign_faces        (after step 6; needs both models)
"""
import csv
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from analyses import crosscheck as V

DET_SHA = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
REC_SHA = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]])
LFW = V._common.LFW_DIR   # the configured LFW tree (LFW_DIR, paths.conf, or the kagglehub default)
MODELS = V.REPO / "models"


class Aligner:
    def __init__(self):
        assert V.sha256(MODELS / "det_10g.onnx") == DET_SHA and V.sha256(MODELS / "w600k_r50.onnx") == REC_SHA
        so = ort.SessionOptions(); so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.det = ort.InferenceSession(str(MODELS / "det_10g.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(str(MODELS / "w600k_r50.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.di = self.det.get_inputs()[0].name; self.ri = self.rec.get_inputs()[0].name
        self.ro = self.rec.get_outputs()[0].name

    def landmarks(self, img):
        w, h = img.size
        if h / w > 1:
            nh, nw = 640, int(640 / (h / w))
        else:
            nw, nh = 640, int(640 * (h / w))
        scale = nh / h
        x = np.zeros((640, 640, 3), np.float32)
        x[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), np.float32)
        out = self.det.run(None, {self.di: ((x - 127.5) / 128.0).transpose(2, 0, 1)[None]})
        B, K, P = [], [], []
        for s_i, stride in enumerate((8, 16, 32)):
            g = 640 // stride
            yy, xx = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
            cen = np.repeat(np.stack([xx.ravel(), yy.ravel()], 1) * stride, 2, axis=0).astype(np.float32)
            sc = out[s_i][:, 0]; keep = sc >= 0.5
            if not keep.any():
                continue
            d = out[s_i + 3][keep] * stride; k = out[s_i + 6][keep] * stride; c = cen[keep]
            B.append(np.c_[c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]])
            K.append(np.stack([c[:, 0:1] + k[:, 0::2], c[:, 1:2] + k[:, 1::2]], axis=2))
            P.append(sc[keep])
        if not P:
            return None
        B = np.concatenate(B) / scale; K = np.concatenate(K) / scale; P = np.concatenate(P)
        idx = list(np.argsort(-P)); kept = []
        area = (B[:, 2] - B[:, 0] + 1) * (B[:, 3] - B[:, 1] + 1)
        while idx:
            i = idx.pop(0); kept.append(i)
            rest = np.array(idx, dtype=int)
            if not len(rest):
                break
            iw = np.maximum(0, np.minimum(B[i, 2], B[rest, 2]) - np.maximum(B[i, 0], B[rest, 0]) + 1)
            ih = np.maximum(0, np.minimum(B[i, 3], B[rest, 3]) - np.maximum(B[i, 1], B[rest, 1]) + 1)
            inter = iw * ih
            idx = [j for j, o in zip(rest, inter / (area[i] + area[rest] - inter)) if o <= 0.4]
        kept = np.array(kept)
        b = B[kept]
        val = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - 2.0 * (((b[:, 0] + b[:, 2]) / 2 - w / 2) ** 2
                                                                + ((b[:, 1] + b[:, 3]) / 2 - h / 2) ** 2)
        return K[kept[int(np.argmax(val))]].astype(np.float64)

    @staticmethod
    def similarity(src):
        """Least-squares similarity (Umeyama 1991) mapping src onto the ArcFace template."""
        ms, md = src.mean(0), TEMPLATE.mean(0)
        a, b = src - ms, TEMPLATE - md
        cov = b.T @ a / len(src)
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(2)
        if np.linalg.det(cov) < 0:
            S[1, 1] = -1
        R = U @ S @ Vt
        s = np.trace(np.diag(D) @ S) / a.var(0).sum()
        return np.c_[s * R, md - s * R @ ms]

    @staticmethod
    def warp(img, M):
        inv = np.linalg.inv(np.r_[M, [[0, 0, 1]]])
        A, t = inv[:2, :2], inv[:2, 2]
        off = t + 0.5 - A @ np.array([0.5, 0.5])      # OpenCV pixel-centre convention, in PIL
        return img.transform((112, 112), Image.AFFINE, (A[0, 0], A[0, 1], off[0], A[1, 0], A[1, 1], off[1]),
                             resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    @staticmethod
    def centre(img):
        w, h = img.size
        if min(w, h) > 150:
            l, t = (w - 150) // 2, (h - 150) // 2
            img = img.crop((l, t, l + 150, t + 150))
        return img.resize((112, 112), Image.BILINEAR)

    def embed(self, im112):
        a = ((np.asarray(im112, np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        v = self.rec.run([self.ro], {self.ri: a})[0][0]
        return v / (np.linalg.norm(v) + 1e-10)

    def aligned(self, path):
        img = Image.open(path).convert("RGB")
        k = self.landmarks(img)
        if k is None:
            return self.embed(self.centre(img)), False
        return self.embed(self.warp(img, self.similarity(k))), True


def main():
    t0 = time.time(); A = Aligner()
    # ---- corpus: which stored aligned vectors are centre-crop fallbacks (bitwise equal)?
    ref_al = np.load(V.UNDER_TEST / "aligned_embeddings.npz", allow_pickle=True)["embeddings"].item()
    ref_cc = np.load(V.EMB, allow_pickle=True)["embeddings"].item()
    subs = sorted(ref_al)
    paths = {s: sorted(p for p in (LFW / s).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) for s in subs}
    items = [(s, i) for s in subs for i in range(len(ref_al[s]))]
    fallbacks = [(s, i) for s, i in items if np.array_equal(ref_al[s][i], ref_cc[s][i])]
    rng = np.random.RandomState(20261006)
    sample = [items[j] for j in rng.choice(len(items), 300, replace=False)]
    corpus = []
    for s, i in sorted(set(sample) | set(fallbacks)):
        v, det = A.aligned(paths[s][i])
        corpus.append({"subject": s, "index": i, "file": paths[s][i].name, "detected_here": det,
                       "stored_is_fallback": (s, i) in fallbacks,
                       "max_abs_diff": float(np.abs(v - ref_al[s][i]).max()), "cos": float(v @ ref_al[s][i])})
    print(f"corpus check: {len(corpus)} images ({time.time() - t0:.0f}s)", flush=True)
    # ---- LFW pairs
    D = LFW.parent.parent
    pairs = []
    for r in list(csv.reader(open(D / "pairs.csv")))[1:]:
        r = (r + ["", "", "", ""])[:4]
        f = lambda nm, k: LFW / nm / f"{nm}_{int(k):04d}.jpg"
        pairs.append((f(r[0], r[1]), f(r[0], r[2]), 1) if r[3].strip() == "" else (f(r[0], r[1]), f(r[2], r[3]), 0))
    uniq = sorted({p for a, b, _ in pairs for p in (a, b)})
    emb = {}; nodet = []
    for j, p in enumerate(uniq):
        emb[str(p)], det = A.aligned(p)
        if not det:
            nodet.append(str(p.relative_to(LFW)))
        if (j + 1) % 1000 == 0:
            print(f"  pairs {j + 1}/{len(uniq)} ({time.time() - t0:.0f}s)", flush=True)
    keys = sorted(emb)
    np.savez(V.OUTV / "aligned_pairs_independent.npz", keys=np.array(keys), emb=np.array([emb[k] for k in keys]))
    V.json.dump({"git": V.git_state(), "fallbacks_stored": [{"subject": s, "index": i, "file": paths[s][i].name}
                                                             for s, i in fallbacks],
                 "corpus_recheck": corpus, "pairs_unique": len(uniq), "pairs_no_detection": nodet,
                 "seconds": round(time.time() - t0, 1)},
                open(V.OUTV / "realign_faces.json", "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s); fallbacks stored {len(fallbacks)}; pair images without detection {len(nodet)}")


if __name__ == "__main__":
    main()
