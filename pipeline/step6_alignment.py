"""Step 6: what changes when the centre crop is replaced by 5-point landmark
alignment (bit-error rate, N_eff, residual and closure at six lengths, and the
LFW 6,000-pair accuracy).

Detector: SCRFD det_10g.onnx from the same InsightFace buffalo_l pack as the
recognition model (SHA-256 checked). Decoding follows InsightFace's SCRFD
(640x640 canvas, (x-127.5)/128 RGB, strides 8/16/32, 2 anchors, score 0.5, NMS 0.4;
the face maximising area - 2*offset^2 is kept). Alignment is the Umeyama
similarity transform to the ArcFace 112x112 template, warped bilinearly with
cv2.warpAffine's pixel convention reproduced in PIL (border 0). The aligned crop
then goes through the same normalisation and the same w600k_r50 as step 1.
Images where no face is found keep the centre crop.

Runs on all 5,425 corpus images and on the 7,701 images of the LFW 6,000-pair
protocol. Needs models/det_10g.onnx.

    python -m pipeline.step6_alignment
"""
import csv
import time

from pipeline.provenance import OUT, EMB_NPZ, Log, dump, require_committed_code, sha256_file

DET_SHA = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
ARCFACE_DST = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
               [41.5493, 92.3655], [70.7299, 92.2041]]
LENGTHS = [9, 11, 15, 16, 24, 32]


def main():
    st = require_committed_code()
    log = Log("step6_alignment")
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from scipy.stats import beta as bdist
    from fuzzycommit.common import (LFW_DIR, MODEL_PATH, MODEL_DIR, CENTER_CROP_SIZE, MIN_IMAGES_PER_SUBJECT,
                            get_W, build_subject_info, select_capacity, bch_parity, bch_true_t,
                            compute_bch_m, DECODE_FAILURE_TARGET as TARGET)
    from fuzzycommit.embedding import load_subject_images
    from fuzzycommit.false_accept import compute_neff
    from fuzzycommit.entropy import enrolled_bitstrings

    det_path = MODEL_DIR / "det_10g.onnx"
    assert sha256_file(det_path) == DET_SHA, "detector hash"
    # Two sessions in one process: with the default spinning thread pools each session's
    # idle workers spin and starve the other (measured 0.80 s/image vs 0.12 without).
    # Scheduling only; the bit-identity check below confirms outputs are unchanged.
    so = ort.SessionOptions()
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    det = ort.InferenceSession(str(det_path), sess_options=so, providers=["CPUExecutionProvider"])
    rec = ort.InferenceSession(str(MODEL_PATH), sess_options=so, providers=["CPUExecutionProvider"])
    din, rin, rout = det.get_inputs()[0].name, rec.get_inputs()[0].name, rec.get_outputs()[0].name
    dst = np.array(ARCFACE_DST, dtype=np.float64)

    def detect(img):
        w, h = img.size
        if h / w > 1.0:
            nh = 640; nw = int(nh / (h / w))
        else:
            nw = 640; nh = int(nw * (h / w))
        scale = nh / h
        canvas = np.zeros((640, 640, 3), np.float32)
        canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), np.float32)
        blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = det.run(None, {din: blob})
        boxes, kpss, scores = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            sc = outs[i][:, 0]; bb = outs[i + 3] * stride; kp = outs[i + 6] * stride
            g = 640 // stride
            ctr = (np.stack(np.mgrid[:g, :g][::-1], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
            ctr = np.stack([ctr] * 2, axis=1).reshape(-1, 2)
            keep = np.where(sc >= 0.5)[0]
            if len(keep) == 0:
                continue
            c = ctr[keep]; b = bb[keep]; k = kp[keep]
            boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], 1))
            kpss.append(np.stack([c[:, 0:1] + k[:, 0::2], c[:, 1:2] + k[:, 1::2]], -1))
            scores.append(sc[keep])
        if not scores:
            return None
        boxes = np.concatenate(boxes) / scale; kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)
        order = scores.argsort()[::-1]; keep = []
        x1, y1, x2, y2 = boxes.T; area = (x2 - x1 + 1) * (y2 - y1 + 1)
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
            order = order[1:][inter / (area[i] + area[order[1:]] - inter) <= 0.4]
        keep = np.array(keep)
        bx = boxes[keep]
        ar = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
        off = np.stack([(bx[:, 0] + bx[:, 2]) / 2 - w / 2, (bx[:, 1] + bx[:, 3]) / 2 - h / 2], 1)
        j = keep[int(np.argmax(ar - 2.0 * (off ** 2).sum(1)))]
        return kpss[j], float(scores[j])

    def umeyama(src):
        n, dim = src.shape
        sm, dm = src.mean(0), dst.mean(0)
        sd, dd = src - sm, dst - dm
        A = dd.T @ sd / n
        d = np.ones(dim)
        if np.linalg.det(A) < 0:
            d[-1] = -1
        U, S, Vt = np.linalg.svd(A)
        R = U @ np.diag(d) @ Vt
        scale = (S @ d) / sd.var(0).sum()
        M = np.zeros((2, 3)); M[:, :2] = scale * R; M[:, 2] = dm - scale * R @ sm
        return M

    def warp(img, M):
        Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
        A, b = Mi[:, :2], Mi[:, 2]
        c = b + 0.5 - A @ np.array([0.5, 0.5])           # cv2 pixel-centre convention in PIL
        return img.transform((112, 112), Image.AFFINE, (A[0, 0], A[0, 1], c[0], A[1, 0], A[1, 1], c[1]),
                             resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    def center_crop(img):                                 # same as fuzzycommit/embedding.py
        w, h = img.size
        if CENTER_CROP_SIZE and min(w, h) > CENTER_CROP_SIZE:
            l, t = (w - CENTER_CROP_SIZE) // 2, (h - CENTER_CROP_SIZE) // 2
            img = img.crop((l, t, l + CENTER_CROP_SIZE, t + CENTER_CROP_SIZE))
        return img.resize((112, 112), Image.BILINEAR)

    def embed(img112):
        arr = ((np.asarray(img112, np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        v = rec.run([rout], {rin: arr})[0][0]
        return v / (np.linalg.norm(v) + 1e-10)

    cache = {}
    stats = {"detected": 0, "fallback_center_crop": 0, "det_scores": []}

    def aligned_embedding(path):
        if path in cache:
            return cache[path]
        img = Image.open(path).convert("RGB")
        r = detect(img)
        if r is None:
            stats["fallback_center_crop"] += 1
            v = embed(center_crop(img))
        else:
            stats["detected"] += 1; stats["det_scores"].append(r[1])
            v = embed(warp(img, umeyama(r[0].astype(np.float64))))
        cache[path] = v
        return v

    # ---- sanity: the centre-crop path here reproduces the regenerated embeddings bit-for-bit
    ref = np.load(EMB_NPZ, allow_pickle=True)["embeddings"].item()
    subject_images = load_subject_images(LFW_DIR)
    s0 = sorted(subject_images)[0]
    chk = np.array([embed(center_crop(Image.open(p).convert("RGB"))) for p in subject_images[s0][:3]])
    assert np.array_equal(chk, ref[s0][:3]), "centre-crop path here differs from fuzzycommit/embedding.py"
    log("centre-crop path reproduces the regenerated embeddings exactly (3 images)")
    import os
    if os.environ.get("QUICK_RUN"):                      # quick look at three subjects, then stop
        for s in sorted(subject_images)[:3]:
            va = np.array([aligned_embedding(p) for p in subject_images[s][:6]])
            vc = ref[s][:6]
            log(s, "cos(aligned, centre-crop) per image", np.round(np.sum(va * vc, 1), 3).tolist(),
                "| genuine cos aligned", np.round(va[1:] @ va[0], 3).tolist(),
                "| centre-crop", np.round(vc[1:] @ vc[0], 3).tolist())
        log("det scores", np.round(stats["det_scores"], 3).tolist(), "fallbacks", stats["fallback_center_crop"])
        return

    # ---- phase 1: aligned embeddings for the full corpus
    t0 = time.time(); done = 0
    aligned = {}
    for s in sorted(subject_images):
        aligned[s] = np.array([aligned_embedding(p) for p in subject_images[s]])
        done += len(subject_images[s])
        if done % 500 < len(subject_images[s]):
            log(f"  corpus {done}/5425 ({time.time() - t0:.0f}s), fallbacks {stats['fallback_center_crop']}")
    np.savez(OUT / "aligned_embeddings.npz", embeddings=aligned)

    def envelope(embd):
        e = {k: v for k, v in embd.items() if len(v) >= MIN_IMAGES_PER_SUBJECT}
        mu = np.concatenate(list(e.values())).mean(0)
        cen = {k: v - mu for k, v in e.items()}
        subs = sorted(cen); W = get_W()
        TB = enrolled_bitstrings(cen, W).astype(np.uint8)
        out = {}
        gen = [float(np.mean(e[s][5:] @ e[s][:5].mean(0) / np.linalg.norm(e[s][:5].mean(0)))) for s in subs]
        for n in LENGTHS:
            info, sels = build_subject_info(cen, W, n, subs)
            hds = [x[3] for x in info]; allh = np.concatenate(hds); m = compute_bch_m(n)
            _, p_pw, t_pw = select_capacity(allh, n)
            sw = None
            for t in range(n + 1):
                ta = bch_true_t(m, t)
                if float(np.mean([np.mean(h > ta) for h in hds])) < TARGET:
                    sw = (bch_parity(m, t), ta); break
            ne = float(compute_neff(cen, W, n, info))
            ks = np.array([np.unique(TB[:, sel], axis=0, return_counts=True)[1].max() for _, sel in sels])
            fl = float(np.mean(-np.log2(bdist.ppf(0.99, ks + 1, len(subs) - ks))))
            out[n] = {"beta_probe": float(allh.mean() / n),
                      "beta_subject": float(np.mean([h.mean() / n for h in hds])),
                      "t_probe": int(t_pw), "parity_probe": int(p_pw), "t_subject": int(sw[1]),
                      "parity_subject": int(sw[0]), "neff": ne, "residual_probe": ne - p_pw,
                      "residual_subject": ne - sw[0], "mcv_floor_mean": fl}
        return out, float(np.mean(gen))

    base, gen_b = envelope(ref)
    alig, gen_a = envelope(aligned)
    R = {"git": st, "detector_sha256": DET_SHA, "lengths": LENGTHS,
         "corpus_detected": stats["detected"], "corpus_fallback_center_crop": stats["fallback_center_crop"],
         "det_score_min": float(np.min(stats["det_scores"])), "center_crop": base, "aligned": alig,
         "genuine_cos_to_enrol_mean": {"center_crop": gen_b, "aligned": gen_a},
         "corpus_seconds": round(time.time() - t0, 1)}
    for n in LENGTHS:
        log(f"n={n}: beta {base[n]['beta_probe']:.4f} -> {alig[n]['beta_probe']:.4f} | N_eff "
            f"{base[n]['neff']:.3f} -> {alig[n]['neff']:.3f} | parity pw {base[n]['parity_probe']}->"
            f"{alig[n]['parity_probe']} sw {base[n]['parity_subject']}->{alig[n]['parity_subject']} | "
            f"res sw {base[n]['residual_subject']:+.2f} -> {alig[n]['residual_subject']:+.2f}")
    dump(R, "step6_alignment.json")

    # ---- phase 2: LFW 6,000-pair accuracy with alignment (same protocol as step 3)
    pairs_csv = LFW_DIR.parent.parent / "pairs.csv"
    img = lambda nm, k: LFW_DIR / nm / f"{nm}_{int(k):04d}.jpg"
    pairs = []
    for r in list(csv.reader(open(pairs_csv)))[1:]:
        r = (r + ["", "", "", ""])[:4]
        pairs.append((img(r[0], r[1]), img(r[0], r[2]), 1) if r[3].strip() == ""
                     else (img(r[0], r[1]), img(r[2], r[3]), 0))
    t1 = time.time()
    a = np.array([aligned_embedding(p) for p, _, _ in pairs]); b = np.array([aligned_embedding(q) for _, q, _ in pairs])
    cos = np.sum(a * b, 1); y = np.array([l for _, _, l in pairs]); accs = []
    for f in range(10):
        te = np.zeros(6000, bool); te[600 * f:600 * (f + 1)] = True; tr = ~te; best = (0, 0)
        for th in np.unique(cos[tr]):
            acc = np.mean((cos[tr] >= th).astype(int) == y[tr])
            if acc > best[0]:
                best = (acc, th)
        accs.append(np.mean((cos[te] >= best[1]).astype(int) == y[te]))
    R["lfw_pairs_aligned_accuracy_pct"] = 100 * float(np.mean(accs))
    R["lfw_pairs_aligned_sd_pct"] = 100 * float(np.std(accs))
    R["lfw_total_detected"] = stats["detected"]; R["lfw_total_fallback"] = stats["fallback_center_crop"]
    R["lfw_seconds"] = round(time.time() - t1, 1)
    dump(R, "step6_alignment.json")
    log(f"LFW 6,000 pairs, aligned: {R['lfw_pairs_aligned_accuracy_pct']:.3f}% +/- {R['lfw_pairs_aligned_sd_pct']:.3f}; "
        f"fallbacks {stats['fallback_center_crop']}")


if __name__ == "__main__":
    main()
