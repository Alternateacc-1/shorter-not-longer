#!/usr/bin/env python3
"""
ArcFace R50 (ONNX) embeddings for the LFW images: centre crop, resize to
112x112, normalise, one 512-d vector per image.
"""

import time
from collections import defaultdict

import numpy as np
from PIL import Image

from .common import (
    LFW_DIR, MODEL_PATH, CENTER_CROP_SIZE,
    MIN_IMAGES_PER_SUBJECT, DATA_DIR, log,
)


def load_subject_images(lfw_dir=None):
    """Load subject→image_paths mapping, filtered to >= MIN_IMAGES_PER_SUBJECT."""
    if lfw_dir is None:
        lfw_dir = LFW_DIR
    subjects = defaultdict(list)
    for person_dir in sorted(lfw_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted([p for p in person_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if len(imgs) >= MIN_IMAGES_PER_SUBJECT:
            subjects[person_dir.name] = imgs
    return dict(subjects)


def extract_embeddings(subject_images, model_path=None, cache_path=None):
    """
    Run ArcFace R50 ONNX inference over all images.

    Args:
        subject_images: dict {subject_name: [image_paths]}
        model_path: path to w600k_r50.onnx
        cache_path: output path for embeddings_cache.npz

    Returns:
        embeddings: dict {subject_name: (n_images, 512) array}
    """
    import onnxruntime as ort

    if model_path is None:
        model_path = MODEL_PATH
    if cache_path is None:
        cache_path = DATA_DIR / "embeddings_cache.npz"

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    log(f"  ONNX session ready. Input: {input_name}, Output: {output_name}")
    log(f"  Input shape expected: {sess.get_inputs()[0].shape}")

    embeddings = {}
    total_images = sum(len(v) for v in subject_images.values())
    processed = 0
    t0 = time.time()

    for subj, img_paths in subject_images.items():
        vecs = []
        for img_path in img_paths:
            # Preprocess: center crop, resize 112x112, RGB, (pixel-127.5)/127.5, HWC->CHW
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if CENTER_CROP_SIZE and min(w, h) > CENTER_CROP_SIZE:
                left = (w - CENTER_CROP_SIZE) // 2
                top = (h - CENTER_CROP_SIZE) // 2
                img = img.crop((left, top, left + CENTER_CROP_SIZE, top + CENTER_CROP_SIZE))
            img = img.resize((112, 112), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            arr = (arr - 127.5) / 127.5
            arr = arr.transpose(2, 0, 1)
            arr = arr[np.newaxis, :]

            out = sess.run([output_name], {input_name: arr})[0]
            vec = out[0]
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            vecs.append(vec)
            processed += 1

        embeddings[subj] = np.array(vecs)

        if processed % 100 == 0 or subj == list(subject_images.keys())[-1]:
            elapsed = time.time() - t0
            log(f"    {processed}/{total_images} images "
                f"({processed / total_images * 100:.1f}%) -- {elapsed:.1f}s")

    # Cache
    np.savez(cache_path, embeddings=embeddings)
    log(f"  Embeddings cached to {cache_path}")

    return embeddings
