#!/usr/bin/env python3
"""
Accuracy harness for the MemeMatch vision pipeline.

The curated `Reactions/<name>` folders are human-labelled ground truth. This
scores the *vision-only* prediction against them with the folder prior held
out, so the number reflects what CLIP actually sees rather than the label
leaking back in.

Ambiguous folders ("Funny - Not funny", "Dumb - Genius") are excluded - a
wrong answer there is not necessarily wrong.

Usage:
    python eval_vision.py                 # score current settings
    python eval_vision.py --norm zscore   # compare a normalization scheme
"""

import argparse
import json
import os
import sys

import numpy as np

import vision_taxonomy as tax
from analyze_memes import (ROOT, EMBED_CACHE, OCR_CACHE, build_prototypes, encode_texts,
                           find_images, keyword_scores, load_clip, normalize_text)

# Folders whose emotional content is unambiguous enough to grade against.
# value = set of categories that count as correct.
GRADED = {
    "Angry - Wicked":                  {"angry", "mocking"},
    "Attack - Mockery":                {"mocking", "angry"},
    "Horny":                           {"love", "awkward"},
    "Humm - Not interesting - Boring": {"bored", "neutral"},
    "No - Stop - Police":              {"angry", "disgusted", "fearful"},
    "Offend":                          {"disgusted", "angry", "mocking"},
    "Sad - Oof - Lose":                {"sad", "crying", "facepalm"},
    "Sweat - Run away":                {"fearful", "awkward"},
    "WTF":                             {"surprised", "confused", "disgusted"},
    "Yes - Win - Love":                {"happy", "love", "smug", "laughing"},
}


def rank_int(sims):
    """Rank-based inverse-normal transform, applied per category column.

    Each column is replaced by the probit of its within-dataset percentile.
    Unlike a z-score this is immune to skew and heavy tails, which is what
    lets a category with a long right tail (e.g. "flexing" firing on every
    photo of a large person) stop dominating the ranking.
    """
    from scipy.special import ndtri
    n = sims.shape[0]
    out = np.empty_like(sims)
    for j in range(sims.shape[1]):
        order = np.argsort(sims[:, j])
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        out[:, j] = ndtri((ranks - 0.5) / n)          # Blom-style plotting position
    return out.astype(np.float32)


def zscore(sims):
    mu = sims.mean(axis=0, keepdims=True)
    sd = sims.std(axis=0, keepdims=True) + 1e-6
    return (sims - mu) / sd


def raw(sims):
    return sims * 100.0


NORMALIZERS = {"rankint": rank_int, "zscore": zscore, "raw": raw}


def softmax(x, temp):
    x = x / temp
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="rankint", choices=list(NORMALIZERS))
    ap.add_argument("--temp", type=float, default=0.55)
    ap.add_argument("--w-vibe", type=float, default=0.25, help="weight of vibe vs expression bank")
    ap.add_argument("--w-ocr", type=float, default=0.25, help="weight of OCR caption signal")
    ap.add_argument("--w-kw", type=float, default=0.10, help="weight of filename keywords")
    ap.add_argument("--per-folder", action="store_true", help="show a per-folder breakdown")
    args = ap.parse_args()

    paths_folders = find_images(ROOT)
    paths = [p for p, _ in paths_folders]
    folders = {p: f for p, f in paths_folders}

    if not os.path.exists(EMBED_CACHE):
        print("no embedding cache - run analyze_memes.py first")
        return 1
    z = np.load(EMBED_CACHE, allow_pickle=True)
    if list(z["paths"]) != paths:
        print("embedding cache is stale - rerun analyze_memes.py")
        return 1
    img_emb = z["emb"]

    model, _proc, tok = load_clip()
    expr_p = build_prototypes(model, tok, tax.EXPRESSION_PROMPTS)
    vibe_p = build_prototypes(model, tok, tax.VIBE_PROMPTS)

    norm = NORMALIZERS[args.norm]
    expr_s = softmax(norm(img_emb @ expr_p.T), args.temp)
    vibe_s = softmax(norm(img_emb @ vibe_p.T), args.temp)

    ocr_map = {}
    if os.path.exists(OCR_CACHE):
        try:
            ocr_map = json.load(open(OCR_CACHE, encoding="utf-8"))
        except Exception:
            pass
    ocr_s = np.zeros_like(expr_s)
    idx = [i for i, p in enumerate(paths) if len(ocr_map.get(p, "")) >= 4]
    if idx:
        te = encode_texts(model, tok, [ocr_map[paths[i]][:300] for i in idx])
        ocr_s[idx] = softmax(norm(te @ vibe_p.T), args.temp)

    kw_s = np.stack([keyword_scores(normalize_text(os.path.basename(p))) for p in paths])

    # Fuse WITHOUT the folder prior - that is the label we are grading against.
    vision = (1 - args.w_vibe) * expr_s + args.w_vibe * vibe_s
    w = np.zeros((len(paths), 3), dtype=np.float32)
    w[:, 0] = 1.0
    w[:, 1] = args.w_ocr * (ocr_s.sum(axis=1) > 0)
    w[:, 2] = args.w_kw * (kw_s.sum(axis=1) > 0)
    w /= w.sum(axis=1, keepdims=True)
    fused = w[:, 0:1] * vision + w[:, 1:2] * ocr_s + w[:, 2:3] * kw_s

    top1 = top3 = total = 0
    per_folder = {}
    for i, p in enumerate(paths):
        accept = GRADED.get(folders[p])
        if not accept:
            continue
        order = np.argsort(-fused[i])
        names = [tax.CATEGORIES[k] for k in order]
        hit1 = names[0] in accept
        hit3 = any(n in accept for n in names[:3])
        top1 += hit1
        top3 += hit3
        total += 1
        st = per_folder.setdefault(folders[p], [0, 0, 0])
        st[0] += hit1
        st[1] += hit3
        st[2] += 1

    print(f"\n  norm={args.norm} temp={args.temp} w_vibe={args.w_vibe} "
          f"w_ocr={args.w_ocr} w_kw={args.w_kw}  ocr_rows={len(idx)}")
    print(f"  graded {total} memes across {len(per_folder)} folders")
    print(f"  TOP-1 {top1/total*100:5.1f}%     TOP-3 {top3/total*100:5.1f}%")

    if args.per_folder:
        print()
        for f in sorted(per_folder, key=lambda k: -per_folder[k][2]):
            h1, h3, n = per_folder[f]
            print(f"    {f:34s} n={n:4d}  top1 {h1/n*100:5.1f}%  top3 {h3/n*100:5.1f}%")

    # Where does the whole dataset land? A healthy run is spread out.
    tops = [tax.CATEGORIES[k] for k in np.argmax(fused, axis=1)]
    print("\n  dominant-category spread (all 1301):")
    for c in tax.CATEGORIES:
        n = tops.count(c)
        print(f"    {c:11s} {n:4d}  {'#' * min(n // 4, 40)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
