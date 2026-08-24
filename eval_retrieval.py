#!/usr/bin/env python3
"""
Retrieval accuracy harness - the metric that actually matches the product.

The app does not classify memes; it *ranks* them for a detected expression.
So the question worth measuring is: when the user looks sad, are the top-ranked
memes actually sad memes? That is Precision@K over a category query, graded
against the curated `Reactions/` folders.

This also makes per-category prototype bias mostly irrelevant (an additive
offset shared by a whole column cannot reorder that column), which is why this
scores so differently from a plain zero-shot classification eval.

Usage:
    python eval_retrieval.py
    python eval_retrieval.py --model laion/CLIP-ViT-L-14-laion2B-s32B-b82K
    python eval_retrieval.py --norm center --w-ocr 0.3 --detail
"""

import argparse
import json
import os
import sys

import numpy as np

import vision_taxonomy as tax
from analyze_memes import (ROOT, OCR_CACHE, build_prototypes, encode_images, encode_texts,
                           find_images, keyword_scores, load_clip, normalize_text)

# Which curated folders count as a correct hit for a query category.
# Categories with no clean folder home (pointing / facepalm / flexing) are not
# graded - there is no ground truth for them here.
CATEGORY_FOLDERS = {
    "happy":     {"Yes - Win - Love"},
    "sad":       {"Sad - Oof - Lose"},
    "crying":    {"Sad - Oof - Lose"},
    "angry":     {"Angry - Wicked", "No - Stop - Police"},
    "disgusted": {"Offend", "Cursed - NSFW", "No - Stop - Police"},
    "surprised": {"WTF"},
    "confused":  {"WTF", "Dumb - Genius"},
    "fearful":   {"Sweat - Run away"},
    "awkward":   {"Sweat - Run away", "Horny", "Cursed - NSFW"},
    "bored":     {"Humm - Not interesting - Boring"},
    "neutral":   {"Humm - Not interesting - Boring"},
    "love":      {"Horny", "Yes - Win - Love"},
    "laughing":  {"Funny - Not funny"},
    "mocking":   {"Attack - Mockery", "Offend"},
    "smug":      {"Yes - Win - Love", "Liar - Sauce", "Dumb - Genius"},
}


# --------------------------------------------------------------------------
# Column normalizers. All operate per-category across the meme pool.
# --------------------------------------------------------------------------

def norm_raw(s):
    return s.copy()


def norm_center(s):
    """Subtract each category's dataset mean - removes prototype bias only."""
    return s - s.mean(axis=0, keepdims=True)


def norm_zscore(s):
    return (s - s.mean(axis=0, keepdims=True)) / (s.std(axis=0, keepdims=True) + 1e-6)


def norm_rankint(s):
    from scipy.special import ndtri
    n = s.shape[0]
    out = np.empty_like(s, dtype=np.float32)
    for j in range(s.shape[1]):
        order = np.argsort(s[:, j])
        r = np.empty(n, dtype=np.float64)
        r[order] = np.arange(1, n + 1)
        out[:, j] = ndtri((r - 0.5) / n)
    return out


NORMS = {"raw": norm_raw, "center": norm_center, "zscore": norm_zscore, "rankint": norm_rankint}


def embed_for_model(model_id, paths, model, proc):
    """CLIP image embeddings for `model_id`, cached per model."""
    cache = os.path.join(ROOT, f".cache_clip_{model_id.split('/')[-1]}.npz")
    if os.path.exists(cache):
        try:
            z = np.load(cache, allow_pickle=True)
            if list(z["paths"]) == paths:
                return z["emb"]
        except Exception:
            pass
    emb, failed = encode_images(model, proc, paths)
    for rel, err in failed:
        print(f"  [warn] unreadable {rel}: {err}")
    np.savez_compressed(cache, emb=emb, paths=np.array(paths, dtype=object))
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="override CLIP model id")
    ap.add_argument("--norm", default="center", choices=list(NORMS))
    ap.add_argument("--w-vibe", type=float, default=0.25)
    ap.add_argument("--w-ocr", type=float, default=0.25)
    ap.add_argument("--w-kw", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=20, help="K for Precision@K")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    if args.model:
        import analyze_memes
        analyze_memes.MODEL_ID = args.model
    import analyze_memes
    model_id = analyze_memes.MODEL_ID

    paths_folders = find_images(ROOT)
    paths = [p for p, _ in paths_folders]
    folders = {p: f for p, f in paths_folders}

    model, proc, tok = load_clip()
    img_emb = embed_for_model(model_id, paths, model, proc)

    expr_p = build_prototypes(model, tok, tax.EXPRESSION_PROMPTS)
    vibe_p = build_prototypes(model, tok, tax.VIBE_PROMPTS)

    normfn = NORMS[args.norm]
    expr_s = normfn(img_emb @ expr_p.T)
    vibe_s = normfn(img_emb @ vibe_p.T)
    vision = (1 - args.w_vibe) * expr_s + args.w_vibe * vibe_s

    # OCR caption signal, scored against the vibe bank (captions carry meaning).
    ocr_map = {}
    if os.path.exists(OCR_CACHE):
        try:
            ocr_map = json.load(open(OCR_CACHE, encoding="utf-8"))
        except Exception:
            pass
    ocr_s = np.zeros_like(vision)
    idx = [i for i, p in enumerate(paths) if len(ocr_map.get(p, "")) >= 4]
    if idx and args.w_ocr > 0:
        te = encode_texts(model, tok, [ocr_map[paths[i]][:300] for i in idx])
        ocr_s[idx] = normfn(te @ vibe_p.T)

    kw_s = np.stack([keyword_scores(normalize_text(os.path.basename(p))) for p in paths])
    if args.norm != "raw" and kw_s.any():
        kw_s = kw_s * 2.0 - kw_s.mean(axis=0, keepdims=True)

    scores = vision + args.w_ocr * ocr_s + args.w_kw * kw_s

    # Grade only over labelled memes, so Precision@K has a clean denominator.
    labelled = np.array([bool(folders[p]) and folders[p] in tax.FOLDER_PRIORS for p in paths])
    pool = np.flatnonzero(labelled)
    pool_folders = [folders[paths[i]] for i in pool]

    print(f"\n  model={model_id.split('/')[-1]}  norm={args.norm}  w_vibe={args.w_vibe} "
          f"w_ocr={args.w_ocr} w_kw={args.w_kw}  K={args.k}  ocr_rows={len(idx)}")
    print(f"  retrieval pool: {len(pool)} labelled memes\n")

    rows, weighted = [], []
    for cat, good in CATEGORY_FOLDERS.items():
        col = scores[pool, tax.CAT_INDEX[cat]]
        top = np.argsort(-col)[:args.k]
        hits = sum(1 for t in top if pool_folders[t] in good)
        base = sum(1 for f in pool_folders if f in good) / len(pool)
        p_at_k = hits / args.k
        lift = p_at_k / base if base > 0 else 0.0
        rows.append((cat, p_at_k, base, lift))
        weighted.append(p_at_k)

    print(f"    {'category':11s} {'P@K':>7s} {'chance':>7s} {'lift':>6s}")
    print("    " + "-" * 34)
    for cat, p, b, l in sorted(rows, key=lambda r: -r[3]):
        print(f"    {cat:11s} {p*100:6.1f}% {b*100:6.1f}% {l:5.2f}x")
    mean_p = float(np.mean(weighted))
    mean_lift = float(np.mean([r[3] for r in rows]))
    print("    " + "-" * 34)
    print(f"    {'MEAN':11s} {mean_p*100:6.1f}% {'':7s} {mean_lift:5.2f}x")

    if args.detail:
        print("\n  top-5 retrieved per category:")
        for cat in CATEGORY_FOLDERS:
            col = scores[pool, tax.CAT_INDEX[cat]]
            top = np.argsort(-col)[:5]
            print(f"\n    [{cat}]")
            for t in top:
                p = paths[pool[t]]
                mark = "OK " if pool_folders[t] in CATEGORY_FOLDERS[cat] else "   "
                print(f"      {mark}{pool_folders[t][:28]:30s} {os.path.basename(p)[:44]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
