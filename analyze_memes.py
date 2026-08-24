#!/usr/bin/env python3
"""
MemeMatch vision pipeline.

Runs a real vision model (CLIP ViT-B/32, LAION-2B weights) over every meme in
this folder and produces `meme_catalog.json` + `meme_embeddings.bin`, which the
browser app uses to match your live facial expression to a meme.

Four independent signals are fused per image:

  1. CLIP zero-shot on the *pixels*, against a prompt-ensembled bank describing
     facial/body expressions (the dominant signal - the runtime query is a face).
  2. CLIP zero-shot on the *caption text*, read off the image with OCR.
  3. The human-curated `Reactions/<category>` folder the meme lives in.
  4. Keywords in the filename.

Raw CLIP cosine similarities are badly calibrated across categories (some
prompt sets simply sit closer to the image manifold than others), so each
category column is z-scored across the whole dataset before softmax. That
single step is what turns "everything is 'neutral'" into usable rankings.

Usage:
    python analyze_memes.py              # full run (CLIP + OCR), resumable
    python analyze_memes.py --skip-ocr   # ~1 min, slightly less accurate
    python analyze_memes.py --force      # ignore caches, recompute everything
    python analyze_memes.py --device cpu # force CPU (default: CUDA if present)

CLIP runs on the GPU in fp16 whenever a CUDA-enabled torch is installed, and
the stages that can only run on the CPU (OCR, face scan) are held to half
the cores so the machine stays usable during a full run.
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata

import numpy as np

import vision_taxonomy as tax

ROOT = os.path.dirname(os.path.abspath(__file__))
CATALOG_OUT = os.path.join(ROOT, "meme_catalog.json")
EMBED_OUT = os.path.join(ROOT, "meme_embeddings.bin")
OCR_CACHE = os.path.join(ROOT, ".cache_ocr.json")
EMBED_CACHE = os.path.join(ROOT, ".cache_clip.npz")
IGNORE_FILE = os.path.join(ROOT, ".memeignore")

MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Fusion weights. Vision dominates; the rest nudge.
W_EXPRESSION = 0.50   # CLIP on pixels, expression prompts
W_VIBE = 0.16         # CLIP on pixels, meme-vibe prompts
W_OCR = 0.16          # CLIP text encoder on the OCR'd caption
W_FOLDER = 0.12       # curated Reactions/ folder
W_KEYWORD = 0.06      # filename keywords

SOFTMAX_TEMP = 0.55   # lower = peakier category distributions


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE_REQUEST = "auto"   # "auto" | "cuda" | "cpu"; also read from MEMEMATCH_DEVICE
_DEVICE = None            # resolved torch.device
_DTYPE = None             # fp16 on CUDA, fp32 on CPU
_CPU_THREADS = 0          # 0 = auto (half the cores)


def cpu_threads():
    """Cores the CPU-bound stages may use, leaving headroom for the desktop.

    Half the machine by default: OCR is the only stage that still needs the
    CPU once CLIP is on the GPU, and it is throughput-bound work nobody is
    waiting on interactively. Raise it with --cpu-threads for an unattended run.
    """
    if _CPU_THREADS > 0:
        return _CPU_THREADS
    return max(1, (os.cpu_count() or 4) // 2)


def resolve_device():
    """Pick the compute device once, and report it.

    fp16 on CUDA is safe here: the pipeline only ever consumes cosine
    similarities between L2-normalized vectors, and half precision moves those
    by ~1e-3 - orders of magnitude below the gaps that decide a ranking.
    """
    global _DEVICE, _DTYPE
    if _DEVICE is not None:
        return _DEVICE, _DTYPE

    import torch
    want = os.environ.get("MEMEMATCH_DEVICE", DEVICE_REQUEST).strip().lower() or "auto"
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want.startswith("cuda") and not torch.cuda.is_available():
        log(f"  [warn] CUDA requested but torch {torch.__version__} cannot see a GPU"
            " - falling back to CPU")
        want = "cpu"

    _DEVICE = torch.device(want)
    _DTYPE = torch.float16 if _DEVICE.type == "cuda" else torch.float32
    if _DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        i = _DEVICE.index or 0
        free, total = torch.cuda.mem_get_info(i)
        log(f"  device: cuda:{i} {torch.cuda.get_device_name(i)} "
            f"({total / 2**30:.1f} GB, {free / 2**30:.1f} GB free), fp16")
    else:
        torch.set_num_threads(cpu_threads())
        log(f"  device: cpu ({torch.get_num_threads()} of {os.cpu_count()} threads)")
    return _DEVICE, _DTYPE


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def load_ignores():
    """Patterns from .memeignore: one fnmatch glob per line, # for comments.

    Matched against the path relative to this folder and against the bare
    filename, so both of these drop the same meme:

        Various & templates (no HD)/Reactions/Humm - Not interesting - Boring/Harold.jpg
        Harold.jpg
    """
    if not os.path.exists(IGNORE_FILE):
        return []
    with open(IGNORE_FILE, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def find_images(root):
    """Walk the meme folder, returning (relative_path, folder_name) pairs.

    Anything listed in .memeignore is skipped, which is how a meme leaves the
    app without leaving the disk.
    """
    import fnmatch
    ignores = load_ignores()
    out, skipped = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__", "node_modules"))]
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(fn, p) for p in ignores):
                skipped += 1
                continue
            out.append((rel, os.path.basename(dirpath) if dirpath != root else ""))
    if skipped:
        log(f"  .memeignore: {skipped} image(s) excluded")
    out.sort()
    return out


def normalize_text(s):
    """Lowercase, strip accents, collapse separators - for keyword matching."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", s)


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

def load_clip():
    import torch
    from transformers import CLIPModel, CLIPTokenizerFast
    try:
        from transformers import CLIPImageProcessorPil as ImgProc
    except ImportError:
        from transformers import CLIPImageProcessor as ImgProc

    dev, dtype = resolve_device()
    log(f"  loading {MODEL_ID} ...")
    model = CLIPModel.from_pretrained(MODEL_ID).eval().to(device=dev, dtype=dtype)
    proc = ImgProc.from_pretrained(MODEL_ID)
    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    torch.set_grad_enabled(False)
    return model, proc, tok


def _features(out):
    """Unwrap CLIP features.

    transformers >=5 returns a BaseModelOutputWithPooling from get_*_features,
    where the projected embedding is `pooler_output`; v4 returned a bare tensor.
    """
    return out.pooler_output if hasattr(out, "pooler_output") else out


def encode_texts(model, tok, texts, batch=64):
    """Encode text -> L2-normalized embeddings."""
    import torch
    dev, _ = resolve_device()
    chunks = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=77, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}   # ids stay integral
        f = _features(model.get_text_features(**enc))
        # Normalize in fp32: the half-precision reciprocal square root is the
        # one step here where the error would be visible.
        chunks.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(chunks).numpy().astype(np.float32)


def pick_subject(model, tok, img_emb, valid, requested="auto"):
    """Choose which prompt bank to aim at this library.

    Prompts about the wrong subject still *rank* consistently - a constant
    offset shared by a category column cannot reorder it - but they spend
    their descriptive power on something absent from the picture. So ask the
    images: whichever subject noun most of them sit closer to wins.
    """
    if requested != "auto":
        return requested

    names = list(tax.SUBJECT_BANKS)
    probes = encode_texts(model, tok, [f"a photo of a {n}" for n in names])
    sims = img_emb[valid] @ probes.T
    votes = np.bincount(sims.argmax(axis=1), minlength=len(names))
    total = max(int(votes.sum()), 1)
    best = names[int(votes.argmax())]
    detail = ", ".join(f"{n} {100 * v / total:.0f}%" for n, v in zip(names, votes))
    log(f"  subject: {best} (auto-detected - {detail})")
    return best


def build_prototypes(model, tok, prompt_bank):
    """Prompt ensembling: mean of the per-prompt embeddings, renormalized.

    Averaging several phrasings per category is a well-established CLIP
    zero-shot accuracy win over any single prompt.
    """
    protos = np.zeros((len(tax.CATEGORIES), model.config.projection_dim), dtype=np.float32)
    for cat, prompts in prompt_bank.items():
        embs = encode_texts(model, tok, prompts)
        v = embs.mean(axis=0)
        protos[tax.CAT_INDEX[cat]] = v / (np.linalg.norm(v) + 1e-8)
    return protos


def _open_rgb(rel):
    """Decode one meme to RGB. Returns (image, error).

    Deliberately identical to what the single-threaded CPU path did - `draft`
    is a free DCT-domain downscale that only applies to JPEGs - so embeddings
    stay bit-comparable with the existing caches. Pre-shrinking everything
    else to 448 would be faster and cheaper, but it moves cosines by up to
    0.03 on detailed PNGs, which is enough to reorder a retrieval ranking.

    The one exception is a size guard: MAX_IMAGE_PIXELS is disabled globally,
    so without it a single pathological file could decode into gigabytes.
    """
    from PIL import Image
    try:
        im = Image.open(os.path.join(ROOT, rel))
        im.draft("RGB", (448, 448))
        rgb = im.convert("RGB")
        im.close()
        if rgb.size[0] * rgb.size[1] > 40_000_000:
            rgb.thumbnail((4096, 4096), Image.BICUBIC)
        return rgb, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _encode_pixels(model, px, dev, dtype):
    """Image tower over `px`, halving the chunk size whenever CUDA runs out."""
    import torch
    chunk = px.shape[0]
    outs, i = [], 0
    while i < px.shape[0]:
        part = px[i:i + chunk].to(device=dev, dtype=dtype, non_blocking=True)
        try:
            f = _features(model.get_image_features(pixel_values=part))
        except torch.OutOfMemoryError:
            del part
            torch.cuda.empty_cache()
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            log(f"    [warn] CUDA out of memory - retrying at chunk {chunk}")
            continue
        outs.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        i += part.shape[0]
    return np.concatenate(outs).astype(np.float32)


def encode_images(model, proc, paths, batch=None):
    """Encode images -> L2-normalized embeddings. Unreadable files get zeros."""
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    dev, dtype = resolve_device()
    if batch is None:
        batch = 48 if dev.type == "cuda" else 32

    dim = model.config.projection_dim
    out = np.zeros((len(paths), dim), dtype=np.float32)
    failed = []
    t0 = time.time()

    # With the tower on the GPU, JPEG decode is what gates throughput - so the
    # next batch is decoded on worker threads while this one encodes. At most
    # two batches are ever in flight, which is what bounds the memory.
    with ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 4)) as pool:
        def submit(start):
            group = paths[start:start + batch]
            return [(start + k, pool.submit(_open_rgb, rel)) for k, rel in enumerate(group)]

        pending = submit(0)
        for i in range(0, len(paths), batch):
            futures, pending = pending, submit(i + batch)
            pils, keep = [], []
            for idx, fut in futures:
                im, err = fut.result()
                if im is None:
                    failed.append((paths[idx], err))
                else:
                    pils.append(im)
                    keep.append(idx)
            if not pils:
                continue
            px = proc(images=pils, return_tensors="pt")["pixel_values"]
            out[keep] = _encode_pixels(model, px, dev, dtype)
            for p in pils:
                p.close()

            done = min(i + batch, len(paths))
            if done % 320 < batch or done == len(paths):
                rate = done / max(time.time() - t0, 1e-6)
                log(f"    {done}/{len(paths)} images  ({rate:.0f}/s)")

    return out, failed


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _enable_ort_cuda():
    """Make onnxruntime's CUDA provider actually attachable. True if usable.

    onnxruntime-gpu 1.29 ships CUDA as a *plugin* EP. It lists in
    get_available_providers() either way, but a session quietly falls back to
    CPU - logging only "No registered plugin EP device found" - until the
    bundled plugin is registered and the DLLs from the nvidia-* wheels are
    preloaded. Both calls are harmless where they are not needed.
    """
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    try:
        ort.preload_dlls()
    except Exception:
        pass                                    # older builds resolve DLLs themselves
    register = getattr(ort, "_register_bundled_cuda_plugin_ep", None)
    if register is None:
        return True                             # pre-plugin build: EP attaches directly
    try:
        register()
        return True
    except Exception as e:
        log(f"  [warn] CUDA plugin EP registration failed: {type(e).__name__}: {e}")
        return False


def ocr_providers(engine):
    """Providers the detector session really attached - never trust the request."""
    for member in vars(engine.text_detector).values():
        session = getattr(member, "session", None)
        if session is not None:
            return session.get_providers()
    return []


def ocr_engine():
    """RapidOCR, on the GPU when the installed onnxruntime can reach one.

    The stock `onnxruntime` wheel is CPU-only; `onnxruntime-gpu` exposes
    CUDAExecutionProvider, and RapidOCR takes it per module. Failing that we
    stay on the CPU but cap the thread pool, because ONNX Runtime otherwise
    takes every core and the machine stops responding.
    """
    import onnxruntime as ort
    import rapidocr_onnxruntime.utils as ru
    from rapidocr_onnxruntime import RapidOCR

    if _enable_ort_cuda():
        # model_path=None keeps RapidOCR's bundled weights; the kwarg has to be
        # present because its config updater reads it unconditionally.
        engine = RapidOCR(det_use_cuda=True, det_model_path=None,
                          cls_use_cuda=True, cls_model_path=None,
                          rec_use_cuda=True, rec_model_path=None)
        got = ocr_providers(engine)
        if any("CUDA" in p for p in got):
            log(f"  OCR: on GPU ({', '.join(got)})")
            return engine
        log(f"  [warn] OCR asked for CUDA but the session attached {got} - falling back")

    n = cpu_threads()
    log(f"  OCR: on CPU, {n} threads")
    base = ru.SessionOptions
    if not getattr(base, "_capped", False):
        def capped_session_options():
            so = base()
            so.intra_op_num_threads = n
            return so
        capped_session_options._capped = True
        ru.SessionOptions = capped_session_options
    return RapidOCR()


def run_ocr(paths, cache):
    """Read caption text off each meme. Resumable via `cache` (mutated in place)."""
    engine = ocr_engine()
    todo = [p for p in paths if p not in cache]
    if not todo:
        log(f"  OCR: all {len(paths)} cached")
        return
    log(f"  OCR: {len(todo)} to read ({len(paths) - len(todo)} cached)")

    t0 = time.time()
    errors = 0
    for n, rel in enumerate(todo, 1):
        try:
            res, _ = engine(os.path.join(ROOT, rel))
            parts = []
            for box in (res or []):
                text = box[1]
                # RapidOCR hands back confidence as a *string*; cast before comparing.
                try:
                    conf = float(box[2]) if len(box) > 2 and box[2] is not None else 1.0
                except (TypeError, ValueError):
                    conf = 1.0
                if conf > 0.5 and text:
                    parts.append(str(text))
            cache[rel] = " ".join(parts).strip()
        except Exception as e:
            cache[rel] = ""
            errors += 1
            if errors <= 5:
                log(f"    [warn] OCR failed on {rel}: {type(e).__name__}: {e}")
        if n % 100 == 0 or n == len(todo):
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            log(f"    {n}/{len(todo)} OCR  ({el/n:.2f}s each, ETA {eta/60:.1f} min)")
            json.dump(cache, open(OCR_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(cache, open(OCR_CACHE, "w", encoding="utf-8"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Face prominence
# ---------------------------------------------------------------------------

YUNET_MODEL = os.path.join(ROOT, "models", "face_detection_yunet_2023mar.onnx")


def face_detector():
    """(kind, detector) for the best face detector installed, or None.

    OpenCV 5 dropped `CascadeClassifier` and ships no Haar XML at all, so the
    old path is simply gone on a current install. YuNet - a 230 KB ONNX
    detector, and a far better one - is the primary route when its weights are
    sitting in models/; the cascade stays as a fallback for OpenCV 4.
    """
    import cv2
    if hasattr(cv2, "FaceDetectorYN") and os.path.exists(YUNET_MODEL):
        log("  detector: YuNet ONNX")
        return "yunet", cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), 0.6)

    if hasattr(cv2, "CascadeClassifier"):
        xml = os.path.join(getattr(cv2.data, "haarcascades", ""),
                           "haarcascade_frontalface_default.xml")
        if os.path.exists(xml):
            cascade = cv2.CascadeClassifier(xml)
            if not cascade.empty():
                log("  detector: Haar cascade")
                return "haar", cascade

    log(f"  [warn] no face detector available - OpenCV {cv2.__version__} has no usable "
        "cascade, and models/face_detection_yunet_2023mar.onnx is missing")
    log("  [warn] face prominence disabled (every meme scores 0.0)")
    return None


def face_prominence(paths):
    """Fraction of the image covered by the largest detected face.

    A close-up reaction face is a better answer to "match my expression" than a
    wide multi-panel template, so this becomes a ranking bonus at runtime.
    """
    import cv2
    cv2.setNumThreads(cpu_threads())   # OpenCV grabs every core by default

    out = np.zeros(len(paths), dtype=np.float32)
    found = face_detector()
    if found is None:
        return out
    kind, det = found

    t0 = time.time()
    for i, rel in enumerate(paths):
        try:
            data = np.fromfile(os.path.join(ROOT, rel), dtype=np.uint8)  # unicode-safe
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)   # YuNet wants 3 channels
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = 512.0 / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                h, w = img.shape[:2]
            if kind == "yunet":
                det.setInputSize((w, h))
                _, faces = det.detect(img)
                boxes = [] if faces is None else [(f[2], f[3]) for f in faces]
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                boxes = [(fw, fh) for (_, _, fw, fh) in det.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=5,
                    minSize=(max(24, w // 20), max(24, h // 20)))]
            if boxes:
                out[i] = max(fw * fh for fw, fh in boxes) / float(w * h)
        except Exception:
            pass
        if (i + 1) % 400 == 0 or i + 1 == len(paths):
            log(f"    {i+1}/{len(paths)} face scan ({time.time()-t0:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def zscore_softmax(sims, temp=SOFTMAX_TEMP, valid=None):
    """Per-category z-score across the dataset, then row-wise softmax.

    Without the z-score, categories whose prompts happen to sit closer to the
    image manifold win almost every row and the ranking is useless.
    """
    if valid is None:
        valid = np.ones(len(sims), dtype=bool)
    ref = sims[valid] if valid.any() else sims
    mu = ref.mean(axis=0, keepdims=True)
    sd = ref.std(axis=0, keepdims=True) + 1e-6
    z = (sims - mu) / sd
    z = z / temp
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def prior_from_dict(mapping):
    """Turn a {category: weight} dict into a normalized 18-vector."""
    v = np.zeros(len(tax.CATEGORIES), dtype=np.float32)
    for cat, w in mapping.items():
        v[tax.CAT_INDEX[cat]] = w
    s = v.sum()
    return v / s if s > 0 else v


def keyword_scores(text):
    """Score a normalized filename against the keyword priors."""
    v = np.zeros(len(tax.CATEGORIES), dtype=np.float32)
    padded = f" {text} "
    for cat, words in tax.KEYWORD_PRIORS.items():
        hits = sum(1 for w in words if f" {w} " in padded)
        if hits:
            v[tax.CAT_INDEX[cat]] = float(hits)
    s = v.sum()
    return v / s if s > 0 else v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze memes with a CLIP vision model.")
    ap.add_argument("--skip-ocr", action="store_true", help="skip caption OCR (faster, less accurate)")
    ap.add_argument("--skip-faces", action="store_true", help="skip face-prominence scan")
    ap.add_argument("--force", action="store_true", help="ignore caches and recompute")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="where CLIP runs (default: cuda when available)")
    ap.add_argument("--subject", default="auto",
                    choices=["auto"] + list(tax.SUBJECT_BANKS),
                    help="which prompt bank to aim at the library (default: detect it)")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="cores for the CPU-only stages (default: half of them)")
    args = ap.parse_args()

    global DEVICE_REQUEST, _CPU_THREADS
    DEVICE_REQUEST = args.device
    _CPU_THREADS = max(0, args.cpu_threads)

    log("=" * 64)
    log("  MemeMatch - vision analysis")
    log("=" * 64)
    log(f"  folder: {ROOT}")

    paths_folders = find_images(ROOT)
    if not paths_folders:
        log("  no images found - nothing to do")
        return 1
    paths = [p for p, _ in paths_folders]
    folders = {p: f for p, f in paths_folders}
    log(f"  found {len(paths)} images\n")

    # -- 1. CLIP image embeddings ------------------------------------------
    log("[1/5] CLIP image embeddings")
    model, proc, tok = load_clip()

    img_emb, failed = None, []
    if not args.force and os.path.exists(EMBED_CACHE):
        try:
            z = np.load(EMBED_CACHE, allow_pickle=True)
            if list(z["paths"]) == paths:
                img_emb = z["emb"]
                log(f"  reused cached embeddings ({img_emb.shape})")
        except Exception:
            pass
    if img_emb is None:
        img_emb, failed = encode_images(model, proc, paths)
        np.savez_compressed(EMBED_CACHE, emb=img_emb, paths=np.array(paths, dtype=object))
    for rel, err in failed:
        log(f"  [warn] unreadable: {rel} ({err})")
    readable = np.linalg.norm(img_emb, axis=1) > 0.5
    log(f"  {int(readable.sum())}/{len(paths)} images encoded\n")

    # -- 2. Category prototypes --------------------------------------------
    log("[2/5] building prompt-ensembled category prototypes")
    subject = pick_subject(model, tok, img_emb, readable, args.subject)
    expr_bank, vibe_bank = tax.SUBJECT_BANKS[subject]
    expr_protos = build_prototypes(model, tok, expr_bank)
    vibe_protos = build_prototypes(model, tok, vibe_bank)
    log(f"  {len(tax.CATEGORIES)} categories x 2 banks\n")

    # -- 3. OCR -------------------------------------------------------------
    log("[3/5] OCR captions")
    ocr_map = {}
    if not args.skip_ocr:
        if not args.force and os.path.exists(OCR_CACHE):
            try:
                ocr_map = json.load(open(OCR_CACHE, encoding="utf-8"))
            except Exception:
                ocr_map = {}
        run_ocr(paths, ocr_map)
    else:
        log("  skipped (--skip-ocr)")
    ocr_texts = [ocr_map.get(p, "") for p in paths]
    n_with_text = sum(1 for t in ocr_texts if len(t) >= 4)
    log(f"  {n_with_text}/{len(paths)} images have readable caption text\n")

    # -- 4. Face prominence -------------------------------------------------
    log("[4/5] face prominence scan")
    face_ratio = (np.zeros(len(paths), dtype=np.float32) if args.skip_faces
                  else face_prominence(paths))
    log(f"  {int((face_ratio > 0.02).sum())} images contain a detectable face\n")

    # -- 5. Fuse ------------------------------------------------------------
    log("[5/5] fusing signals")
    expr_scores = zscore_softmax(img_emb @ expr_protos.T, valid=readable)
    vibe_scores = zscore_softmax(img_emb @ vibe_protos.T, valid=readable)

    ocr_scores = np.zeros_like(expr_scores)
    has_text = np.array([len(t) >= 4 for t in ocr_texts])
    if has_text.any():
        idx = np.flatnonzero(has_text)
        txt_emb = encode_texts(model, tok, [ocr_texts[i][:300] for i in idx])
        # Compare caption text to the vibe bank: captions carry meaning, not looks.
        ocr_scores[idx] = zscore_softmax(txt_emb @ vibe_protos.T)

    folder_scores = np.zeros_like(expr_scores)
    kw_scores = np.zeros_like(expr_scores)
    for i, rel in enumerate(paths):
        fp = tax.FOLDER_PRIORS.get(folders[rel])
        if fp:
            folder_scores[i] = prior_from_dict(fp)
        kw_scores[i] = keyword_scores(normalize_text(os.path.basename(rel)))

    # Renormalize weights per row over the signals actually present, so a meme
    # with no caption/folder is not silently pushed toward a flat distribution.
    w = np.zeros((len(paths), 5), dtype=np.float32)
    w[:, 0] = W_EXPRESSION * readable
    w[:, 1] = W_VIBE * readable
    w[:, 2] = W_OCR * (ocr_scores.sum(axis=1) > 0)
    w[:, 3] = W_FOLDER * (folder_scores.sum(axis=1) > 0)
    w[:, 4] = W_KEYWORD * (kw_scores.sum(axis=1) > 0)
    w /= (w.sum(axis=1, keepdims=True) + 1e-8)

    final = (w[:, 0:1] * expr_scores + w[:, 1:2] * vibe_scores + w[:, 2:3] * ocr_scores
             + w[:, 3:4] * folder_scores + w[:, 4:5] * kw_scores)
    final /= (final.sum(axis=1, keepdims=True) + 1e-12)

    # -- write --------------------------------------------------------------
    memes = []
    for i, rel in enumerate(paths):
        order = np.argsort(-final[i])[:3]
        memes.append({
            "path": rel,
            "name": os.path.splitext(os.path.basename(rel))[0],
            "folder": folders[rel],
            "text": ocr_texts[i][:220],
            "face": round(float(face_ratio[i]), 4),
            "top": [tax.CATEGORIES[k] for k in order],
            "scores": [round(float(x), 5) for x in final[i]],
        })

    catalog = {
        "version": 3,
        "model": MODEL_ID,
        "subject": subject,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "categories": tax.CATEGORIES,
        "embedDim": int(img_emb.shape[1]),
        "embedFile": os.path.basename(EMBED_OUT),
        "count": len(memes),
        "memes": memes,
    }
    with open(CATALOG_OUT, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, separators=(",", ":"))

    # float16 embeddings, row-aligned with catalog["memes"], for semantic search
    img_emb.astype(np.float16).tofile(EMBED_OUT)

    # -- report -------------------------------------------------------------
    log("\n  dominant category distribution:")
    tops = [m["top"][0] for m in memes]
    for cat in tax.CATEGORIES:
        n = tops.count(cat)
        log(f"    {cat:11s} {n:4d}  {'#' * min(n // 4, 44)}")

    log(f"\n  wrote {CATALOG_OUT} ({os.path.getsize(CATALOG_OUT)/1024:.0f} KB)")
    log(f"  wrote {EMBED_OUT} ({os.path.getsize(EMBED_OUT)/1024:.0f} KB)")
    log(f"  {len(memes)} memes analyzed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
