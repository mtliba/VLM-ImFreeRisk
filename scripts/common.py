"""Shared utilities: paths, prompts, answer normalisation, scoring, image interventions."""
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"

ACTIONS = ["keep", "reprompt", "enhance", "zoom"]
ACTION_NAMES = {"keep": "KEEP", "reprompt": "REPROMPT", "enhance": "ENHANCE", "zoom": "ZOOM"}

MODEL_IDS = {
    "medgemma": "google/medgemma-4b-it",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    # additional VLMs for the error-predictability analysis (KEEP prompt + interventions on test only)
    "qwen3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2vl2b": "Qwen/Qwen2-VL-2B-Instruct",
    "gemma3": "google/gemma-3-4b-it",
    # generalisation benchmarks / additional VLMs
    "lingshu7b": "lingshu-medical-mllm/Lingshu-7B",
    "internvl3_8b": "OpenGVLab/InternVL3-8B-hf",
    "qwen3vl8b": "Qwen/Qwen3-VL-8B-Instruct",
    "huatuo7b": "FreedomIntelligence/HuatuoGPT-Vision-7B",
    "medgemma27b": "google/medgemma-27b-it",
    # scale control: same family as qwen / qwen3b, 32B
    "qwen25vl32b": "Qwen/Qwen2.5-VL-32B-Instruct",
}
# LoRA-fine-tuned variants (fine-tuning control): "<base>_ft_<dataset>" -> base weights + adapter in results/lora/<name>
for _b in ["medgemma", "qwen"]:
    for _d in ["vqa_rad", "slake", "pathvqa"]:
        MODEL_IDS[f"{_b}_ft_{_d}"] = MODEL_IDS[_b]

# ----------------------------------------------------------------------------- I/O
def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_prompt(name):
    return (PROMPTS / f"{name}.txt").read_text().strip()


# ----------------------------------------------------------------------------- answer normalisation / scoring
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s):
    """lowercase, strip punctuation, drop articles, collapse whitespace (SQuAD-style)."""
    s = str(s).lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


_QT_FIX = {"ATRIB": "ATTRIB", "PRSE": "PRES"}


def canonical_qtype(qt):
    """VQA-RAD labels contain compound ('POS, PRES') and misspelt types; keep the first, fix typos."""
    qt = str(qt).upper().split(",")[0].strip()
    return _QT_FIX.get(qt, qt)


def is_closed(ref):
    return normalize_answer(ref) in {"yes", "no"}


def token_f1(pred, ref):
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


def closed_pred(pred):
    """Map a free-form prediction to yes/no using its first token; None if neither."""
    toks = normalize_answer(pred).split()
    if not toks:
        return None
    if toks[0] in {"yes", "no"}:
        return toks[0]
    return None


_MCQ = re.compile(r"^[a-d]$")


def is_mcq(ref):
    return bool(_MCQ.match(normalize_answer(ref)))


def mcq_pred(pred):
    """first letter-like token of a prediction ('B', 'B:', '(b)', 'B) Magnetic ...') -> 'b'; None if absent."""
    m = re.match(r"^\W*([a-dA-D])(?:\b|[\W_])", str(pred).strip())
    return m.group(1).lower() if m else None


def score(pred, ref):
    """Q(y, Y) in [0,1]: exact match for yes/no questions and MCQ letters, token-F1 otherwise."""
    if is_mcq(ref):
        return float(mcq_pred(pred) == normalize_answer(ref))
    if is_closed(ref):
        return float(closed_pred(pred) == normalize_answer(ref))
    return token_f1(pred, ref)


# ----------------------------------------------------------------------------- image interventions
def enhance_image(img: Image.Image) -> Image.Image:
    """A2 ENHANCE: percentile (1-99) intensity stretch + CLAHE (clip 2.0, 8x8 tiles) on the
    luminance channel (LAB L). Chroma is untouched; geometry is untouched."""
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    lo, hi = np.percentile(L, 1), np.percentile(L, 99)
    if hi > lo:
        L = np.clip((L - lo) / (hi - lo) * 255.0, 0, 255)
    L = L.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(L)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def zoom_views(img: Image.Image, frac=0.6):
    """A3 ZOOM: full image + four overlapping corner crops, each covering `frac` of the
    width/height (60% -> 20% overlap between neighbours). Crops are upsampled to the
    full image's size so every view carries the same pixel budget."""
    img = img.convert("RGB")
    W, H = img.size
    cw, ch = int(round(W * frac)), int(round(H * frac))
    boxes = [
        (0, 0, cw, ch),
        (W - cw, 0, W, ch),
        (0, H - ch, cw, H),
        (W - cw, H - ch, W, H),
    ]
    crops = [img.crop(b).resize((W, H), Image.BICUBIC) for b in boxes]
    return [img] + crops


def zoom_montage(img: Image.Image, tile=448, frac=0.6):
    """Square montage fallback (not used by default): full image on top-left half etc."""
    views = zoom_views(img, frac)
    views = [v.resize((tile, tile), Image.BICUBIC) for v in views]
    canvas = Image.new("RGB", (3 * tile, 2 * tile), "white")
    canvas.paste(views[0], (0, 0))
    canvas.paste(views[1], (tile, 0))
    canvas.paste(views[2], (2 * tile, 0))
    canvas.paste(views[3], (tile, tile))
    canvas.paste(views[4], (2 * tile, tile))
    return canvas


def build_input(sample, action):
    """Return (list_of_PIL_images, prompt_text) for an action."""
    img = Image.open(sample["image_path"]).convert("RGB")
    q = sample["question"].strip()
    if action == "keep":
        return [img], load_prompt("baseline").format(question=q)
    if action.startswith("keep_v"):  # prompt-template robustness: prompts/baseline_v{k}.txt
        return [img], load_prompt("baseline_" + action[5:]).format(question=q)
    if action == "reprompt":
        return [img], load_prompt("reprompt").format(question=q)
    if action == "enhance":
        return [enhance_image(img)], load_prompt("baseline").format(question=q)
    if action == "zoom":
        return zoom_views(img), load_prompt("zoom").format(question=q)
    raise ValueError(action)


def load_matrix():
    """intervention_matrix.csv with answer strings kept verbatim ('None' is a real answer) and numeric columns coerced."""
    import pandas as pd

    df = pd.read_csv(RESULTS / "intervention_matrix.csv", keep_default_na=False)
    for c in df.columns:
        if c.endswith(("_score", "_conf", "_improvement")) or c.startswith("dq_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
