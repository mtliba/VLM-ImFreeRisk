"""Download VQA-RAD and SLAKE (English), write images to disk and one jsonl per split.

VQA-RAD : flaviagiammarino/vqa-rad (HF)  official train/test. There is no official
          validation split, so 15% of *training images* (grouped by image, seed 0) are
          held out as validation. Question types are recovered from the original OSF
          release (question_type / answer_type) by matching (question, answer).
SLAKE   : BoKelvin/SLAKE (HF, original json + imgs.zip). q_lang == 'en' only,
          official train/validation/test.
"""
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from common import DATA, RESULTS, normalize_answer, is_closed, write_jsonl

stats = {}


def img_hash(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return hashlib.md5(b.getvalue()).hexdigest()


# ------------------------------------------------------------------ VQA-RAD
def load_osf_vqarad():
    """Original VQA-RAD json (for question_type). Returns dict (q_norm, a_norm) -> meta."""
    cands = list(DATA.glob("vqa_rad_osf/**/*.json")) + list(Path.home().glob("data/vqa_rad_osf/**/*.json"))
    zips = list(Path.home().glob("data/vqa_rad_osf/*.zip"))
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(".json") and "public" in n.lower():
                        with zf.open(n) as f:
                            rows = json.load(f)
                        return rows
        except zipfile.BadZipFile:
            pass
    for c in cands:
        if "public" in c.name.lower():
            return json.load(open(c))
    return None


def build_vqarad():
    out_dir = DATA / "vqa_rad"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    ds = load_dataset("flaviagiammarino/vqa-rad")
    osf = load_osf_vqarad()
    meta = {}
    if osf:
        for r in osf:
            key = (normalize_answer(r["question"]), normalize_answer(r["answer"]))
            meta.setdefault(key, r)
        print(f"OSF VQA-RAD metadata loaded: {len(osf)} rows")
    else:
        print("WARNING: OSF VQA-RAD metadata not found; question_type will be 'unknown'")

    rows_by_split = {}
    img_paths = {}
    for split in ["train", "test"]:
        rows = []
        for i, ex in enumerate(ds[split]):
            img = ex["image"].convert("RGB")
            h = img_hash(img)
            if h not in img_paths:
                p = out_dir / "images" / f"{h}.png"
                if not p.exists():
                    img.save(p)
                img_paths[h] = str(p)
            key = (normalize_answer(ex["question"]), normalize_answer(ex["answer"]))
            m = meta.get(key, {})
            rows.append(dict(
                sample_id=f"vqarad_{split}_{i}",
                dataset="vqa_rad",
                image_id=h,
                image_path=img_paths[h],
                question=ex["question"].strip(),
                answer=str(ex["answer"]).strip(),
                answer_type="CLOSED" if is_closed(ex["answer"]) else "OPEN",
                question_type=str(m.get("question_type", "unknown")).upper(),
                osf_answer_type=str(m.get("answer_type", "unknown")).upper(),
                modality=None, location=None,
            ))
        rows_by_split[split] = rows

    # held-out validation from train, grouped by image
    rng = np.random.RandomState(0)
    train_imgs = sorted({r["image_id"] for r in rows_by_split["train"]})
    rng.shuffle(train_imgs)
    n_val = int(round(0.15 * len(train_imgs)))
    val_imgs = set(train_imgs[:n_val])
    tr = [r for r in rows_by_split["train"] if r["image_id"] not in val_imgs]
    va = [r for r in rows_by_split["train"] if r["image_id"] in val_imgs]
    test_imgs = {r["image_id"] for r in rows_by_split["test"]}
    splits = {"train": tr, "validation": va, "test": rows_by_split["test"]}
    for s, rows in splits.items():
        write_jsonl(rows, out_dir / f"{s}.jsonl")
    stats["vqa_rad"] = {
        s: dict(n_qa=len(rows), n_images=len({r["image_id"] for r in rows}),
                closed=sum(r["answer_type"] == "CLOSED" for r in rows),
                open=sum(r["answer_type"] == "OPEN" for r in rows),
                question_types=dict(Counter(r["question_type"] for r in rows)))
        for s, rows in splits.items()
    }
    stats["vqa_rad"]["note"] = ("validation = 15% of official-train images held out (seed 0, grouped by image). "
                                f"Official test images overlapping with train images: {len(test_imgs & set(train_imgs))} "
                                "(a known property of VQA-RAD; the official split is kept as is). "
                                f"question_type matched from OSF for {sum(r['question_type']!='UNKNOWN' for s in splits.values() for r in s)} / "
                                f"{sum(len(s) for s in splits.values())} QA pairs.")
    print(json.dumps(stats["vqa_rad"], indent=1))


# ------------------------------------------------------------------ SLAKE
def build_slake():
    out_dir = DATA / "slake"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    zpath = hf_hub_download("BoKelvin/SLAKE", "imgs.zip", repo_type="dataset")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(out_dir / "images")
    excluded = Counter()
    for split, fname in [("train", "train.json"), ("validation", "validation.json"), ("test", "test.json")]:
        jp = hf_hub_download("BoKelvin/SLAKE", fname, repo_type="dataset")
        rows = []
        for r in json.load(open(jp)):
            if r.get("q_lang") != "en":
                excluded[f"{split}:non-english"] += 1
                continue
            p = out_dir / "images" / r["img_name"]
            if not p.exists():
                # imgs.zip may nest under imgs/
                alt = list((out_dir / "images").glob(f"**/{r['img_name']}"))
                if not alt:
                    excluded[f"{split}:missing-image"] += 1
                    continue
                p = alt[0]
            rows.append(dict(
                sample_id=f"slake_{split}_{r['qid']}",
                dataset="slake",
                image_id=r["img_name"],
                image_path=str(p),
                question=r["question"].strip(),
                answer=str(r["answer"]).strip(),
                answer_type="CLOSED" if is_closed(r["answer"]) else "OPEN",
                question_type=str(r.get("content_type", "unknown")).upper(),
                osf_answer_type=str(r.get("answer_type", "unknown")).upper(),
                modality=r.get("modality"), location=r.get("location"),
            ))
        write_jsonl(rows, out_dir / f"{split}.jsonl")
        stats.setdefault("slake", {})[split] = dict(
            n_qa=len(rows), n_images=len({r["image_id"] for r in rows}),
            closed=sum(r["answer_type"] == "CLOSED" for r in rows),
            open=sum(r["answer_type"] == "OPEN" for r in rows),
            question_types=dict(Counter(r["question_type"] for r in rows)))
    stats["slake"]["excluded"] = dict(excluded)
    stats["slake"]["note"] = "official splits; English questions only; answer_type CLOSED := reference answer is yes/no"
    print(json.dumps(stats["slake"], indent=1))


if __name__ == "__main__":
    build_vqarad()
    build_slake()
    RESULTS.mkdir(exist_ok=True)
    json.dump(stats, open(RESULTS / "dataset_stats.json", "w"), indent=1)
