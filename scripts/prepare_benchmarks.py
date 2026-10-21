"""Prepare the generalisation benchmarks with the same jsonl schema as data/{vqa_rad,slake}:
  pathvqa   : flaviagiammarino/path-vqa (official splits; subsample train 5000 / val 1000 / test 2000, seed 0, stratified by closed/open)
  pmcvqa    : RadGenome/PMC-VQA test_clean.csv (2000 MCQ) + train.csv subsample 5000 train / 1000 val (image-disjoint from test), MCQ prompt with options
  omnimedvqa: foreverbeliever/OmniMedVQA, MCQ; stratified by modality: 5000 train / 1000 val / 2000 test, image-disjoint, seed 0
Question text for MCQ = "<question>\nOptions: A. ...  B. ...  C. ...  D. ...\nAnswer with the option letter only."  answer = letter."""
import argparse, hashlib, io, json, glob, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, is_closed, write_jsonl
HOME = Path.home()
LET = "ABCD"

def subsample(rows, n, rng, strat_key=None, group_key="image_id"):
    """image-disjoint stratified subsample of about n rows."""
    if len(rows) <= n: return rows
    groups = {}
    for r in rows: groups.setdefault(r[group_key], []).append(r)
    keys = list(groups); rng.shuffle(keys); out = []
    for k in keys:
        out += groups[k]
        if len(out) >= n: break
    return out

def pathvqa(rng):
    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/path-vqa"); out_dir = DATA / "pathvqa"; (out_dir / "images").mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, n in [("train", 5000), ("validation", 1000), ("test", 2000)]:
        rows = []; cache = {}
        idx = np.arange(len(ds[split])); rng.shuffle(idx)
        for i in idx:
            ex = ds[split][int(i)]; img = ex["image"].convert("RGB"); b = io.BytesIO(); img.save(b, format="PNG"); h = hashlib.md5(b.getvalue()).hexdigest()
            if h not in cache:
                p = out_dir / "images" / f"{h}.png"
                if not p.exists(): p.write_bytes(b.getvalue())
                cache[h] = str(p)
            rows.append(dict(sample_id=f"pathvqa_{split}_{int(i)}", dataset="pathvqa", image_id=h, image_path=cache[h], question=ex["question"].strip(), answer=str(ex["answer"]).strip(),
                             answer_type="CLOSED" if is_closed(ex["answer"]) else "OPEN", question_type="CLOSED" if is_closed(ex["answer"]) else "OPEN", modality="pathology", location=None))
            if len(rows) >= n * 1.3: break
        rows = subsample(rows, n, rng)
        write_jsonl(rows, out_dir / f"{split}.jsonl"); stats[split] = dict(n_qa=len(rows), n_images=len({r["image_id"] for r in rows}), closed=sum(r["answer_type"] == "CLOSED" for r in rows))
    print("pathvqa", stats); return stats

def mcq_question(q, opts):
    return f"{q.strip()}\nOptions: " + "  ".join(f"{LET[i]}. {o}" for i, o in enumerate(opts)) + "\nAnswer with the option letter only."

def pmcvqa(rng):
    root = HOME / "data" / "pmcvqa"; imgdir = root / "images"; out_dir = DATA / "pmcvqa"; out_dir.mkdir(parents=True, exist_ok=True)
    def rows_from(csv, split, n):
        df = pd.read_csv(csv); rows = []
        for i, r in df.iterrows():
            p = imgdir / str(r["Figure_path"]).strip()
            if not p.exists(): continue
            opts = [str(r[f"Choice {c}"]).strip() for c in LET]; opts = [o.split(":", 1)[1].strip() if o[:2] in [f"{c}:" for c in LET] else o for o in opts]
            ans = str(r["Answer_label"]).strip().upper()[:1]
            if ans not in LET: continue
            rows.append(dict(sample_id=f"pmcvqa_{split}_{i}", dataset="pmcvqa", image_id=str(r["Figure_path"]).strip(), image_path=str(p), question=mcq_question(str(r["Question"]), opts), answer=ans,
                             answer_type="MCQ", question_type="MCQ", modality=None, location=None, raw_question=str(r["Question"]).strip(), answer_text=str(r["Answer"]).strip()))
        return rows
    test = rows_from(root / "test_clean.csv", "test", 2000)
    test_imgs = {r["image_id"] for r in test}
    tr_all = [r for r in rows_from(root / "train.csv", "train", None) if r["image_id"] not in test_imgs]
    rng.shuffle(tr_all); train = subsample(tr_all, 5000, rng); rest = [r for r in tr_all if r["image_id"] not in {x["image_id"] for x in train}]; val = subsample(rest, 1000, rng)
    for s in val: s["sample_id"] = s["sample_id"].replace("_train_", "_validation_")
    stats = {}
    for split, rows in [("train", train), ("validation", val), ("test", test)]:
        write_jsonl(rows, out_dir / f"{split}.jsonl"); stats[split] = dict(n_qa=len(rows), n_images=len({r["image_id"] for r in rows}))
    print("pmcvqa", stats, "missing-image test rows:", 2000 - len(test)); return stats

def omnimedvqa(rng):
    import os
    root = HOME / "data" / "omnimedvqa"; js = sorted(glob.glob(str(root / "**" / "*.json"), recursive=True)); out_dir = DATA / "omnimedvqa"; out_dir.mkdir(parents=True, exist_ok=True)
    # index the image tree once (one os.walk) instead of 128k NFS stat calls
    img_root = root / "OmniMedVQA"; have = set()
    for dp, _, fns in os.walk(img_root / "Images"):
        for f in fns: have.add(os.path.relpath(os.path.join(dp, f), img_root))
    print("indexed", len(have), "image files", flush=True)
    items = []; missing = 0
    for j in js:
        for r in json.load(open(j)):
            opts = [r.get(f"option_{c}") for c in LET]
            if any(o is None for o in opts[:2]): continue
            opts = [o for o in opts if o is not None]
            gt = str(r.get("gt_answer", "")).strip()
            if gt not in opts: continue
            rel = str(r["image_path"])
            if rel not in have: missing += 1; continue
            img = img_root / rel
            items.append(dict(sample_id=f"omni_{r.get('question_id', len(items))}", dataset="omnimedvqa", image_id=str(img), image_path=str(img), question=mcq_question(r["question"], opts), answer=LET[opts.index(gt)],
                              answer_type="MCQ", question_type=str(r.get("question_type", "unknown")), modality=str(r.get("modality_type", "unknown")), location=str(r.get("dataset", "")), raw_question=r["question"], answer_text=gt))
    print("items", len(items), "missing images", missing, flush=True)
    rng.shuffle(items)
    # image-disjoint stratified-by-modality split: test 2000, val 1000, train 5000
    by_img = {}
    for it in items: by_img.setdefault(it["image_id"], []).append(it)
    imgs = list(by_img); rng.shuffle(imgs)
    splits = {"test": [], "validation": [], "train": []}; quota = {"test": 2000, "validation": 1000, "train": 5000}
    mod_count = {s: Counter() for s in splits}; mods = Counter(it["modality"] for it in items); n_mod = len(mods)
    for im in imgs:
        its = by_img[im]; mod = its[0]["modality"]
        for s in ["test", "validation", "train"]:
            if len(splits[s]) < quota[s] and mod_count[s][mod] < quota[s] / n_mod * 2.5:
                splits[s] += its; mod_count[s][mod] += len(its); break
    stats = {}
    for s, rows in splits.items():
        for r in rows: r["sample_id"] = f"omni_{s}_{r['sample_id'].split('_',1)[1]}"
        write_jsonl(rows, out_dir / f"{s}.jsonl"); stats[s] = dict(n_qa=len(rows), n_images=len({r["image_id"] for r in rows}), modalities=dict(Counter(r["modality"] for r in rows)))
    print("omnimedvqa", {s: (v["n_qa"], v["n_images"]) for s, v in stats.items()}, "total available", len(items)); return stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--which", nargs="+", default=["pathvqa", "pmcvqa", "omnimedvqa"]); a = ap.parse_args()
    rng = np.random.RandomState(0); stats = {}
    for w in a.which: stats[w] = {"pathvqa": pathvqa, "pmcvqa": pmcvqa, "omnimedvqa": omnimedvqa}[w](rng)
    p = RESULTS / "dataset_stats_extension.json"; old = json.load(open(p)) if p.exists() else {}; old.update(stats); json.dump(old, open(p, "w"), indent=1, default=str)
