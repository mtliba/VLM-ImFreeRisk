"""Signal-level image degradation: answer the KEEP prompt with progressively degraded versions of the real image.
Conditions: blur{2,4,8,16} (Gaussian sigma in px on the 512-px-resized image), down{8,16,32} (downsample to k x k then upsample back),
noise{0.25,0.5} (additive Gaussian noise, std as fraction of the intensity range), grey (uniform grey = blank, for reference).
Output: results/degradation/{dataset}_{model}_{split}_{cond}.jsonl (sample_id, answer_raw, score, unchanged flag is computed offline)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageFilter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, load_prompt, read_jsonl, score
from run_vlm import generate_batch, load_model, make_messages
CONDS = ["blur2", "blur4", "blur8", "blur16", "down32", "down16", "down8", "noise0.25", "noise0.5"]

def degrade(img, cond, seed=0):
    img = img.convert("RGB"); w, h = img.size; s = 512 / max(w, h); img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    if cond.startswith("blur"): return img.filter(ImageFilter.GaussianBlur(float(cond[4:])))
    if cond.startswith("down"):
        k = int(cond[4:]); return img.resize((k, k), Image.BOX).resize(img.size, Image.NEAREST)
    if cond.startswith("noise"):
        a = np.asarray(img).astype(np.float32); rng = np.random.RandomState(seed); a = a + rng.normal(0, float(cond[5:]) * 255, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    raise ValueError(cond)

def run(model_name, dataset, splits, conds, bs):
    model, processor = load_model(model_name); out_dir = RESULTS / "degradation"; out_dir.mkdir(exist_ok=True)
    for split in splits:
        samples = read_jsonl(DATA / dataset / f"{split}.jsonl")
        for cond in conds:
            op = out_dir / f"{dataset}_{model_name}_{split}_{cond}.jsonl"
            done = {r["sample_id"] for r in read_jsonl(op)} if op.exists() else set(); todo = [s for s in samples if s["sample_id"] not in done]
            print(f"[{dataset}/{model_name}/{split}/{cond}] {len(done)} done, {len(todo)} to do", flush=True); t0 = time.time()
            with open(op, "a") as f:
                for i in range(0, len(todo), bs):
                    chunk = todo[i:i + bs]
                    msgs = [make_messages([degrade(Image.open(s["image_path"]), cond, seed=j)], load_prompt("baseline").format(question=s["question"].strip())) for j, s in enumerate(chunk, i)]
                    try: texts, _ = generate_batch(model, processor, msgs)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache(); texts = sum([generate_batch(model, processor, [m])[0] for m in msgs], [])
                    for s, t in zip(chunk, texts):
                        f.write(json.dumps(dict(sample_id=s["sample_id"], cond=cond, answer_raw=t, score=score(t, s["answer"])), ensure_ascii=False) + "\n")
                    f.flush()
                    if (i // bs) % 20 == 0: print(f"  {i+len(chunk)}/{len(todo)} {(time.time()-t0)/60:.1f} min", flush=True)
            print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"]); ap.add_argument("--conds", nargs="+", default=CONDS); ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args(); run(a.model, a.dataset, a.splits, a.conds, a.batch_size)
