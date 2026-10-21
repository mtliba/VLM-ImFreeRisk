"""A1 image-counterfactual: answer the KEEP prompt with (noimage | blank | swap) instead of the real image.
Output: results/counterfactual/{dataset}_{model}_{split}_{cond}.jsonl (sample_id, answer_raw, score, score_b, swap_image)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import score_canonical
from common import DATA, MODEL_IDS, RESULTS, load_prompt, read_jsonl, score
from run_vlm import generate_batch, load_model, make_messages

def swap_map(samples, seed=0):
    rng = np.random.RandomState(seed); imgs = sorted({s["image_path"] for s in samples}); mod = {s["image_path"]: s.get("modality") for s in samples}
    out = {}
    for s in samples:
        cands = [p for p in imgs if p != s["image_path"] and (mod.get(p) == s.get("modality") if s.get("modality") else True)] or [p for p in imgs if p != s["image_path"]]
        out[s["sample_id"]] = cands[rng.randint(len(cands))]
    return out

def run(model_name, dataset, splits, conds, bs):
    model, processor = load_model(model_name)
    out_dir = RESULTS / "counterfactual"; out_dir.mkdir(exist_ok=True)
    blank = Image.new("RGB", (512, 512), (128, 128, 128))
    for split in splits:
        samples = read_jsonl(DATA / dataset / f"{split}.jsonl"); sw = swap_map(samples)
        for cond in conds:
            op = out_dir / f"{dataset}_{model_name}_{split}_{cond}.jsonl"
            done = {r["sample_id"] for r in read_jsonl(op)} if op.exists() else set()
            todo = [s for s in samples if s["sample_id"] not in done]
            print(f"[{dataset}/{model_name}/{split}/{cond}] {len(done)} done, {len(todo)} to do", flush=True); t0 = time.time()
            with open(op, "a") as f:
                for i in range(0, len(todo), bs):
                    chunk = todo[i:i + bs]; msgs = []
                    for s in chunk:
                        prompt = load_prompt("baseline").format(question=s["question"].strip())
                        if cond == "noimage": msgs.append([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
                        elif cond == "blank": msgs.append(make_messages([blank], prompt))
                        else: msgs.append(make_messages([Image.open(sw[s["sample_id"]]).convert("RGB")], prompt))
                    try: texts, _ = generate_batch(model, processor, msgs)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache(); texts = [generate_batch(model, processor, [m])[0][0] for m in msgs]
                    for s, t in zip(chunk, texts):
                        f.write(json.dumps(dict(sample_id=s["sample_id"], cond=cond, answer_raw=t, score=score(t, s["answer"]), score_b=score_canonical(t, s["answer"], s["question"]),
                                                swap_image=sw[s["sample_id"]] if cond == "swap" else None), ensure_ascii=False) + "\n")
                    f.flush()
                    if (i // bs) % 10 == 0: print(f"  {i+len(chunk)}/{len(todo)} {(time.time()-t0)/60:.1f} min ex: {texts[0][:40]!r}", flush=True)
            print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"]); ap.add_argument("--conds", nargs="+", default=["noimage", "blank", "swap"]); ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args(); torch.manual_seed(0); run(a.model, a.dataset, a.splits, a.conds, a.batch_size)
