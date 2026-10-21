"""Sampling-based self-consistency baseline: K sampled answers to the KEEP prompt (T=0.7, top-p 0.95).
Output: results/sampling/{dataset}_{model}_{split}.jsonl with sample_id, samples (list), scores (list), agree_greedy (fraction equal
to the greedy KEEP answer after canonicalisation), n_distinct."""
import argparse, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonicalize, closed_pred_canonical
from common import DATA, MODEL_IDS, RESULTS, build_input, read_jsonl, score
from run_vlm import load_model, make_messages

def key(a, q):
    yn = closed_pred_canonical(a, q); return yn if yn in ("yes", "no") else canonicalize(a)

@torch.no_grad()
def sample_batch(model, processor, msgs, k, seed):
    inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    out = model.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.7, top_p=0.95, top_k=None, num_return_sequences=k)
    gen = out[:, inputs["input_ids"].shape[1]:]
    texts = [t.strip() for t in processor.batch_decode(gen, skip_special_tokens=True)]
    return [texts[i * k:(i + 1) * k] for i in range(len(msgs))]

def run(model_name, dataset, splits, k, bs):
    model, processor = load_model(model_name)
    out_dir = RESULTS / "sampling"; out_dir.mkdir(exist_ok=True)
    for split in splits:
        samples = read_jsonl(DATA / dataset / f"{split}.jsonl")
        keep = {r["sample_id"]: r["answer_raw"] for r in read_jsonl(RESULTS / "raw" / f"{dataset}_{model_name}_{split}_keep.jsonl")}
        op = out_dir / f"{dataset}_{model_name}_{split}.jsonl"
        done = {r["sample_id"] for r in read_jsonl(op)} if op.exists() else set()
        todo = [s for s in samples if s["sample_id"] not in done]
        print(f"[{dataset}/{model_name}/{split}/sampling] {len(done)} done, {len(todo)} to do", flush=True); t0 = time.time()
        with open(op, "a") as f:
            for i in range(0, len(todo), bs):
                chunk = todo[i:i + bs]
                msgs = [make_messages(*build_input(s, "keep")) for s in chunk]
                try:
                    outs = sample_batch(model, processor, msgs, k, seed=i)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); outs = [sample_batch(model, processor, [m], k, seed=i + j)[0] for j, m in enumerate(msgs)]
                for s, ss in zip(chunk, outs):
                    g = key(keep[s["sample_id"]], s["question"]); ks = [key(x, s["question"]) for x in ss]
                    f.write(json.dumps(dict(sample_id=s["sample_id"], samples=ss, scores=[score(x, s["answer"]) for x in ss],
                                            agree_greedy=sum(x == g for x in ks) / k, n_distinct=len(set(ks + [g]))), ensure_ascii=False) + "\n")
                f.flush()
                if (i // bs) % 10 == 0: print(f"  {i+len(chunk)}/{len(todo)} {(time.time()-t0)/60:.1f} min", flush=True)
        print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "test"]); ap.add_argument("--k", type=int, default=5); ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args(); run(a.model, a.dataset, a.splits, a.k, a.batch_size)
