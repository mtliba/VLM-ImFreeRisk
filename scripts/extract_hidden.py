"""A3: last-layer hidden state at the last prompt token of the KEEP prompt (with the real image), per split.
Output: results/hidden/{dataset}_{model}_{split}.npz (sample_id, h [n, d] float16)."""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, build_input, read_jsonl
from run_vlm import load_model, make_messages

@torch.no_grad()
def run(model_name, dataset, splits, bs):
    model, processor = load_model(model_name); out_dir = RESULTS / "hidden"; out_dir.mkdir(exist_ok=True)
    for split in splits:
        samples = read_jsonl(DATA / dataset / f"{split}.jsonl"); op = out_dir / f"{dataset}_{model_name}_{split}.npz"
        if op.exists(): print("exists", op); continue
        H = []; t0 = time.time()
        for i in range(0, len(samples), bs):
            chunk = samples[i:i + bs]; msgs = [make_messages(*build_input(s, "keep")) for s in chunk]
            inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True).to(model.device)
            def fwd(inp):
                try:
                    o = model(**inp, output_hidden_states=True, logits_to_keep=1)
                except TypeError:
                    o = model(**inp, output_hidden_states=True)
                return o.hidden_states[-1][:, -1, :]  # left padding -> last position is the last prompt token
            try:
                hs = fwd(inputs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); hs = []
                for m in msgs:
                    inp = processor.apply_chat_template([m], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True).to(model.device)
                    hs.append(fwd(inp))
                hs = torch.cat(hs)
            H.append(hs.float().cpu().numpy().astype(np.float16)); del hs
            if (i // bs) % 20 == 0: print(f"  {i+len(chunk)}/{len(samples)} {(time.time()-t0)/60:.1f} min", flush=True)
        np.savez(op, sample_id=np.array([s["sample_id"] for s in samples]), h=np.concatenate(H)); print("saved", op, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"]); ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args(); run(a.model, a.dataset, a.splits, a.batch_size)
