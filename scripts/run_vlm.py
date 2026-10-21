"""Run a frozen VLM on every (split, action) of a dataset. Greedy decoding, batched.

Output: results/raw/{dataset}_{model}_{split}_{action}.jsonl with
    sample_id, action, answer_raw, score, confidence (mean generated-token log-prob)
Resumable: already-written sample_ids are skipped.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTIONS, DATA, MODEL_IDS, RESULTS, build_input, read_jsonl, score  # noqa: E402

GEN_KW = dict(max_new_tokens=32, do_sample=False, temperature=None, top_p=None, top_k=None)


def load_model(name):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    mid = MODEL_IDS[name]
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if name.startswith("qwen"):
        processor = AutoProcessor.from_pretrained(mid)
        ip = processor.image_processor
        # fixed pixel budget per image: 256*28*28 .. 1024*28*28 (transformers>=5 SizeDict / dict, older min/max_pixels)
        sz = getattr(ip, "size", None)
        if sz is not None and (isinstance(sz, dict) or hasattr(sz, "shortest_edge")):
            ip.size = {"shortest_edge": 256 * 28 * 28, "longest_edge": 1024 * 28 * 28}
        for k, v in [("min_pixels", 256 * 28 * 28), ("max_pixels", 1024 * 28 * 28)]:
            if hasattr(ip, k):
                setattr(ip, k, v)
        print("qwen image size cfg:", getattr(ip, "size", None), getattr(ip, "min_pixels", None), getattr(ip, "max_pixels", None))
    else:
        processor = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(mid, **kw)
    if "_ft_" in name:  # LoRA adapter trained by finetune_lora.py, merged into the base weights
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(RESULTS / "lora" / name)).merge_and_unload()
        print("merged adapter", RESULTS / "lora" / name)
    model = model.eval()
    processor.tokenizer.padding_side = "left"
    return model, processor


def make_messages(images, prompt):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
    return [{"role": "user", "content": content}]


@torch.no_grad()
def generate_batch(model, processor, batch_msgs):
    inputs = processor.apply_chat_template(
        batch_msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", padding=True,
    ).to(model.device)
    out = model.generate(**inputs, **GEN_KW, output_scores=True, return_dict_in_generate=True)
    in_len = inputs["input_ids"].shape[1]
    gen = out.sequences[:, in_len:]
    trans = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    pad = processor.tokenizer.pad_token_id
    eos = processor.tokenizer.eos_token_id
    texts = processor.batch_decode(gen, skip_special_tokens=True)
    confs = []
    for i in range(gen.shape[0]):
        mask = (gen[i] != pad) & (gen[i] != eos)
        lp = trans[i][: gen.shape[1]][mask]
        confs.append(float(lp.mean()) if lp.numel() else float("nan"))
    return [t.strip() for t in texts], confs


def run(model_name, dataset, splits, actions, batch_size, limit=None):
    model, processor = load_model(model_name)
    out_dir = RESULTS / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        samples = read_jsonl(DATA / dataset / f"{split}.jsonl")
        if limit:
            samples = samples[:limit]
        for action in actions:
            op = out_dir / f"{dataset}_{model_name}_{split}_{action}.jsonl"
            done = {r["sample_id"] for r in read_jsonl(op)} if op.exists() else set()
            todo = [s for s in samples if s["sample_id"] not in done]
            print(f"[{dataset}/{model_name}/{split}/{action}] {len(done)} done, {len(todo)} to do", flush=True)
            if not todo:
                continue
            bs = batch_size if action != "zoom" else max(1, batch_size // 4)
            t0 = time.time()
            with open(op, "a") as f:
                for i in range(0, len(todo), bs):
                    chunk = todo[i : i + bs]
                    msgs = []
                    for s in chunk:
                        ims, prompt = build_input(s, action)
                        msgs.append(make_messages(ims, prompt))
                    try:
                        texts, confs = generate_batch(model, processor, msgs)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        texts, confs = [], []
                        for m in msgs:
                            t, c = generate_batch(model, processor, [m])
                            texts += t
                            confs += c
                    for s, t, c in zip(chunk, texts, confs):
                        f.write(json.dumps(dict(
                            sample_id=s["sample_id"], action=action, answer_raw=t,
                            score=score(t, s["answer"]), confidence=c), ensure_ascii=False) + "\n")
                    f.flush()
                    if (i // bs) % 10 == 0:
                        el = time.time() - t0
                        print(f"  {i+len(chunk)}/{len(todo)}  {el/60:.1f} min  ex: {texts[0][:60]!r}", flush=True)
            print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "validation", "train"])
    ap.add_argument("--actions", nargs="+", default=ACTIONS)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    run(a.model, a.dataset, a.splits, a.actions, a.batch_size, a.limit)
