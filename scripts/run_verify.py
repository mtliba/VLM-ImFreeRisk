"""Anchored verification interventions (VERIFY_GLOBAL / VERIFY_ENHANCE / VERIFY_ZOOM) on the existing initial (KEEP) answers.

Output: results/verification_raw/{dataset}_{model}_{split}_{verifier}.jsonl with
    sample_id, verifier, initial_answer, output_raw, decision (KEEP|CORRECT|INVALID), final_answer, score, score_b, confidence
Resumable. Greedy decoding, max_new_tokens 32, same image transforms as before (common.enhance_image / zoom_views).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import score_canonical  # noqa: E402
from common import DATA, MODEL_IDS, RESULTS, enhance_image, load_prompt, read_jsonl, score, zoom_views  # noqa: E402
from run_vlm import generate_batch, load_model, make_messages  # noqa: E402

VERIFIERS = ["global", "enhance", "zoom"]


def parse(out):
    t = out.strip()
    if re.match(r"^\W*keep\b", t, flags=re.I):
        return "KEEP", None
    m = re.match(r"^\W*correct\s*:\s*(.+)$", t, flags=re.I | re.S)
    if m:
        ans = m.group(1).strip().split("\n")[0].strip().strip('"').strip("*").strip()
        if ans:
            return "CORRECT", ans
    return "INVALID", None


def build(sample, initial, verifier):
    img = Image.open(sample["image_path"]).convert("RGB")
    q = sample["question"].strip()
    prompt = load_prompt(f"verify_{verifier}").format(initial_answer=initial, question=q)
    if verifier == "global":
        return [img], prompt
    if verifier == "enhance":
        return [enhance_image(img)], prompt
    return zoom_views(img), prompt


def run(model_name, dataset, splits, verifiers, batch_size, limit=None):
    model, processor = load_model(model_name)
    out_dir = RESULTS / "verification_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        samples = read_jsonl(DATA / dataset / f"{split}.jsonl")
        if limit:
            samples = samples[:limit]
        keep = {r["sample_id"]: r["answer_raw"] for r in read_jsonl(RESULTS / "raw" / f"{dataset}_{model_name}_{split}_keep.jsonl")}
        for v in verifiers:
            op = out_dir / f"{dataset}_{model_name}_{split}_{v}.jsonl"
            done = {r["sample_id"] for r in read_jsonl(op)} if op.exists() else set()
            todo = [s for s in samples if s["sample_id"] not in done and s["sample_id"] in keep]
            print(f"[{dataset}/{model_name}/{split}/verify_{v}] {len(done)} done, {len(todo)} to do", flush=True)
            bs = batch_size if v != "zoom" else max(1, batch_size // 4)
            t0 = time.time()
            with open(op, "a") as f:
                for i in range(0, len(todo), bs):
                    chunk = todo[i:i + bs]
                    msgs = [make_messages(*build(s, keep[s["sample_id"]], v)) for s in chunk]
                    try:
                        texts, confs = generate_batch(model, processor, msgs)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache(); texts, confs = [], []
                        for m_ in msgs:
                            t, c = generate_batch(model, processor, [m_]); texts += t; confs += c
                    for s, t, c in zip(chunk, texts, confs):
                        dec, ans = parse(t)
                        final = ans if dec == "CORRECT" else keep[s["sample_id"]]
                        f.write(json.dumps(dict(sample_id=s["sample_id"], verifier=v, initial_answer=keep[s["sample_id"]], output_raw=t, decision=dec,
                                                final_answer=final, score=score(final, s["answer"]), score_b=score_canonical(final, s["answer"], s["question"]),
                                                confidence=c), ensure_ascii=False) + "\n")
                    f.flush()
                    if (i // bs) % 10 == 0:
                        print(f"  {i+len(chunk)}/{len(todo)} {(time.time()-t0)/60:.1f} min ex: {texts[0][:50]!r}", flush=True)
            print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    ap.add_argument("--dataset", required=True, choices=["vqa_rad", "slake"])
    ap.add_argument("--splits", nargs="+", default=["validation"])
    ap.add_argument("--verifiers", nargs="+", default=VERIFIERS)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    run(a.model, a.dataset, a.splits, a.verifiers, a.batch_size, a.limit)
