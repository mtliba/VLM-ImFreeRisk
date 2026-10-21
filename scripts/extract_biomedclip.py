"""Frozen BiomedCLIP embeddings for image, question and base answer (per VLM).

Output: results/features/{dataset}_{model}_{split}.npz with
    sample_id, z_img (512), z_q (512), z_y0 (512)
Optionally --encoder clip  -> generic OpenAI CLIP ViT-B/16 (ablation), written to features_clip/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, read_jsonl  # noqa: E402

ENCODERS = {
    "biomedclip": ("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", None),
    "clip": ("ViT-B-16", "openai"),
}


@torch.no_grad()
def main(encoder, datasets, models, splits, bs=64):
    name, pretrained = ENCODERS[encoder]
    if pretrained:
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(name)
    else:
        model, preprocess = open_clip.create_model_from_pretrained(name)
        tokenizer = open_clip.get_tokenizer(name)
    model = model.cuda().eval()
    out_dir = RESULTS / ("features" if encoder == "biomedclip" else "features_clip")
    out_dir.mkdir(parents=True, exist_ok=True)

    def enc_text(texts):
        out = []
        for i in range(0, len(texts), bs):
            t = tokenizer(texts[i:i + bs], context_length=256 if encoder == "biomedclip" else 77).cuda()
            out.append(torch.nn.functional.normalize(model.encode_text(t), dim=-1).float().cpu())
        return torch.cat(out).numpy()

    img_cache = {}

    def enc_images(paths):
        todo = [p for p in dict.fromkeys(paths) if p not in img_cache]
        for i in range(0, len(todo), bs):
            ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in todo[i:i + bs]]).cuda()
            z = torch.nn.functional.normalize(model.encode_image(ims), dim=-1).float().cpu().numpy()
            for p, zz in zip(todo[i:i + bs], z):
                img_cache[p] = zz
        return np.stack([img_cache[p] for p in paths])

    for ds in datasets:
        for split in splits:
            samples = read_jsonl(DATA / ds / f"{split}.jsonl")
            for m in models:
                kp = RESULTS / "raw" / f"{ds}_{m}_{split}_keep.jsonl"
                if not kp.exists():
                    print("missing", kp); continue
                base = {r["sample_id"]: r["answer_raw"] for r in read_jsonl(kp)}
                ss = [s for s in samples if s["sample_id"] in base]
                z_img = enc_images([s["image_path"] for s in ss])
                z_q = enc_text([s["question"] for s in ss])
                z_y0 = enc_text([base[s["sample_id"]] or "" for s in ss])
                np.savez(out_dir / f"{ds}_{m}_{split}.npz", sample_id=np.array([s["sample_id"] for s in ss]),
                         z_img=z_img, z_q=z_q, z_y0=z_y0)
                print(ds, m, split, z_img.shape, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="biomedclip", choices=list(ENCODERS))
    ap.add_argument("--datasets", nargs="+", default=["vqa_rad", "slake"])
    ap.add_argument("--models", nargs="+", default=["medgemma", "qwen"])
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    a = ap.parse_args()
    main(a.encoder, a.datasets, a.models, a.splits)
