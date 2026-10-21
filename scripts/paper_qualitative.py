"""Qualitative figure: 4 cases x (real image | swapped image | answers under real/swap/no-image/zoom for MedGemma & Qwen-7B,
text-prior P(error), 5-VLM correctness strip). -> paper/figures/fig_qualitative.pdf"""
import sys
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
Q = ROOT / "paper" / "figures" / "qual"
rows = pd.read_pickle("/tmp/sel.pkl")
labels = {"slake_test_12750": "(a) prior-driven correct", "vqarad_test_242": "(b) image-sensitive, wrong", "slake_test_12115": "(c) convention error", "slake_test_12253": "(d) visual yes/no error"}
FIVE = ["MedGemma", "Gemma-3", "Qwen-7B", "Qwen-3B", "Qwen2-2B"]
def lp(p): return Q / (p.split("/data/")[1].replace("/", "_"))
fig = plt.figure(figsize=(7.2, 7.0)); gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 2.6], hspace=0.35, wspace=0.08)
for r, (sid, row) in enumerate(rows.iterrows()):
    for c, (p, t) in enumerate([(row.image, "real image"), (row.swap_image, "swapped image")]):
        ax = fig.add_subplot(gs[r, c]); ax.imshow(Image.open(lp(p)).convert("RGB")); ax.axis("off"); ax.set_title(t, fontsize=7, pad=2)
    ax = fig.add_subplot(gs[r, 2]); ax.axis("off")
    ref = row.ref if len(row.ref) < 40 else row.ref[:38] + "…"
    lines = [f"{labels[sid]}   [{row.dataset.upper().replace('_','-')}]", f"Q: {row.question}", f"reference: {ref}      text-prior P(error) = {row.p_err:.2f}", ""]
    lines.append(f"{'':12s}{'real':>12s}{'swapped':>12s}{'no image':>12s}{'zoom':>12s}")
    for m, nm in [("medgemma", "MedGemma-4B"), ("qwen", "Qwen-7B")]:
        a = [str(row[f"{m}_{k}"]) for k in ["real", "swap", "noimg", "zoom"]]; a = [x if len(x) <= 11 else x[:10] + "…" for x in a]
        lines.append(f"{nm:12s}" + "".join(f"{x:>12s}" for x in a))
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=6.6, transform=ax.transAxes)
    # 5-model correctness strip
    for j, (ok, nm) in enumerate(zip(row.correct, FIVE)):
        ax.add_patch(plt.Rectangle((0.02 + j * 0.19, 0.02), 0.17, 0.13, transform=ax.transAxes, color="#2ca02c" if ok else "#d62728", alpha=0.8))
        ax.text(0.105 + j * 0.19, 0.085, nm, ha="center", va="center", fontsize=5.5, color="w", transform=ax.transAxes)
fig.savefig(ROOT / "paper" / "figures" / "fig_qualitative.pdf", bbox_inches="tight"); fig.savefig(ROOT / "paper" / "figures" / "fig_qualitative.png", dpi=150, bbox_inches="tight"); print("ok")
