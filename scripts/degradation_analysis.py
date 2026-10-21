"""J5 analysis: accuracy and answer stability under progressive image degradation, split by real-image correctness and by swap-sensitivity.
-> results/degradation_summary.csv, paper/figures/fig_degradation.pdf, paper/tables/degradation.tex"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonicalize, closed_pred_canonical
from common import DATA, RESULTS, ROOT, read_jsonl
ORDER = ["real", "blur2", "blur4", "blur8", "blur16", "down32", "down16", "down8", "noise0.25", "noise0.5", "blank", "noimage"]
LAB = {"real": "real", "blur2": "blur 2", "blur4": "blur 4", "blur8": "blur 8", "blur16": "blur 16", "down32": "32²", "down16": "16²", "down8": "8²", "noise0.25": "noise .25", "noise0.5": "noise .5", "blank": "blank", "noimage": "no image"}
NAME = {"medgemma": "MedGemma-4B", "gemma3": "Gemma-3-4B", "qwen": "Qwen2.5-VL-7B", "qwen3b": "Qwen2.5-VL-3B", "qwen2vl2b": "Qwen2-VL-2B", "lingshu7b": "Lingshu-7B", "internvl3_8b": "InternVL3-8B", "qwen3vl8b": "Qwen3-VL-8B"}
BN = {"vqa_rad": "VQA-RAD", "slake": "SLAKE", "pathvqa": "PathVQA", "pmcvqa": "PMC-VQA", "omnimedvqa": "OmniMedVQA"}
ALLB = list(BN); ALLM = list(NAME)
def key(a, q):
    yn = closed_pred_canonical(a, q); return yn if yn in ("yes", "no") else canonicalize(a)
rows = []
for d in ALLB:
    S = {r["sample_id"]: r for r in read_jsonl(DATA / d / "test.jsonl")}
    for m in ALLM:
        if not (RESULTS / "degradation" / f"{d}_{m}_test_blur8.jsonl").exists(): continue
        K = {r["sample_id"]: r for r in read_jsonl(RESULTS / "raw" / f"{d}_{m}_test_keep.jsonl")}
        C = {c: {r["sample_id"]: r for r in read_jsonl(RESULTS / "counterfactual" / f"{d}_{m}_test_{c}.jsonl")} for c in ["swap", "blank", "noimage"]}
        D = {}
        for c in ORDER[1:-2]:
            p = RESULTS / "degradation" / f"{d}_{m}_test_{c}.jsonl"
            if p.exists(): D[c] = {r["sample_id"]: r for r in read_jsonl(p)}
        D["blank"], D["noimage"] = C["blank"], C["noimage"]
        ids = [i for i in K if all(i in D[c] for c in D) and i in C["swap"]]
        k0 = {i: key(K[i]["answer_raw"], S[i]["question"]) for i in ids}; ok0 = np.array([K[i]["score"] >= 1 for i in ids])
        insens = np.array([key(C["swap"][i]["answer_raw"], S[i]["question"]) == k0[i] for i in ids])
        for c in ["real"] + [c for c in ORDER[1:] if c in D]:
            R = K if c == "real" else D[c]
            acc = np.array([R[i]["score"] >= 1 for i in ids]); unch = np.array([key(R[i]["answer_raw"], S[i]["question"]) == k0[i] for i in ids])
            rows.append(dict(dataset=d, model=m, cond=c, n=len(ids), acc=acc.mean(), unchanged=unch.mean(), unch_correct=unch[ok0].mean(), unch_wrong=unch[~ok0].mean(),
                             unch_insens=unch[insens].mean(), unch_sens=unch[~insens].mean(), acc_insens=acc[insens].mean(), acc_sens=acc[~insens].mean()))
R = pd.DataFrame(rows); R.to_csv(RESULTS / "degradation_summary.csv", index=False); print(R.round(3).to_string())
# figure: per (dataset, model): accuracy curve + unchanged (correct vs wrong) curve
plt.rcParams.update({"font.size": 8})
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex="col")
for c, (d, m) in enumerate([(d, m) for d in ["vqa_rad", "slake"] for m in ["medgemma", "qwen"]]):
    g = R[(R.dataset == d) & (R.model == m)].set_index("cond"); conds = [x for x in ORDER if x in g.index]; x = np.arange(len(conds))
    ax = axs[0, c]; ax.plot(x, 100 * g.loc[conds, "acc"], "-o", ms=3, color="k", label="accuracy"); ax.axhline(100 * g.loc["noimage", "acc"], ls=":", color="grey", lw=0.8)
    ax.set_title(f"{BN[d]} / {NAME[m]}", fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 80)
    if c == 0: ax.set_ylabel("accuracy (%)")
    ax = axs[1, c]; ax.plot(x, 100 * g.loc[conds, "unch_correct"], "-o", ms=3, color="#2ca02c", label="answer unchanged: correct items"); ax.plot(x, 100 * g.loc[conds, "unch_wrong"], "-s", ms=3, color="#d62728", label="answer unchanged: wrong items")
    ax.set_xticks(x); ax.set_xticklabels([LAB[k] for k in conds], rotation=60, ha="right", fontsize=6); ax.grid(alpha=0.3); ax.set_ylim(0, 100)
    if c == 0: ax.set_ylabel("unchanged answers (%)")
h, l = axs[1, 0].get_legend_handles_labels(); h2, l2 = axs[0, 0].get_legend_handles_labels()
fig.legend(h2 + h, l2 + l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout(rect=(0, 0.06, 1, 1))
(ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True); plt.savefig(ROOT / "paper" / "figures" / "fig_degradation.pdf", bbox_inches="tight")
plt.close()
# all-pairs figure: every available (dataset, model): unchanged-correct minus unchanged-wrong gap vs level, one panel per dataset
pairs = sorted(set(zip(R.dataset, R.model)), key=lambda t: (ALLB.index(t[0]), ALLM.index(t[1])))
dsets = [d for d in ALLB if any(p[0] == d for p in pairs)]
fig2, axs2 = plt.subplots(2, len(dsets), figsize=(2.6 * len(dsets), 4.2), sharex="col", squeeze=False)
cols = dict(zip(ALLM, plt.cm.tab10(np.arange(10))))
for c, d in enumerate(dsets):
    for m in [m for m in ALLM if (d, m) in pairs]:
        g = R[(R.dataset == d) & (R.model == m)].set_index("cond"); conds = [x for x in ORDER if x in g.index]; x = np.arange(len(conds))
        axs2[0, c].plot(x, 100 * g.loc[conds, "acc"], "-o", ms=2.5, color=cols[m], label=NAME[m])
        axs2[1, c].plot(x, 100 * (g.loc[conds, "unch_correct"] - g.loc[conds, "unch_wrong"]), "-o", ms=2.5, color=cols[m])
    axs2[0, c].set_title(BN[d], fontsize=8); axs2[1, c].set_xticks(x); axs2[1, c].set_xticklabels([LAB[k] for k in conds], rotation=60, ha="right", fontsize=6)
    axs2[1, c].axhline(0, color="grey", lw=0.6); [a.grid(alpha=0.3) for a in axs2[:, c]]
axs2[0, 0].set_ylabel("accuracy (%)"); axs2[1, 0].set_ylabel("unchanged: correct - wrong (pts)")
h, l = axs2[0, 0].get_legend_handles_labels()
if len(dsets) > 1:
    for c in range(1, len(dsets)):
        h2, l2 = axs2[0, c].get_legend_handles_labels(); [ (h.append(a), l.append(b)) for a, b in zip(h2, l2) if b not in l ]
fig2.legend(h, l, loc="lower center", ncol=min(8, len(l)), frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
fig2.tight_layout(rect=(0, 0.07, 1, 1)); fig2.savefig(ROOT / "paper" / "figures" / "fig_degradation_all.pdf", bbox_inches="tight"); plt.close(fig2)
# table (compact): accuracy and unchanged-correct / unchanged-wrong at selected levels
sel = ["real", "blur4", "blur8", "blur16", "down16", "down8", "noise0.5", "blank", "noimage"]
lines = []
for d, m in pairs:
        g = R[(R.dataset == d) & (R.model == m)].set_index("cond")
        cells = " & ".join(f"{100*g.loc[c,'acc']:.0f} / {100*g.loc[c,'unch_correct']:.0f} / {100*g.loc[c,'unch_wrong']:.0f}" if c in g.index else "--" for c in sel)
        lines.append(f"{BN[d]} & {NAME[m]} & {cells} \\\\")
body = r"Dataset & VLM & " + " & ".join(LAB[c] for c in sel) + r" \\\midrule" + "\n" + "\n".join(lines) + "\n"
tex_out = (r"\begin{table*}[t]\centering\caption{Image degradation (test). Cells: accuracy / \% of items answered correctly with the real image whose answer is unchanged / the same for items answered wrongly. Blur = Gaussian $\sigma$ in px on a 512-px image; $k^2$ = box-downsampled to $k\times k$; noise = additive Gaussian, std as a fraction of the intensity range.}\label{tab:degr}\scriptsize\resizebox{\textwidth}{!}{\begin{tabular}{ll" + "c" * len(sel) + r"}\toprule" + "\n" + body + r"\bottomrule\end{tabular}}\end{table*}" + "\n")
(ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True); open(ROOT / "paper" / "tables" / "degradation.tex", "w").write(tex_out)
print("done")
