"""Selective accuracy on the official test split, one regime for every VLM: q+a_B (BiomedCLIP z_q+z_y0, StandardScaler + logistic C=0.1)
fitted on train+validation, items ranked by predicted error probability, accuracy of the answered fraction at coverage 0.9..0.5,
1000-sample bootstrap CIs. Also token log-probability for reference.
-> results/selective_qaB_test.csv, paper/figures/fig_risk_coverage.pdf """
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, read_jsonl
MODELS = ["medgemma", "gemma3", "qwen", "qwen3b", "qwen2vl2b"]; NAME = {"medgemma": "MedGemma-4B", "gemma3": "Gemma-3-4B", "qwen": "Qwen2.5-VL-7B", "qwen3b": "Qwen2.5-VL-3B", "qwen2vl2b": "Qwen2-VL-2B"}
BN = {"vqa_rad": "VQA-RAD", "slake": "SLAKE"}; COVS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

def load(d, m, s):
    z = np.load(RESULTS / "features" / f"{d}_{m}_{s}.npz"); K = {r["sample_id"]: r for r in read_jsonl(RESULTS / "raw" / f"{d}_{m}_{s}_keep.jsonl")}
    ids = list(z["sample_id"]); X = np.concatenate([z["z_q"], z["z_y0"]], 1); y = np.array([int(K[i]["score"] < 1) for i in ids])
    conf = np.nan_to_num(np.array([K[i].get("confidence", np.nan) for i in ids], float), nan=-50); return X, y, conf

def sel_acc(score, y, cov):
    o = np.argsort(score); return 1 - y[o[:max(1, int(round(cov * len(y))))]].mean()

rows = []; curves = {}
rng = np.random.RandomState(0)
for d in BN:
    for m in MODELS:
        Xtr, ytr, _ = load(d, m, "train"); Xva, yva, _ = load(d, m, "validation"); Xte, yte, conf = load(d, m, "test")
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
        p = clf.predict_proba(Xte)[:, 1]
        for sig, sc in [("q+a_B", p), ("log-prob", -conf)]:
            for cov in COVS:
                a = sel_acc(sc, yte, cov)
                bs = [sel_acc(sc[i], yte[i], cov) for i in (rng.randint(0, len(yte), len(yte)) for _ in range(1000))]
                rows.append(dict(dataset=d, model=m, signal=sig, coverage=cov, selective_acc=a, ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5)))
        curves[(d, m)] = (p, yte, -conf)
R = pd.DataFrame(rows); R.to_csv(RESULTS / "selective_qaB_test.csv", index=False)
print(R[(R.signal == "q+a_B") & R.coverage.isin([0.7, 0.5])].pivot_table(index=["dataset", "model"], columns="coverage", values="selective_acc").mul(100).round(1).to_string())
# figure: risk-coverage curves on the test split, q+a_B (band = bootstrap CI) and log-prob, two main VLMs per dataset
plt.rcParams.update({"font.size": 7})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
for ax, d in zip(axes, BN):
    for m, col in [("medgemma", "tab:blue"), ("qwen", "tab:orange")]:
        p, y, lp = curves[(d, m)]; covs = np.linspace(0.3, 1.0, 36)
        acc = np.array([sel_acc(p, y, c) for c in covs]); lo, hi = [], []
        for c in covs:
            bs = [sel_acc(p[i], y[i], c) for i in (rng.randint(0, len(y), len(y)) for _ in range(300))]; lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        ax.plot(100 * covs, 100 * acc, color=col, label=f"{NAME[m]} q+a$_B$"); ax.fill_between(100 * covs, 100 * np.array(lo), 100 * np.array(hi), color=col, alpha=0.15)
        ax.plot(100 * covs, [100 * sel_acc(lp, y, c) for c in covs], color=col, ls=":", lw=1, label=f"{NAME[m]} log-prob")
    ax.set_title(BN[d], fontsize=8); ax.set_xlabel("coverage (%)"); ax.grid(alpha=0.3); ax.set_xlim(30, 100)
axes[0].set_ylabel("selective accuracy (%)")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.19, 1, 1), pad=0.3)
(ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)
for out in [ROOT / "paper" / "figures" / "fig_risk_coverage"]:
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("done")
