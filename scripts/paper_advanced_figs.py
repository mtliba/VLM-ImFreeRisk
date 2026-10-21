"""Advanced visualisations, one per claim. -> paper/figures/adv_*.pdf (+png)"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, DATA, read_jsonl, ACTIONS
from canonical import canonicalize, closed_pred_canonical
F = ROOT / "paper" / "figures"
FIVE = ["medgemma", "gemma3", "qwen", "qwen3b", "qwen2vl2b"]; NAME = {"medgemma": "MedGemma-4B", "gemma3": "Gemma-3-4B", "qwen": "Qwen2.5-VL-7B", "qwen3b": "Qwen2.5-VL-3B", "qwen2vl2b": "Qwen2-VL-2B"}
DSN = {"vqa_rad": "VQA-RAD", "slake": "SLAKE"}; COL = {"medgemma": "#1f77b4", "gemma3": "#7fb3d5", "qwen": "#d62728", "qwen3b": "#f1948a", "qwen2vl2b": "#c49c94"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
def key(a, q):
    yn = closed_pred_canonical(a, q); return yn if yn in ("yes", "no") else canonicalize(a)
def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight"); fig.savefig(F / f"{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig); print("ok", name)
SB = pd.read_csv(RESULTS / "stageB_matrix.csv"); CF = pd.read_csv(RESULTS / "stageA_counterfactual.csv"); PR = pd.read_csv(RESULTS / "stageA_probe.csv"); ADV = pd.read_csv(RESULTS / "paper_advanced_pertype.csv")

# ---------------- 1. radar per dataset: 6 metrics x 5 VLMs
def radar():
    metrics = [("acc", "accuracy"), ("auroc_qa_test", "text-prior AUROC"), ("auroc_probe_test", "probe AUROC"), ("swap_insensitive", "swap-insensitive"), ("frac_correct_insensitive", "correct answers\nimage-insensitive"), ("gap", "oracle gap\n(×5)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), subplot_kw=dict(polar=True), gridspec_kw={"wspace": 0.55})
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist(); ang += ang[:1]
    for ax, d in zip(axes, DSN):
        for m in FIVE:
            r = SB[(SB.benchmark == d) & (SB.model == m)].iloc[0]
            vals = [r.acc, r.auroc_qa_test, r.auroc_probe_test, r.swap_insensitive, r.frac_correct_insensitive, min(1, 5 * r.gap)]; vals += vals[:1]
            ax.plot(ang, vals, color=COL[m], lw=1.4, label=NAME[m]); ax.fill(ang, vals, color=COL[m], alpha=0.06)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels([l for _, l in metrics], fontsize=6.5); ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["", ".5", "", "1"], fontsize=6); ax.set_title(DSN[d], fontsize=9, pad=12)
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.45, -0.15), fontsize=6.5, frameon=False)
    save(fig, "adv_radar")

# ---------------- 2. UpSet-style plot of error sets
def upset():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), gridspec_kw={"height_ratios": [2.2, 1.3], "hspace": 0.05, "wspace": 0.25})
    for c, d in enumerate(DSN):
        E = {m: {r["sample_id"]: r["score"] < 1 for r in read_jsonl(RESULTS / "raw" / f"{d}_{m}_test_keep.jsonl")} for m in FIVE}
        ids = sorted(set.intersection(*[set(E[m]) for m in FIVE])); X = np.array([[E[m][i] for m in FIVE] for i in ids], bool)
        pats = {}
        for row in X: pats[tuple(row)] = pats.get(tuple(row), 0) + 1
        pats = sorted(pats.items(), key=lambda kv: -kv[1])[:14]
        ax = axes[0, c]; xs = np.arange(len(pats)); counts = [v for _, v in pats]
        ax.bar(xs, counts, color=["#d62728" if all(p) else ("#2ca02c" if not any(p) else "#7f7f7f") for p, _ in pats])
        for x, v in zip(xs, counts): ax.text(x, v + 3, f"{100*v/len(ids):.0f}%", ha="center", fontsize=6)
        ax.set_xticks([]); ax.set_ylabel("# test items"); ax.set_title(f"{DSN[d]}: error patterns (red = all 5 wrong, green = none wrong)", fontsize=7.5)
        ax = axes[1, c]
        for x, (p, _) in enumerate(pats):
            for j, on in enumerate(p):
                ax.scatter(x, j, s=28, color="#333" if on else "#ddd", zorder=3)
            on_idx = [j for j, on in enumerate(p) if on]
            if len(on_idx) > 1: ax.plot([x, x], [min(on_idx), max(on_idx)], color="#333", lw=1.2, zorder=2)
        ax.set_yticks(range(5)); ax.set_yticklabels([NAME[m] for m in FIVE], fontsize=6.5); ax.set_xticks([]); ax.set_ylim(-0.6, 4.6); ax.invert_yaxis()
        for s in ["top", "right", "bottom", "left"]: ax.spines[s].set_visible(False)
    save(fig, "adv_upset")

# ---------------- 3. transition matrices real -> swap
def transitions():
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.2))
    for r, d in enumerate(DSN):
        S = {x["sample_id"]: x for x in read_jsonl(DATA / d / "test.jsonl")}
        for c, m in enumerate(FIVE):
            K = {x["sample_id"]: x for x in read_jsonl(RESULTS / "raw" / f"{d}_{m}_test_keep.jsonl")}; W = {x["sample_id"]: x for x in read_jsonl(RESULTS / "counterfactual" / f"{d}_{m}_test_swap.jsonl")}
            ids = [i for i in K if i in W]; ok0 = np.array([K[i]["score"] >= 1 for i in ids]); ok1 = np.array([W[i]["score"] >= 1 for i in ids]); same = np.array([key(W[i]["answer_raw"], S[i]["question"]) == key(K[i]["answer_raw"], S[i]["question"]) for i in ids])
            M = np.array([[(ok0 & ok1).mean(), (ok0 & ~ok1).mean()], [(~ok0 & ok1).mean(), (~ok0 & ~ok1).mean()]])
            ax = axes[r, c]; ax.imshow(M, cmap="Blues", vmin=0, vmax=0.6)
            for i in range(2):
                for j in range(2): ax.text(j, i, f"{100*M[i,j]:.0f}%", ha="center", va="center", fontsize=7, color="w" if M[i, j] > 0.35 else "k")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["correct", "wrong"], fontsize=6); ax.set_yticklabels(["correct", "wrong"], fontsize=6)
            if c == 0: ax.set_ylabel(f"{DSN[d]}\nwith real image", fontsize=7)
            if r == 1: ax.set_xlabel("with swapped image", fontsize=7)
            ax.set_title(f"{NAME[m]}\nanswer unchanged {100*same.mean():.0f}%", fontsize=6.5)
    fig.tight_layout(); save(fig, "adv_transitions")

# ---------------- 4. per-type scatter: error rate vs predictability, size n, colour swap sensitivity
def pertype_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, d in zip(axes, DSN):
        g = ADV[(ADV.dataset == d) & (~ADV.qtype.str.startswith("["))]
        g = g.groupby("qtype").agg(err=("err", "mean"), auroc=("auroc", "mean"), n=("n", "first"), sens=("swap_sensitive", "mean")).reset_index()
        sc = ax.scatter(100 * g.err, g.auroc, s=8 + g.n / 12, c=100 * g.sens, cmap="coolwarm", vmin=20, vmax=70, edgecolor="k", lw=0.4)
        for _, r in g.iterrows(): ax.annotate(r.qtype, (100 * r.err, r.auroc), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.axhline(0.5, ls="--", lw=0.6, color="k"); ax.set_xlabel("error rate of the question type (%)"); ax.set_ylabel("error-detection AUROC (text prior)"); ax.set_title(f"{DSN[d]} (mean over 5 VLMs; size = # QA)", fontsize=7.5)
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02); cb.set_label("image-swap sensitivity (%)", fontsize=7)
    save(fig, "adv_pertype_scatter")

# ---------------- 5. per-type polar (MedGemma & Qwen on SLAKE): error rate, AUROC, swap sensitivity
def pertype_polar():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), subplot_kw=dict(polar=True), gridspec_kw={"wspace": 0.55})
    for ax, m in zip(axes, ["medgemma", "qwen"]):
        g = ADV[(ADV.dataset == "slake") & (ADV.model == m) & (~ADV.qtype.str.startswith("["))].sort_values("qtype")
        ang = np.linspace(0, 2 * np.pi, len(g), endpoint=False).tolist(); ang += ang[:1]
        for col, lab, c in [("err", "error rate", "#d62728"), ("auroc", "text-prior AUROC", "#1f77b4"), ("swap_sensitive", "image-swap sensitivity", "#2ca02c")]:
            v = g[col].tolist(); v += v[:1]; ax.plot(ang, v, color=c, lw=1.4, label=lab); ax.fill(ang, v, color=c, alpha=0.07)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels([t.title() for t in g.qtype], fontsize=6.5); ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["", ".5", "", "1"], fontsize=6); ax.set_title(f"SLAKE / {NAME[m]}", fontsize=9, pad=12)
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.4, -0.15), fontsize=6.5, frameon=False)
    save(fig, "adv_pertype_polar")

# ---------------- 6. convention gallery: same question across 6 images
def gallery(question_substr="how many kidneys", n=6, name="adv_gallery_kidneys"):
    S = {x["sample_id"]: x for x in read_jsonl(DATA / "slake" / "test.jsonl")}
    K = {m: {x["sample_id"]: x for x in read_jsonl(RESULTS / "raw" / f"slake_{m}_test_keep.jsonl")} for m in ["medgemma", "qwen"]}
    ids = [i for i in S if question_substr in S[i]["question"].lower()][:n]
    # images: fetch locally cached copies (paper/figures/qual/) if present, else skip image panel
    fig, axes = plt.subplots(2, n, figsize=(7.2, 3.0), gridspec_kw={"height_ratios": [1, 0.55], "hspace": 0.05})
    for j, i in enumerate(ids):
        p = F / "qual" / S[i]["image_path"].split("/data/")[1].replace("/", "_")
        ax = axes[0, j]; ax.axis("off")
        if p.exists(): ax.imshow(Image.open(p).convert("RGB"))
        else: ax.text(0.5, 0.5, "image\n(on cluster)", ha="center", va="center", fontsize=6, transform=ax.transAxes)
        ax.set_title(f"ref: {S[i]['answer']}", fontsize=7)
        ax = axes[1, j]; ax.axis("off")
        for k, (m, nm) in enumerate([("medgemma", "MG"), ("qwen", "Q7")]):
            a = K[m][i]["answer_raw"]; ok = K[m][i]["score"] >= 1
            ax.text(0.5, 0.75 - 0.5 * k, f"{nm}: {a}", ha="center", va="center", fontsize=7, color="#2ca02c" if ok else "#d62728", transform=ax.transAxes)
    fig.suptitle(f"SLAKE test — “{S[ids[0]]['question']}” across images (green = credited, red = scored wrong)", fontsize=8, y=1.02)
    save(fig, name); return ids

# ---------------- 7. headroom realisation: oracle gap vs every policy's realised gain and harm
def realisation():
    R = pd.read_csv(RESULTS / "revised_selector_results.csv"); R = R[R.metric == "A"]; R["config"] = R.config.fillna("")
    CS = pd.read_csv(RESULTS / "consensus_stage1.csv"); CS = CS[CS.metric == "A"]
    S4 = [("vqa_rad", "medgemma"), ("vqa_rad", "qwen"), ("slake", "medgemma"), ("slake", "qwen")]; lab = ["VQA-RAD\nMedGemma", "VQA-RAD\nQwen-7B", "SLAKE\nMedGemma", "SLAKE\nQwen-7B"]
    pol = [("Best fixed", "Best fixed", "", R), ("Q-type router", "Question router", "", R), ("Direct ΔQ", "Direct dQ selector", "per_setting/biomedclip/iqy", R), ("Gate+classifier", "Gate + action classifier", "per_setting/biomedclip/iqy", R), ("Consensus (2-of-3)", "2-intervention corroboration", None, CS), ("Gate+consensus", "Gate + corroboration", None, CS), ("Oracle", "Oracle", "", R)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]; w = 0.11
    for j, (l, meth, cfg, df) in enumerate(pol):
        ys = []
        for d, m in S4:
            base = R[(R.dataset == d) & (R.model == m) & (R.method == "Base")].score.iloc[0]
            g = df[(df.dataset == d) & (df.model == m) & ((df.method == meth) if "method" in df else (df.policy == meth))]
            if cfg is not None and "config" in g: g = g[g.config == cfg]
            ys.append(100 * (g.score.mean() - base))
        ax.bar(np.arange(4) + (j - 3) * w, ys, w, label=l, color=plt.cm.tab10(j) if l != "Oracle" else "#333")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=7); ax.set_ylabel("gain over initial answer (pts)"); ax.legend(fontsize=5.5, ncol=2, frameon=False); ax.set_title("How much of the oracle headroom each policy realises", fontsize=8)
    ax = axes[1]
    for j, (l, meth, cfg, df) in enumerate(pol[:-1]):
        for k, (d, m) in enumerate(S4):
            g = df[(df.dataset == d) & (df.model == m) & ((df.method == meth) if "method" in df else (df.policy == meth))]
            if cfg is not None and "config" in g: g = g[g.config == cfg]
            harm = g.harmful_rate.mean() if "harmful_rate" in g else g.harmful_correction_rate.mean()
            base = R[(R.dataset == d) & (R.model == m) & (R.method == "Base")].score.iloc[0]
            ax.scatter(100 * harm, 100 * (g.score.mean() - base), color=plt.cm.tab10(j), marker="os^D*P"[k % 6], s=28, label=l if k == 0 else None)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(5, color="k", lw=0.6, ls="--"); ax.set_xlabel("harmful fraction of interventions / changed answers (%)"); ax.set_ylabel("gain (pts)"); ax.set_title("gain vs harm (marker = setting; dashed = 5% harm)", fontsize=8); ax.legend(fontsize=5, frameon=False)
    save(fig, "adv_realisation")

if __name__ == "__main__":
    radar(); upset(); transitions(); pertype_scatter(); pertype_polar(); realisation(); ids = gallery()
    print("gallery ids:", ids)
