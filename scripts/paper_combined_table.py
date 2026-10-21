"""One combined table for the short ICASSP version: per (benchmark, VLM) all headline quantities. -> paper/tables/combined.tex"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT
P = ROOT / "paper" / "tables"
E = pd.read_csv(RESULTS / "error_predictability.csv"); E = E[E.label == "A"]
CI = pd.read_csv(RESULTS / "paper_auroc_bootstrap.csv")
PR = pd.read_csv(RESULTS / "stageA_probe.csv"); CF = pd.read_csv(RESULTS / "stageA_counterfactual.csv"); SB = pd.read_csv(RESULTS / "stageB_matrix.csv")
RC = pd.read_csv(RESULTS / "selective_qaB_test.csv"); RC = RC[(RC.signal == "q+a_B") & (RC.coverage == 0.5)]  # q+a_B fitted on train+val, official test split (same regime for every VLM)
O = pd.read_csv(RESULTS / "oracle_analysis.csv"); O = O[O.split == "test"]
FIVE = ["medgemma", "gemma3", "qwen", "qwen3b", "qwen2vl2b"]; NAME = {"medgemma": "MedGemma-4B", "gemma3": "Gemma-3-4B", "qwen": "Qwen2.5-VL-7B", "qwen3b": "Qwen2.5-VL-3B", "qwen2vl2b": "Qwen2-VL-2B"}
rows = []
for d, dn in [("vqa_rad", "VQA-RAD"), ("slake", "SLAKE")]:
    for m in FIVE:
        sb = SB[(SB.benchmark == d) & (SB.model == m)].iloc[0]; cf = CF[(CF.dataset == d) & (CF.model == m)].iloc[0]; pr = PR[(PR.dataset == d) & (PR.model == m)].iloc[0]
        g = lambda k: f"{sb[k]:.2f}" if not np.isnan(sb[k]) else "--"
        main = (d, m) in [("vqa_rad", "medgemma"), ("vqa_rad", "qwen"), ("slake", "medgemma"), ("slake", "qwen")]
        # BiomedCLIP q+a and image-only, official train+val -> test, for every VLM (features now extracted for all)
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        from common import read_jsonl
        Z = {s: np.load(RESULTS / "features" / f"{d}_{m}_{s}.npz") for s in ["train", "validation", "test"]}
        K = {s: {r["sample_id"]: r["score"] for r in read_jsonl(RESULTS / "raw" / f"{d}_{m}_{s}_keep.jsonl")} for s in Z}
        Y = {s: np.array([int(K[s][i] < 1) for i in Z[s]["sample_id"]]) for s in Z}
        def auc(cols):
            F = {s: np.concatenate([Z[s][c] for c in cols], 1) for s in Z}
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(np.concatenate([F["train"], F["validation"]]), np.concatenate([Y["train"], Y["validation"]]))
            return roc_auc_score(Y["test"], clf.predict_proba(F["test"])[:, 1])
        qa, img = auc(["z_q", "z_y0"]), auc(["z_img"])
        mean_score = 100 * np.mean(list(K["test"].values()))
        sel = RC[(RC.dataset == d) & (RC.model == m)].selective_acc.iloc[0]
        rows.append(f"{dn} & {NAME[m]} & {mean_score:.1f} & {100*sb.acc:.1f} & {g('auroc_logprob')} & {g('auroc_sampling')} & {g('auroc_perturb')} & {img:.2f} & \\textbf{{{qa:.2f}}} & {sb.auroc_qa_test:.2f} & {pr.probe_test:.2f} & {100*cf.same_swap:.0f} & {100*cf.acc_if_insensitive:.0f} / {100*cf.acc_if_sensitive:.0f} & {100*sb.gap:.1f} & {100*sb.harm_mean:.1f} & {100*sel:.0f} \\\\")
    rows.append("\\midrule")
body = ("\\begin{tabular}{llrr|ccccccc|cc|cc|c}\\toprule & & \\multicolumn{2}{c|}{Base} & \\multicolumn{7}{c|}{Error-detection AUROC (test)} & \\multicolumn{2}{c|}{Image swap} & \\multicolumn{2}{c|}{Interventions} & Selective \\\\"
        " & & score & acc. & \\multicolumn{3}{c}{no training} & \\multicolumn{4}{c|}{trained} & & & & & \\\\ \\cmidrule(lr){5-7}\\cmidrule(lr){8-11}"
        " Dataset & VLM & & & log-p & samp. & pert. & img$_{\\rm B}$ & \\textbf{q+a$_{\\rm B}$} & q+a$_{\\rm T}$ & probe & unch.\\ \\% & acc.\\ unch./ch. & gap & harm & acc@50\\% \\\\\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\\end{tabular}")
(P / "combined.tex").write_text(body); print(body)
