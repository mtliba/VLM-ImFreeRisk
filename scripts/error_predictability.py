"""New analysis: are medical-VLM errors on radiology VQA predictable WITHOUT the image?

Error label: initial (KEEP) answer wrong (Metric A: Q < 1; also Metric B). Predictors (5-fold GroupKFold by image over
train+val+test, so no image leaks across folds; plus the official train+val -> test split):
  - frozen BiomedCLIP embeddings: q, y0, q+y0, img, img+q, img+q+y0 (logistic regression, C=0.1)
  - generic CLIP img+q+y0
  - lookup tables: (question_type x canonical answer), (question text x canonical answer) -> smoothed error rate
  - zero-cost consistency: number of interventions (REPROMPT/ENHANCE/ZOOM) agreeing with KEEP; mean token log-prob
  - combinations; cross-VLM transfer; error overlap between VLMs; selective-answering accuracy at fixed coverage.
Outputs: results/error_predictability.csv, results/error_predictability_transfer.csv, results/error_predictability_coverage.csv,
         tables/error_predictability.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consensus_stage1 import load  # noqa: E402
from common import ACTIONS, RESULTS, ROOT  # noqa: E402

SETTINGS = [(d, m) for d in ["vqa_rad", "slake"] for m in ["medgemma", "qwen"]]
SPLITS = ["train", "validation", "test"]
COVS = [0.9, 0.8, 0.7, 0.5]


def gather(M, ds, m, enc="features"):
    Z = {k: [] for k in ["z_img", "z_q", "z_y0"]}; ya, yb, grp, E, meta, split = [], [], [], [], [], []
    for s in SPLITS:
        z = np.load(RESULTS / enc / f"{ds}_{m}_{s}.npz"); ids = list(z["sample_id"])
        te = M[(M.dataset == ds) & (M.model == m) & (M.split == s)].set_index("sample_id").loc[ids]
        K = te[[f"{a}_key" for a in ACTIONS]].values
        agree = np.array([sum(K[i, j] == K[i, 0] for j in range(1, 4)) for i in range(len(te))], dtype=float)
        conf = te.base_conf.fillna(te.base_conf.min()).values.astype(float)
        for k in Z:
            Z[k].append(z[k])
        ya.append((te.keep_score.values < 1).astype(int)); yb.append((pd.to_numeric(te.keep_score_b).values < 1).astype(int))
        grp.append(te.image_path.values); E.append(np.stack([agree, conf], 1)); split.append(np.full(len(te), s))
        meta.append(pd.DataFrame(dict(qt=te.question_type.values, q=te.question.str.lower().str.strip().values, k=te.keep_key.values)))
    return {k: np.concatenate(v) for k, v in Z.items()}, np.concatenate(ya), np.concatenate(yb), np.concatenate(grp), np.concatenate(E), pd.concat(meta).reset_index(drop=True), np.concatenate(split)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def cv_pred(F, y, folds):
    p = np.zeros(len(y))
    for tr, ho in folds:
        p[ho] = clf().fit(F[tr], y[tr]).predict_proba(F[ho])[:, 1]
    return p


def lookup_pred(meta, y, folds, cols):
    p = np.zeros(len(y)); prior = y.mean()
    for tr, ho in folds:
        d = meta.iloc[tr].copy(); d["y"] = y[tr]; tab = d.groupby(cols).y.agg(["sum", "count"])
        keys = list(zip(*[meta.iloc[ho][c] for c in cols])) if len(cols) > 1 else list(meta.iloc[ho][cols[0]])
        for i, kk in zip(ho, keys):
            p[i] = (tab.loc[kk, "sum"] + 2 * prior) / (tab.loc[kk, "count"] + 2) if kk in tab.index else prior
    return p


def main():
    M = load(); M["keep_score_b"] = pd.to_numeric(M.keep_score_b)
    rows, cov_rows, tr_rows = [], [], []
    cache = {}
    for ds, m in SETTINGS:
        Z, ya, yb, grp, E, meta, split = gather(M, ds, m)
        Zc = gather(M, ds, m, "features_clip")[0]
        cache[(ds, m)] = (Z, ya, split)
        feats = {"q": Z["z_q"], "y0": Z["z_y0"], "q+y0": np.concatenate([Z["z_q"], Z["z_y0"]], 1), "img": Z["z_img"], "img+q": np.concatenate([Z["z_img"], Z["z_q"]], 1),
                 "img+q+y0": np.concatenate([Z["z_img"], Z["z_q"], Z["z_y0"]], 1), "clip img+q+y0": np.concatenate([Zc["z_img"], Zc["z_q"], Zc["z_y0"]], 1),
                 "agree": E[:, :1], "conf": E[:, 1:], "agree+conf": E, "img+q+y0+agree+conf": np.concatenate([Z["z_img"], Z["z_q"], Z["z_y0"], E], 1),
                 "q+y0+agree+conf": np.concatenate([Z["z_q"], Z["z_y0"], E], 1)}
        for lab, y in [("A", ya), ("B", yb)]:
            folds = list(GroupKFold(5).split(feats["q"], y, grp))
            is_te = split == "test"; is_tr = ~is_te
            preds = {}
            for name, F in feats.items():
                preds[name] = cv_pred(F, y, folds)
                p_off = clf().fit(F[is_tr], y[is_tr]).predict_proba(F[is_te])[:, 1]
                rows.append(dict(dataset=ds, model=m, label=lab, predictor=name, auroc_cv=roc_auc_score(y, preds[name]), auprc_cv=average_precision_score(y, preds[name]),
                                 auroc_official_test=roc_auc_score(y[is_te], p_off), n=len(y), error_rate=y.mean()))
            for name, cols in [("lookup qtype x answer", ["qt", "k"]), ("lookup question x answer", ["q", "k"]), ("lookup answer only", ["k"]), ("lookup question only", ["q"])]:
                preds[name] = lookup_pred(meta, y, folds, cols)
                rows.append(dict(dataset=ds, model=m, label=lab, predictor=name, auroc_cv=roc_auc_score(y, preds[name]), auprc_cv=average_precision_score(y, preds[name]), auroc_official_test=np.nan, n=len(y), error_rate=y.mean()))
            for name in ["conf", "agree", "agree+conf", "q+y0", "img+q+y0", "img+q+y0+agree+conf", "lookup question x answer"]:
                for cov in COVS:
                    n = int(cov * len(y)); keep = np.argsort(preds[name])[:n]
                    cov_rows.append(dict(dataset=ds, model=m, label=lab, predictor=name, coverage=cov, selective_accuracy=1 - y[keep].mean(), base_accuracy=1 - y.mean()))
    # cross-VLM transfer (q+y0 embeddings, official train+val -> other model's test)
    for ds in ["vqa_rad", "slake"]:
        for a in ["medgemma", "qwen"]:
            for b in ["medgemma", "qwen"]:
                Za, ya, sa = cache[(ds, a)]; Zb, yb_, sb = cache[(ds, b)]
                Fa = np.concatenate([Za["z_q"], Za["z_y0"]], 1); Fb = np.concatenate([Zb["z_q"], Zb["z_y0"]], 1)
                p = clf().fit(Fa[sa != "test"], ya[sa != "test"]).predict_proba(Fb[sb == "test"])[:, 1]
                tr_rows.append(dict(dataset=ds, train_model=a, test_model=b, auroc=roc_auc_score(yb_[sb == "test"], p)))
        t = M[(M.dataset == ds) & (M.split == "test")].pivot(index="sample_id", columns="model", values="keep_score") < 1
        tr_rows.append(dict(dataset=ds, train_model="error overlap", test_model="", auroc=np.nan, p_both_wrong=(t.medgemma & t.qwen).mean(), p_mg_wrong=t.medgemma.mean(), p_q_wrong=t.qwen.mean(),
                            p_q_wrong_given_mg_wrong=(t.medgemma & t.qwen).sum() / t.medgemma.sum(), phi=np.corrcoef(t.medgemma, t.qwen)[0, 1]))
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "error_predictability.csv", index=False)
    C = pd.DataFrame(cov_rows); C.to_csv(RESULTS / "error_predictability_coverage.csv", index=False)
    T = pd.DataFrame(tr_rows); T.to_csv(RESULTS / "error_predictability_transfer.csv", index=False)
    # table
    out = ["## Error predictability of the initial VLM answer (label = answer wrong)\n", "### AUROC, 5-fold GroupKFold by image over all QA (official train+val→test in parentheses)\n"]
    preds_order = ["conf", "agree", "agree+conf", "img", "q", "y0", "q+y0", "img+q", "img+q+y0", "clip img+q+y0", "q+y0+agree+conf", "img+q+y0+agree+conf", "lookup answer only", "lookup question only", "lookup qtype x answer", "lookup question x answer"]
    for lab in ["A", "B"]:
        out.append(f"\n**Metric {lab} labels**\n"); out.append("| Predictor | VR-MG | VR-Q | SL-MG | SL-Q |"); out.append("|---|---:|---:|---:|---:|")
        for p in preds_order:
            cells = []
            for d, m in SETTINGS:
                g = R[(R.dataset == d) & (R.model == m) & (R.label == lab) & (R.predictor == p)].iloc[0]
                cells.append(f"{g.auroc_cv:.3f}" + (f" ({g.auroc_official_test:.3f})" if not np.isnan(g.auroc_official_test) else ""))
            out.append(f"| {p} | " + " | ".join(cells) + " |")
    out.append("\n### Selective answering (Metric A): accuracy of the retained answers at fixed coverage (5-fold CV predictions)\n")
    out.append("| Setting | base acc | predictor | @90% | @80% | @70% | @50% |"); out.append("|---|---:|---|---:|---:|---:|---:|")
    for d, m in SETTINGS:
        for p in ["conf", "agree+conf", "q+y0", "img+q+y0+agree+conf", "lookup question x answer"]:
            g = C[(C.dataset == d) & (C.model == m) & (C.label == "A") & (C.predictor == p)].set_index("coverage")
            out.append(f"| {d}/{m} | {100*g.base_accuracy.iloc[0]:.1f} | {p} | " + " | ".join(f"{100*g.loc[c, 'selective_accuracy']:.1f}" for c in COVS) + " |")
    out.append("\n### Cross-VLM transfer (q+y0 embeddings; train on one VLM's errors, test on the other's test errors) and error overlap\n")
    out.append("| Dataset | train → test | AUROC |"); out.append("|---|---|---:|")
    for _, r in T[T.train_model != "error overlap"].iterrows():
        out.append(f"| {r.dataset} | {r.train_model} → {r.test_model} | {r.auroc:.3f} |")
    for _, r in T[T.train_model == "error overlap"].iterrows():
        out.append(f"\n{r.dataset}: P(both wrong) = {r.p_both_wrong:.2f}, P(MG wrong) = {r.p_mg_wrong:.2f}, P(Q wrong) = {r.p_q_wrong:.2f}, P(Q wrong | MG wrong) = {r.p_q_wrong_given_mg_wrong:.2f}, φ = {r.phi:.2f}")
    (ROOT / "tables" / "error_predictability.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
