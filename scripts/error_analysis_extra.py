"""Offline analyses for the error-predictability paper: (1) what is systematic — (question, answer) clusters with their error
rate (Metric B labels to exclude lexical artefacts); (2) per-question-type predictability; (3) cross-dataset transfer;
(4) TF-IDF text-only predictor (no embeddings at all); (5) risk-coverage curves with bootstrap CIs + AURC; (6) share of
errors that sit on 'always-wrong' (question, answer) pairs. Outputs results/error_extra_*.csv, tables/error_extra.md"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
sys.path.insert(0, str(Path(__file__).resolve().parent))
from consensus_stage1 import load
from common import RESULTS, ROOT
SETTINGS = [(d, m) for d in ["vqa_rad", "slake"] for m in ["medgemma", "qwen"]]
rng = np.random.RandomState(0)
M = load(); M["keep_score_b"] = pd.to_numeric(M.keep_score_b)
out = ["## Extra analyses — error predictability\n"]

# (1) systematic error clusters, Metric B, all splits
out.append("### (1) Largest systematic-error clusters (question text × canonical initial answer), Metric B labels, all splits\n")
out.append("| Setting | question | initial answer | n | error rate | ref answers (top) |"); out.append("|---|---|---|---:|---:|---|")
clus = []
for d, m in SETTINGS:
    g = M[(M.dataset == d) & (M.model == m)].copy(); g["err"] = (g.keep_score_b < 1).astype(int); g["q"] = g.question.str.lower().str.strip()
    t = g.groupby(["q", "keep_key"]).agg(n=("err", "size"), err=("err", "mean"), refs=("reference_answer", lambda x: ", ".join(f"{k}×{v}" for k, v in x.value_counts().head(3).items()))).reset_index()
    t["errors"] = t.n * t.err; t = t.sort_values("errors", ascending=False)
    tot = g.err.sum()
    top = t.head(8)
    for _, r in top.iterrows():
        out.append(f"| {d}/{m} | {r.q[:60]} | {r.keep_key} | {r.n} | {100*r.err:.0f}% | {r.refs} |")
    always = t[(t.err >= 0.9) & (t.n >= 3)]
    clus.append(dict(dataset=d, model=m, total_errors=int(tot), errors_in_top10_clusters=int(t.head(10).errors.sum()), errors_on_always_wrong_pairs=int(always.errors.sum()), n_always_wrong_pairs=len(always),
                     errors_on_pairs_seen_ge3=int(t[t.n >= 3].errors.sum())))
C = pd.DataFrame(clus); C.to_csv(RESULTS / "error_extra_clusters.csv", index=False)
out.append("\n| Setting | total errors (B) | in top-10 (q,a) clusters | on 'always-wrong' pairs (err≥90%, n≥3) | # such pairs | on pairs seen ≥3 times |"); out.append("|---|---:|---:|---:|---:|---:|")
for _, r in C.iterrows():
    out.append(f"| {r.dataset}/{r.model} | {r.total_errors} | {r.errors_in_top10_clusters} ({100*r.errors_in_top10_clusters/r.total_errors:.0f}%) | {r.errors_on_always_wrong_pairs} ({100*r.errors_on_always_wrong_pairs/r.total_errors:.0f}%) | {r.n_always_wrong_pairs} | {r.errors_on_pairs_seen_ge3} ({100*r.errors_on_pairs_seen_ge3/r.total_errors:.0f}%) |")

# helpers
def feats(d, m):
    Z = {k: [] for k in ["z_img", "z_q", "z_y0"]}; y = []; grp = []; txt = []; qt = []; at = []; sp = []
    for s in ["train", "validation", "test"]:
        z = np.load(RESULTS / "features" / f"{d}_{m}_{s}.npz"); ids = list(z["sample_id"]); te = M[(M.dataset == d) & (M.model == m) & (M.split == s)].set_index("sample_id").loc[ids]
        for k in Z: Z[k].append(z[k])
        y.append((te.keep_score.values < 1).astype(int)); grp.append(te.image_path.values); txt.append((te.question.str.lower() + " ||| " + te.keep_key).values); qt.append(te.question_type.values); at.append(te.answer_type.values); sp.append(np.full(len(te), s))
    return {k: np.concatenate(v) for k, v in Z.items()}, np.concatenate(y), np.concatenate(grp), np.concatenate(txt), np.concatenate(qt), np.concatenate(at), np.concatenate(sp)
def cv(F, y, folds, sparse=False):
    p = np.zeros(len(y))
    for tr, ho in folds:
        c = LogisticRegression(C=1.0, max_iter=3000) if sparse else make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        p[ho] = c.fit(F[tr], y[tr]).predict_proba(F[ho])[:, 1]
    return p

# (2)+(4)+(5)
out.append("\n### (2) Per-question-type / answer-type error-detection AUROC (q+y0 embeddings, image-grouped 5-fold CV predictions) and (4) TF-IDF text-only predictor\n")
out.append("| Setting | overall q+y0 | TF-IDF(question+answer) | TF-IDF question only | closed Qs | open Qs | per question type (n≥30) |"); out.append("|---|---:|---:|---:|---:|---:|---|")
rc_rows = []
for d, m in SETTINGS:
    Z, y, grp, txt, qt, at, sp = feats(d, m); folds = list(GroupKFold(5).split(Z["z_q"], y, grp))
    F = np.concatenate([Z["z_q"], Z["z_y0"]], 1); p = cv(F, y, folds)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2); X = vec.fit_transform(txt); p_tf = cv(X, y, folds, sparse=True)
    vq = TfidfVectorizer(ngram_range=(1, 2), min_df=2); Xq = vq.fit_transform([t.split(" ||| ")[0] for t in txt]); p_tq = cv(Xq, y, folds, sparse=True)
    per = []
    for t in sorted(set(qt)):
        mk = qt == t
        if mk.sum() >= 30 and 0 < y[mk].mean() < 1: per.append(f"{t} {roc_auc_score(y[mk], p[mk]):.2f} (n={mk.sum()}, err {100*y[mk].mean():.0f}%)")
    auc = lambda mk: f"{roc_auc_score(y[mk], p[mk]):.3f}" if 0 < y[mk].mean() < 1 else "n/a"
    out.append(f"| {d}/{m} | {roc_auc_score(y, p):.3f} | {roc_auc_score(y, p_tf):.3f} | {roc_auc_score(y, p_tq):.3f} | {auc(at == 'CLOSED')} | {auc(at == 'OPEN')} | {'; '.join(per)} |")
    # (5) risk-coverage with bootstrap on the test split only (CV predictions restricted to test)
    te = sp == "test"; pt, yt = p[te], y[te]; order = np.argsort(pt)
    for cov in [0.5, 0.7, 0.8, 0.9, 1.0]:
        n = int(cov * len(yt)); keep = order[:n]; accs = []
        idx = rng.randint(0, len(yt), (1000, len(yt)))
        for b in idx:
            pb, yb = pt[b], yt[b]; kb = np.argsort(pb)[:int(cov * len(yb))]; accs.append(1 - yb[kb].mean())
        rc_rows.append(dict(dataset=d, model=m, coverage=cov, selective_acc=1 - yt[keep].mean(), ci_low=np.percentile(accs, 2.5), ci_high=np.percentile(accs, 97.5)))
    # AURC
    risks = [yt[order[:n]].mean() for n in range(1, len(yt) + 1)]; rc_rows.append(dict(dataset=d, model=m, coverage="AURC", selective_acc=np.mean(risks), ci_low=np.nan, ci_high=np.nan))
RC = pd.DataFrame(rc_rows); RC.to_csv(RESULTS / "error_extra_risk_coverage.csv", index=False)
out.append("\n### (5) Risk–coverage on the test split (q+y0 predictor, CV predictions; 95% bootstrap CI)\n")
out.append("| Setting | acc@50% | acc@70% | acc@80% | acc@90% | acc@100% | AURC (mean risk) |"); out.append("|---|---:|---:|---:|---:|---:|---:|")
for d, m in SETTINGS:
    g = RC[(RC.dataset == d) & (RC.model == m)].set_index("coverage")
    cells = [f"{100*g.loc[c,'selective_acc']:.1f} [{100*g.loc[c,'ci_low']:.1f}, {100*g.loc[c,'ci_high']:.1f}]" for c in [0.5, 0.7, 0.8, 0.9, 1.0]]
    out.append(f"| {d}/{m} | " + " | ".join(cells) + f" | {g.loc['AURC','selective_acc']:.3f} |")

# (3) cross-dataset transfer (train on all splits of dataset A/model, test on dataset B test of same model)
out.append("\n### (3) Cross-dataset transfer of the q+y0 predictor (same VLM; train on all QA of one dataset, test on the other's test split)\n")
out.append("| VLM | SLAKE → VQA-RAD test | VQA-RAD → SLAKE test |"); out.append("|---|---:|---:|")
for m in ["medgemma", "qwen"]:
    r = {}
    for a, b in [("slake", "vqa_rad"), ("vqa_rad", "slake")]:
        Za, ya, _, _, _, _, _ = feats(a, m); Zb, yb, _, _, _, _, spb = feats(b, m)
        Fa = np.concatenate([Za["z_q"], Za["z_y0"]], 1); Fb = np.concatenate([Zb["z_q"], Zb["z_y0"]], 1)[spb == "test"]
        p = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Fa, ya).predict_proba(Fb)[:, 1]
        r[(a, b)] = roc_auc_score(yb[spb == "test"], p)
    out.append(f"| {m} | {r[('slake','vqa_rad')]:.3f} | {r[('vqa_rad','slake')]:.3f} |")
(ROOT / "tables" / "error_extra.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
