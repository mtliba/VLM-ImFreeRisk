"""Control for templated questions: grouped CV where examples sharing the same normalised question text are kept in one fold.
Grouping variants: (a) image only (as in the paper), (b) question only, (c) joint = connected components of the graph
linking examples that share an image OR a normalised question (union-find). Predictors: Q emb, Q+A emb (BiomedCLIP), TF-IDF Q, TF-IDF Q+A.
Output: results/repeated_question_control.csv, tables/repeated_question_control.md"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, normalize_answer
from consensus_stage1 import load
M = load()

def components(img, q):
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, qq in zip(img, q): parent[find(("i", i))] = find(("q", qq))
    roots = [find(("i", i)) for i in img]; ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}; return np.array([ids[r] for r in roots])

rows = []
for d in ["vqa_rad", "slake"]:
    for m in ["medgemma", "qwen"]:
        Z = {s: np.load(RESULTS / "features" / f"{d}_{m}_{s}.npz") for s in ["train", "validation", "test"]}
        ids = sum([list(Z[s]["sample_id"]) for s in Z], []); te = M[(M.dataset == d) & (M.model == m)].set_index("sample_id").loc[ids]
        zq = np.concatenate([Z[s]["z_q"] for s in Z]); zy = np.concatenate([Z[s]["z_y0"] for s in Z])
        y = (te.keep_score.values < 1).astype(int); img = te.image_path.values; q = te.question.map(normalize_answer).values
        txt_qa = [f"{a} ||| {b}" for a, b in zip(q, te.keep_key.values)]
        groups = {"image": pd.factorize(img)[0], "question": pd.factorize(q)[0], "joint": components(img, q)}
        for gname, g in groups.items():
            uniq = np.unique(g, return_counts=True)[1]
            if len(uniq) < 5: rows.append(dict(dataset=d, model=m, grouping=gname, n_groups=len(uniq), largest_group=uniq.max(), note="too few groups")); continue
            folds = list(GroupKFold(5).split(zq, y, g))
            def cv_emb(F):
                p = np.zeros(len(y))
                for tr, ho in folds: p[ho] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(F[tr], y[tr]).predict_proba(F[ho])[:, 1]
                return roc_auc_score(y, p)
            def cv_tf(T):
                p = np.zeros(len(y))
                for tr, ho in folds:
                    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b").fit([T[i] for i in tr]); c = LogisticRegression(C=1.0, max_iter=3000).fit(v.transform([T[i] for i in tr]), y[tr]); p[ho] = c.predict_proba(v.transform([T[i] for i in ho]))[:, 1]
                return roc_auc_score(y, p)
            rows.append(dict(dataset=d, model=m, grouping=gname, n_groups=len(uniq), largest_group=int(uniq.max()), q_emb=cv_emb(zq), qa_emb=cv_emb(np.concatenate([zq, zy], 1)), tfidf_q=cv_tf(list(q)), tfidf_qa=cv_tf(txt_qa)))
            print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(RESULTS / "repeated_question_control.csv", index=False)
out = ["| Dataset | VLM | grouping | # groups | largest group | Q emb | Q+A emb (BiomedCLIP) | TF-IDF Q | TF-IDF Q+A |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
for _, r in R.iterrows():
    out.append(f"| {r.dataset} | {r.model} | {r.grouping} | {r.n_groups} | {r.largest_group} | " + (" | ".join(f"{r[k]:.3f}" for k in ["q_emb", "qa_emb", "tfidf_q", "tfidf_qa"]) if "q_emb" in r and not pd.isna(r.get("q_emb", np.nan)) else r.get("note", "")) + " |")
(ROOT / "tables" / "repeated_question_control.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
