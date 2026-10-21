"""Stage A analyses. A1 image-counterfactual sensitivity; A3 hidden-state probe; A4 item-vs-model variance decomposition.
Outputs: results/stageA_counterfactual.csv, results/stageA_probe.csv, results/stageA_variance.csv, tables/stageA.md"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonicalize, closed_pred_canonical
from common import DATA, RESULTS, ROOT, read_jsonl, score
MODELS = ["medgemma", "gemma3", "qwen", "qwen3b", "qwen2vl2b", "lingshu7b", "internvl3_8b", "qwen3vl8b"]
MAIN = list(MODELS)
MODELS = MODELS + ["qwen25vl32b", "medgemma27b"]
NAMES = {"qwen25vl32b": "Qwen2.5-VL-32B", "medgemma27b": "MedGemma-27B", "medgemma": "MedGemma-4B", "gemma3": "Gemma-3-4B", "qwen": "Qwen2.5-VL-7B", "qwen3b": "Qwen2.5-VL-3B", "qwen2vl2b": "Qwen2-VL-2B", "lingshu7b": "Lingshu-7B", "internvl3_8b": "InternVL3-8B", "qwen3vl8b": "Qwen3-VL-8B"}
DS = ["vqa_rad", "slake", "pathvqa", "pmcvqa", "omnimedvqa"]
def key(a, q):
    yn = closed_pred_canonical(a, q); return yn if yn in ("yes", "no") else canonicalize(a)
out = ["## Stage A\n"]
samples = {d: {s: {r["sample_id"]: r for r in read_jsonl(DATA / d / f"{s}.jsonl")} for s in ["train", "validation", "test"]} for d in DS}
def keep(d, m, s):
    p = RESULTS / "raw" / f"{d}_{m}_{s}_keep.jsonl"; return {r["sample_id"]: r for r in read_jsonl(p)} if p.exists() else None
def qy_features(d, m, s):
    """question+answer text features via BiomedCLIP if available for this model, else TF-IDF handled by caller."""
    p = RESULTS / "features" / f"{d}_{m}_{s}.npz"; return np.load(p) if p.exists() else None

# ---------------- A1
out.append("### A1 — image-counterfactual sensitivity (test)\n")
out.append("| Dataset | VLM | acc with image | acc no image | acc blank | acc swapped | unchanged under swap | unchanged under blank | unchanged under no-image | acc (image) \\| insensitive to swap | acc (image) \\| sensitive | error-detection AUROC of 'changed under swap' | swap-insensitive rate among predicted errors vs predicted correct (top/bottom tercile of q+a prior) |")
out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
a1 = []
for d in DS:
    for m in MODELS:
        K = keep(d, m, "test"); C = {c: (RESULTS / "counterfactual" / f"{d}_{m}_test_{c}.jsonl") for c in ["noimage", "blank", "swap"]}
        if K is None or not all(p.exists() for p in C.values()): out.append(f"| {d} | {NAMES[m]} | pending |"); continue
        C = {c: {r["sample_id"]: r for r in read_jsonl(p)} for c, p in C.items()}
        ids = [i for i in K if all(i in C[c] for c in C)]; S = samples[d]["test"]
        kk = np.array([key(K[i]["answer_raw"], S[i]["question"]) for i in ids]); acc = np.array([K[i]["score"] for i in ids])
        same = {c: np.array([key(C[c][i]["answer_raw"], S[i]["question"]) == k for i, k in zip(ids, kk)]) for c in C}
        accc = {c: np.mean([C[c][i]["score"] for i in ids]) for c in C}
        err = (acc < 1).astype(int); au = roc_auc_score(err, (~same["swap"]).astype(int)) if 0 < err.mean() < 1 else np.nan
        # q+a prior (TF-IDF, train+val -> test) to relate predicted error to insensitivity
        from sklearn.feature_extraction.text import TfidfVectorizer
        tr = []; ytr = []
        for s in ["train", "validation"]:
            Ks = keep(d, m, s)
            if Ks is None: continue
            for i, r in Ks.items(): tr.append(S_q := samples[d][s][i]["question"].lower() + " ||| " + key(r["answer_raw"], samples[d][s][i]["question"])); ytr.append(int(r["score"] < 1))
        terc = ""
        if len(tr) > 100:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(tr); clf = LogisticRegression(C=1.0, max_iter=3000).fit(vec.transform(tr), ytr)
            pe = clf.predict_proba(vec.transform([S[i]["question"].lower() + " ||| " + k for i, k in zip(ids, kk)]))[:, 1]
            hi, lo = pe >= np.quantile(pe, 2 / 3), pe <= np.quantile(pe, 1 / 3)
            terc = f"{100*same['swap'][hi].mean():.0f}% vs {100*same['swap'][lo].mean():.0f}% (OR {((same['swap'][hi].mean()/(1-same['swap'][hi].mean()+1e-9))/(same['swap'][lo].mean()/(1-same['swap'][lo].mean()+1e-9))):.2f})"
        r = dict(dataset=d, model=m, n=len(ids), acc=acc.mean(), acc_noimage=accc["noimage"], acc_blank=accc["blank"], acc_swap=accc["swap"], same_swap=same["swap"].mean(), same_blank=same["blank"].mean(), same_noimage=same["noimage"].mean(),
                 acc_if_insensitive=acc[same["swap"]].mean(), acc_if_sensitive=acc[~same["swap"]].mean() if (~same["swap"]).any() else np.nan, auroc_changed=au, terciles=terc)
        a1.append(r)
        out.append(f"| {d} | {NAMES[m]} | {100*r['acc']:.1f} | {100*r['acc_noimage']:.1f} | {100*r['acc_blank']:.1f} | {100*r['acc_swap']:.1f} | {100*r['same_swap']:.1f} | {100*r['same_blank']:.1f} | {100*r['same_noimage']:.1f} | {100*r['acc_if_insensitive']:.1f} | {100*r['acc_if_sensitive']:.1f} | {au:.3f} | {terc} |")
pd.DataFrame(a1).to_csv(RESULTS / "stageA_counterfactual.csv", index=False)

# ---------------- A3 hidden probe
out.append("\n### A3 — hidden-state probe vs question+answer prior (AUROC; official train+val→test and image-grouped 5-fold CV)\n")
out.append("| Dataset | VLM | d | probe (train+val→test) | probe (CV) | TF-IDF q+a (train+val→test) | q+a BiomedCLIP (train+val→test) | probe + q+a (CV) |"); out.append("|---|---|---:|---:|---:|---:|---:|---:|")
a3 = []
for d in DS:
    for m in MODELS:
        H = {s: (RESULTS / "hidden" / f"{d}_{m}_{s}.npz") for s in ["train", "validation", "test"]}
        if not all(p.exists() for p in H.values()) or keep(d, m, "test") is None: out.append(f"| {d} | {NAMES[m]} | pending |"); continue
        Z = {s: np.load(H[s]) for s in H}; Ks = {s: keep(d, m, s) for s in H}
        X = {s: Z[s]["h"].astype(np.float32) for s in H}; y = {s: np.array([int(Ks[s][i]["score"] < 1) for i in Z[s]["sample_id"]]) for s in H}
        grp = {s: np.array([samples[d][s][i]["image_path"] for i in Z[s]["sample_id"]]) for s in H}
        txt = {s: [samples[d][s][i]["question"].lower() + " ||| " + key(Ks[s][i]["answer_raw"], samples[d][s][i]["question"]) for i in Z[s]["sample_id"]] for s in H}
        Xtr = np.concatenate([X["train"], X["validation"]]); ytr = np.concatenate([y["train"], y["validation"]])
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(Xtr, ytr); p_probe = probe.predict_proba(X["test"])[:, 1]
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(txt["train"] + txt["validation"]); tf = LogisticRegression(C=1.0, max_iter=3000).fit(vec.transform(txt["train"] + txt["validation"]), ytr)
        p_tf = tf.predict_proba(vec.transform(txt["test"]))[:, 1]
        a_emb = np.nan
        F = {s: qy_features(d, m, s) for s in H}
        if all(v is not None for v in F.values()):
            pos = {s: {i: j for j, i in enumerate(F[s]["sample_id"])} for s in H}
            E = {s: np.concatenate([F[s]["z_q"], F[s]["z_y0"]], 1)[[pos[s][i] for i in Z[s]["sample_id"]]] for s in H}
            a_emb = roc_auc_score(y["test"], make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(np.concatenate([E["train"], E["validation"]]), ytr).predict_proba(E["test"])[:, 1])
        # CV over all splits
        Xa = np.concatenate([X[s] for s in H]); ya = np.concatenate([y[s] for s in H]); ga = np.concatenate([grp[s] for s in H]); ta = sum([txt[s] for s in H], [])
        folds = list(GroupKFold(5).split(Xa, ya, ga)); pc = np.zeros(len(ya)); pcomb = np.zeros(len(ya))
        for tr, ho in folds:
            pr = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(Xa[tr], ya[tr]); pc[ho] = pr.predict_proba(Xa[ho])[:, 1]
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([ta[i] for i in tr]); t = LogisticRegression(C=1.0, max_iter=3000).fit(v.transform([ta[i] for i in tr]), ya[tr])
            # stack: fit combiner on the training fold's own predictions (in-sample, simple)
            ptr = np.stack([pr.predict_proba(Xa[tr])[:, 1], t.predict_proba(v.transform([ta[i] for i in tr]))[:, 1]], 1); pho = np.stack([pc[ho], t.predict_proba(v.transform([ta[i] for i in ho]))[:, 1]], 1)
            pcomb[ho] = LogisticRegression().fit(ptr, ya[tr]).predict_proba(pho)[:, 1]
        r = dict(dataset=d, model=m, dim=Xa.shape[1], probe_test=roc_auc_score(y["test"], p_probe), probe_cv=roc_auc_score(ya, pc), tfidf_test=roc_auc_score(y["test"], p_tf), qa_emb_test=a_emb, comb_cv=roc_auc_score(ya, pcomb))
        a3.append(r); out.append(f"| {d} | {NAMES[m]} | {r['dim']} | {r['probe_test']:.3f} | {r['probe_cv']:.3f} | {r['tfidf_test']:.3f} | {r['qa_emb_test']:.3f} | {r['comb_cv']:.3f} |")
pd.DataFrame(a3).to_csv(RESULTS / "stageA_probe.csv", index=False)

# ---------------- A4 variance decomposition (test items x models)
out.append("\n### A4 — item vs model effects on correctness (test; logistic fixed effects, McFadden pseudo-R²; ICC from a random-intercept logit approximation)\n")
out.append("| Dataset | # models | # items | R² model effects only | R² item effects only | R² both | item-level ICC (latent scale) |"); out.append("|---|---:|---:|---:|---:|---:|---:|")
a4 = []
for d in DS:
    ks = {m: keep(d, m, "test") for m in MAIN}; ms = [m for m in MAIN if ks[m] is not None]
    ids = sorted(set.intersection(*[set(ks[m]) for m in ms])); Y = np.array([[int(ks[m][i]["score"] >= 1) for m in ms] for i in ids])
    long = pd.DataFrame(dict(item=np.repeat(np.arange(len(ids)), len(ms)), model=np.tile(np.arange(len(ms)), len(ids)), y=Y.reshape(-1)))
    def ll(pred): return np.sum(long.y * np.log(np.clip(pred, 1e-9, 1)) + (1 - long.y) * np.log(np.clip(1 - pred, 1e-9, 1)))
    ll0 = ll(np.full(len(long), long.y.mean()))
    pm = long.groupby("model").y.transform("mean").values; pi = long.groupby("item").y.transform("mean").values
    # both: additive logit fit via simple alternating (item + model) logistic regression with one-hot
    from scipy.sparse import csr_matrix, hstack
    Xi = csr_matrix((np.ones(len(long)), (np.arange(len(long)), long.item.values)), shape=(len(long), len(ids))); Xm = csr_matrix((np.ones(len(long)), (np.arange(len(long)), long.model.values)), shape=(len(long), len(ms)))
    pb = LogisticRegression(C=10.0, max_iter=3000).fit(hstack([Xi, Xm]), long.y).predict_proba(hstack([Xi, Xm]))[:, 1]
    # item ICC: variance of item logit means (shrunk) vs pi^2/3
    pim = np.clip(Y.mean(1), 0.02, 0.98); var_item = np.var(np.log(pim / (1 - pim))); icc = var_item / (var_item + np.pi ** 2 / 3)
    r = dict(dataset=d, n_models=len(ms), n_items=len(ids), r2_model=1 - ll(pm) / ll0, r2_item=1 - ll(np.clip(pi, 1e-3, 1 - 1e-3)) / ll0, r2_both=1 - ll(pb) / ll0, icc_item=icc)
    a4.append(r); out.append(f"| {d} | {len(ms)} | {len(ids)} | {r['r2_model']:.3f} | {r['r2_item']:.3f} | {r['r2_both']:.3f} | {icc:.2f} |")
pd.DataFrame(a4).to_csv(RESULTS / "stageA_variance.csv", index=False)
(ROOT / "tables" / "stageA.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
