"""Pre-registered tests (evaluation/preregistration_errors.md).
H1: q+y0 error predictor vs sampling self-consistency (results/sampling/*.jsonl; signal = agree_greedy, n_distinct; also mean sample score).
H2: extra VLMs (results/raw/{ds}_{model}_*_keep.jsonl for qwen3b, qwen2vl2b, gemma3): error overlap phi with medgemma/qwen on test,
    cross-VLM transfer of the q+y0 predictor (needs BiomedCLIP features of the new model's KEEP answers -> computed here with open_clip if available,
    else falls back to TF-IDF text predictor which needs no features).
H3: image ablation (already in results/error_predictability.csv) summarised.
Outputs: results/error_h1_sampling.csv, results/error_h2_models.csv, tables/error_hypotheses.md"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonicalize, closed_pred_canonical
from common import ACTIONS, DATA, RESULTS, ROOT, read_jsonl
from consensus_stage1 import load
SETTINGS = [(d, m) for d in ["vqa_rad", "slake"] for m in ["medgemma", "qwen"]]
EXTRA = ["qwen3b", "qwen2vl2b", "gemma3"]
def key(a, q):
    yn = closed_pred_canonical(a, q); return yn if yn in ("yes", "no") else canonicalize(a)
M = load()
out = ["## Pre-registered hypothesis tests\n"]

# ---------------- H1
out.append("### H1 — q+y0 prior vs sampling self-consistency (test split; sampling: 5 samples, T=0.7, top-p 0.95)\n")
out.append("| Setting | n | q+y0 embeddings (train+val→test) | TF-IDF q+a (train+val→test) | sampling: agree-with-greedy | sampling: n distinct | log-prob | intervention agreement | q+y0 + sampling |")
out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
h1 = []
for d, m in SETTINGS:
    p = RESULTS / "sampling" / f"{d}_{m}_test.jsonl"
    if not p.exists(): out.append(f"| {d}/{m} | — | sampling not available yet | | | | | | | |"); continue
    S = {r["sample_id"]: r for r in read_jsonl(p)}
    te = M[(M.dataset == d) & (M.model == m) & (M.split == "test")]; te = te[te.sample_id.isin(S)].reset_index(drop=True)
    y = (te.keep_score.values < 1).astype(int)
    agree = np.array([S[s]["agree_greedy"] for s in te.sample_id]); nd = np.array([S[s]["n_distinct"] for s in te.sample_id]); ms = np.array([np.mean(S[s]["scores"]) for s in te.sample_id])
    conf = te.base_conf.fillna(te.base_conf.min()).values
    K = te[[f"{a}_key" for a in ACTIONS]].values; iagree = np.array([sum(K[i, j] == K[i, 0] for j in range(1, 4)) for i in range(len(te))])
    # q+y0 predictor, official train+val -> test
    from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
    ztr = [np.load(RESULTS / "features" / f"{d}_{m}_{s}.npz") for s in ["train", "validation"]]; zte = np.load(RESULTS / "features" / f"{d}_{m}_test.npz")
    Xtr = np.concatenate([np.concatenate([z["z_q"], z["z_y0"]], 1) for z in ztr]); ids_tr = sum([list(z["sample_id"]) for z in ztr], [])
    Mm = M[M.model == m].set_index("sample_id")
    ytr = (Mm.loc[ids_tr].keep_score.values < 1).astype(int)
    Xte_all = np.concatenate([zte["z_q"], zte["z_y0"]], 1); pos = {s: i for i, s in enumerate(zte["sample_id"])}; Xte = Xte_all[[pos[s] for s in te.sample_id]]
    p_emb = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    trtxt = (Mm.loc[ids_tr].question.str.lower() + " ||| " + Mm.loc[ids_tr].keep_key).values
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(trtxt)
    p_tf = LogisticRegression(C=1.0, max_iter=3000).fit(vec.transform(trtxt), ytr).predict_proba(vec.transform((te.question.str.lower() + " ||| " + te.keep_key).values))[:, 1]
    comb = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    # combination fitted on validation (sampling val available for the two main models)
    pv = RESULTS / "sampling" / f"{d}_{m}_validation.jsonl"; a_comb = np.nan
    if pv.exists():
        Sv = {r["sample_id"]: r for r in read_jsonl(pv)}; va = M[(M.dataset == d) & (M.model == m) & (M.split == "validation")]; va = va[va.sample_id.isin(Sv)]
        zva = np.load(RESULTS / "features" / f"{d}_{m}_validation.npz"); posv = {s: i for i, s in enumerate(zva["sample_id"])}
        Xva = np.concatenate([zva["z_q"], zva["z_y0"]], 1)[[posv[s] for s in va.sample_id]]
        ztr0 = np.load(RESULTS / "features" / f"{d}_{m}_train.npz"); Xtr0 = np.concatenate([ztr0["z_q"], ztr0["z_y0"]], 1); ytr0 = (Mm.loc[list(ztr0["sample_id"])].keep_score.values < 1).astype(int)
        base_clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xtr0, ytr0)
        fv = np.stack([base_clf.predict_proba(Xva)[:, 1], [Sv[s]["agree_greedy"] for s in va.sample_id], [Sv[s]["n_distinct"] for s in va.sample_id]], 1); yv = (va.keep_score.values < 1).astype(int)
        ft = np.stack([base_clf.predict_proba(Xte)[:, 1], agree, nd], 1)
        a_comb = roc_auc_score(y, comb.fit(fv, yv).predict_proba(ft)[:, 1])
    r = dict(dataset=d, model=m, n=len(te), auroc_emb=roc_auc_score(y, p_emb), auroc_tfidf=roc_auc_score(y, p_tf), auroc_samp_agree=roc_auc_score(y, -agree), auroc_samp_distinct=roc_auc_score(y, nd),
             auroc_conf=roc_auc_score(y, -conf), auroc_interv_agree=roc_auc_score(y, -iagree), auroc_comb=a_comb)
    h1.append(r)
    out.append(f"| {d}/{m} | {len(te)} | **{r['auroc_emb']:.3f}** | {r['auroc_tfidf']:.3f} | {r['auroc_samp_agree']:.3f} | {r['auroc_samp_distinct']:.3f} | {r['auroc_conf']:.3f} | {r['auroc_interv_agree']:.3f} | {r['auroc_comb']:.3f} |")
pd.DataFrame(h1).to_csv(RESULTS / "error_h1_sampling.csv", index=False)

# ---------------- H2
out.append("\n### H2 — additional VLMs: accuracy, error overlap with MedGemma-4B / Qwen2.5-VL-7B (test), and transfer of a TF-IDF (question+answer) error predictor trained on another VLM's train+val errors\n")
out.append("| Dataset | VLM | test acc (Metric A) | φ with MedGemma | φ with Qwen-7B | P(wrong \\| MedGemma wrong) | TF-IDF self (train+val→test) | trained on MedGemma → this VLM | trained on Qwen-7B → this VLM | oracle over 4 actions (test) | best fixed |")
out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
h2 = []
samples = {d: {s: {r["sample_id"]: r for r in read_jsonl(DATA / d / f"{s}.jsonl")} for s in ["train", "validation", "test"]} for d in ["vqa_rad", "slake"]}
def keep_frame(d, m, split):
    p = RESULTS / "raw" / f"{d}_{m}_{split}_keep.jsonl"
    if not p.exists(): return None
    rows = read_jsonl(p); return pd.DataFrame(dict(sample_id=[r["sample_id"] for r in rows], score=[r["score"] for r in rows], answer=[r["answer_raw"] for r in rows],
                                                    question=[samples[d][split][r["sample_id"]]["question"] for r in rows]))
for d in ["vqa_rad", "slake"]:
    ref = {m: keep_frame(d, m, "test").set_index("sample_id") for m in ["medgemma", "qwen"]}
    tf_train = {}
    for m in ["medgemma", "qwen"] + EXTRA:
        fr = [keep_frame(d, m, s) for s in ["train", "validation"]]
        if any(f is None for f in fr): continue
        f = pd.concat(fr); tf_train[m] = ((f.question.str.lower() + " ||| " + [key(a, q) for a, q in zip(f.answer, f.question)]).values, (f.score.values < 1).astype(int))
    for m in EXTRA + ["medgemma", "qwen"]:
        te = keep_frame(d, m, "test")
        if te is None or m not in tf_train: continue
        te = te.set_index("sample_id"); ids = te.index.intersection(ref["medgemma"].index)
        e = (te.loc[ids].score.values < 1); em = (ref["medgemma"].loc[ids].score.values < 1); eq = (ref["qwen"].loc[ids].score.values < 1)
        phi = lambda a, b: np.corrcoef(a, b)[0, 1]
        txt_te = (te.loc[ids].question.str.lower() + " ||| " + [key(a, q) for a, q in zip(te.loc[ids].answer, te.loc[ids].question)]).values
        def tf_auc(src):
            X, yv = tf_train[src]; vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(X)
            return roc_auc_score(e.astype(int), LogisticRegression(C=1.0, max_iter=3000).fit(vec.transform(X), yv).predict_proba(vec.transform(txt_te))[:, 1])
        # oracle over the 4 actions if interventions exist
        orc = bf = np.nan
        acts = {a: RESULTS / "raw" / f"{d}_{m}_test_{a}.jsonl" for a in ACTIONS}
        if all(p.exists() for p in acts.values()):
            Sc = np.stack([[r["score"] for r in read_jsonl(acts[a])] for a in ACTIONS], 1); orc = Sc.max(1).mean(); bf = Sc.mean(0).max()
        r = dict(dataset=d, model=m, acc=1 - e.mean(), phi_medgemma=phi(e, em) if m != "medgemma" else 1.0, phi_qwen=phi(e, eq) if m != "qwen" else 1.0, p_wrong_given_mg_wrong=(e & em).sum() / em.sum(),
                 tf_self=tf_auc(m), tf_from_medgemma=tf_auc("medgemma"), tf_from_qwen=tf_auc("qwen"), oracle4=orc, best_fixed=bf)
        h2.append(r)
        out.append(f"| {d} | {m} | {100*r['acc']:.1f} | {r['phi_medgemma']:.2f} | {r['phi_qwen']:.2f} | {100*r['p_wrong_given_mg_wrong']:.0f}% | {r['tf_self']:.3f} | {r['tf_from_medgemma']:.3f} | {r['tf_from_qwen']:.3f} | {100*orc:.1f} | {100*bf:.1f} |")
pd.DataFrame(h2).to_csv(RESULTS / "error_h2_models.csv", index=False)
# ---------------- H3
E = pd.read_csv(RESULTS / "error_predictability.csv"); E = E[E.label == "A"]
out.append("\n### H3 — image contribution (AUROC, image-grouped 5-fold CV, Metric A)\n"); out.append("| Setting | q+y0 | img+q+y0 | Δ | agree+conf | img+q+y0+agree+conf | Δ vs q+y0+agree+conf |"); out.append("|---|---:|---:|---:|---:|---:|---:|")
for d, m in SETTINGS:
    g = E[(E.dataset == d) & (E.model == m)].set_index("predictor").auroc_cv
    out.append(f"| {d}/{m} | {g['q+y0']:.3f} | {g['img+q+y0']:.3f} | {g['img+q+y0']-g['q+y0']:+.3f} | {g['agree+conf']:.3f} | {g['img+q+y0+agree+conf']:.3f} | {g['img+q+y0+agree+conf']-g['q+y0+agree+conf']:+.3f} |")
(ROOT / "tables" / "error_hypotheses.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
