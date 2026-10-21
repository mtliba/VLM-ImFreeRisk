"""Stage B (generalisation): every available (benchmark, VLM) pair -> one row with
  accuracy; error predictability (TF-IDF question+answer, image-grouped 5-fold CV; TF-IDF question-only; hidden-state probe if available);
  zero-training signals (log-prob, sampling consistency, perturbation consistency); image-counterfactual (no image / blank / swap accuracy,
  swap-insensitive rate, accuracy given (in)sensitive); interventions (best fixed, oracle, gap, harm); selective accuracy at 70/50 % coverage.
Plus per-benchmark cross-VLM error overlap (phi, all-wrong fraction) and cross-VLM transfer of the TF-IDF predictor.
Outputs: results/stageB_matrix.csv, results/stageB_overlap.csv, results/stageB_transfer.csv, tables/stageB.md"""
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
from canonical import canonicalize, closed_pred_canonical
from common import ACTIONS, DATA, RESULTS, ROOT, read_jsonl, is_mcq, mcq_pred, normalize_answer
BENCH = ["vqa_rad", "slake", "pathvqa", "pmcvqa", "omnimedvqa"]
MODELS = ["medgemma", "gemma3", "qwen", "qwen3b", "qwen2vl2b", "lingshu7b", "internvl3_8b", "qwen3vl8b"]
MAIN = list(MODELS)
MODELS = MODELS + ["qwen25vl32b", "medgemma27b"]  # scale add-on (rows only; overlap/transfer stay on the 8 main VLMs)
NAME = {"qwen25vl32b": "Qwen2.5-VL-32B", "medgemma27b": "MedGemma-27B", "medgemma": "MedGemma-4B", "gemma3": "Gemma-3-4B", "qwen": "Qwen2.5-VL-7B", "qwen3b": "Qwen2.5-VL-3B", "qwen2vl2b": "Qwen2-VL-2B", "lingshu7b": "Lingshu-7B", "internvl3_8b": "InternVL3-8B", "qwen3vl8b": "Qwen3-VL-8B"}
BN = {"vqa_rad": "VQA-RAD", "slake": "SLAKE", "pathvqa": "PathVQA", "pmcvqa": "PMC-VQA", "omnimedvqa": "OmniMedVQA"}
SPL = ["train", "validation", "test"]

def key(a, q):
    if is_mcq(a) or True:
        m = mcq_pred(a)
    yn = closed_pred_canonical(a, q)
    if yn in ("yes", "no"): return yn
    return canonicalize(a)

def load_keep(d, m):
    out = {}
    for s in SPL:
        p = RESULTS / "raw" / f"{d}_{m}_{s}_keep.jsonl"
        if not p.exists(): return None
        out[s] = {r["sample_id"]: r for r in read_jsonl(p)}
    return out

def main():
    samples = {d: {s: {r["sample_id"]: r for r in read_jsonl(DATA / d / f"{s}.jsonl")} for s in SPL if (DATA / d / f"{s}.jsonl").exists()} for d in BENCH}
    rows = []; err_by = {}
    for d in BENCH:
        if len(samples.get(d, {})) < 3: continue
        for m in MODELS:
            K = load_keep(d, m)
            if K is None: continue
            S = samples[d]
            txt, y, grp, sp, ids = [], [], [], [], []
            for s in SPL:
                for i, r in K[s].items():
                    q = S[s][i]; txt.append(q["question"].lower().split("\noptions:")[0] + " ||| " + key(r["answer_raw"], q["question"])); y.append(int(r["score"] < 1)); grp.append(q["image_path"]); sp.append(s); ids.append(i)
            y = np.array(y); sp = np.array(sp); grp = np.array(grp)
            if not (0 < y.mean() < 1): continue
            folds = list(GroupKFold(5).split(txt, y, grp))
            def cv_tf(T):
                p = np.zeros(len(y))
                try:
                    for tr, ho in folds:
                        v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b").fit([T[i] for i in tr]); c = LogisticRegression(C=1.0, max_iter=3000).fit(v.transform([T[i] for i in tr]), y[tr]); p[ho] = c.predict_proba(v.transform([T[i] for i in ho]))[:, 1]
                except ValueError:
                    return None
                return p
            p_qa = cv_tf(txt); p_q = cv_tf([t.split(" ||| ")[0] for t in txt]); p_a = cv_tf([t.split(" ||| ")[1] for t in txt])
            te = sp == "test"; yte = y[te]
            au = lambda pp: roc_auc_score(y, pp) if pp is not None else np.nan
            r = dict(benchmark=d, model=m, n_test=int(te.sum()), acc=1 - yte.mean(), err_rate_all=y.mean(),
                     auroc_qa_cv=au(p_qa), auroc_q_cv=au(p_q), auroc_a_cv=au(p_a), auroc_qa_test=roc_auc_score(yte, p_qa[te]) if 0 < yte.mean() < 1 else np.nan)
            # selective accuracy at coverage on the test split (CV predictions)
            o = np.argsort(p_qa[te])
            for cov in [0.7, 0.5]: r[f"selacc_{int(cov*100)}"] = 1 - yte[o[:int(cov * len(yte))]].mean()
            # zero-training signals on test
            conf = np.array([K["test"][i]["confidence"] if K["test"][i].get("confidence") is not None else np.nan for i in np.array(ids)[te]], float); conf = np.nan_to_num(conf, nan=np.nanmin(conf) if np.isfinite(np.nanmin(conf)) else 0)
            r["auroc_logprob"] = roc_auc_score(yte, -conf)
            ps = RESULTS / "sampling" / f"{d}_{m}_test.jsonl"
            if ps.exists():
                Sm = {x["sample_id"]: x for x in read_jsonl(ps)}; ag = np.array([Sm[i]["agree_greedy"] if i in Sm else np.nan for i in np.array(ids)[te]]); mk = ~np.isnan(ag)
                r["auroc_sampling"] = roc_auc_score(yte[mk], -ag[mk]) if mk.sum() > 10 else np.nan
            acts = {a: RESULTS / "raw" / f"{d}_{m}_test_{a}.jsonl" for a in ACTIONS}
            if all(p.exists() for p in acts.values()):
                Sc = {a: {x["sample_id"]: x for x in read_jsonl(acts[a])} for a in ACTIONS}; tid = [i for i in np.array(ids)[te] if all(i in Sc[a] for a in ACTIONS)]
                X = np.array([[Sc[a][i]["score"] for a in ACTIONS] for i in tid]); yt2 = (X[:, 0] < 1).astype(int)
                Kk = np.array([[key(Sc[a][i]["answer_raw"], S["test"][i]["question"]) for a in ACTIONS] for i in tid]); agree = np.array([sum(Kk[j, k] == Kk[j, 0] for k in range(1, 4)) for j in range(len(tid))])
                r.update(auroc_perturb=roc_auc_score(yt2, -agree) if 0 < yt2.mean() < 1 else np.nan, best_fixed=X.mean(0).max(), oracle=X.max(1).mean(), gap=X.max(1).mean() - X.mean(0).max(), oracle_keep=(X.argmax(1) == 0).mean(),
                         harm_mean=np.mean([(X[:, k] < X[:, 0]).mean() for k in range(1, 4)]), help_mean=np.mean([(X[:, k] > X[:, 0]).mean() for k in range(1, 4)]))
            cf = {c: RESULTS / "counterfactual" / f"{d}_{m}_test_{c}.jsonl" for c in ["noimage", "blank", "swap"]}
            if all(p.exists() for p in cf.values()):
                C = {c: {x["sample_id"]: x for x in read_jsonl(cf[c])} for c in cf}; tid = [i for i in np.array(ids)[te] if all(i in C[c] for c in C)]
                kk = np.array([key(K["test"][i]["answer_raw"], S["test"][i]["question"]) for i in tid]); acc0 = np.array([K["test"][i]["score"] for i in tid])
                same = np.array([key(C["swap"][i]["answer_raw"], S["test"][i]["question"]) == k for i, k in zip(tid, kk)])
                r.update(acc_noimage=np.mean([C["noimage"][i]["score"] for i in tid]), acc_blank=np.mean([C["blank"][i]["score"] for i in tid]), acc_swap=np.mean([C["swap"][i]["score"] for i in tid]),
                         swap_insensitive=same.mean(), acc_if_insensitive=acc0[same].mean() if same.any() else np.nan, acc_if_sensitive=acc0[~same].mean() if (~same).any() else np.nan,
                         frac_correct_insensitive=(same & (acc0 >= 1)).sum() / max((acc0 >= 1).sum(), 1))
            hp = {s: RESULTS / "hidden" / f"{d}_{m}_{s}.npz" for s in SPL}
            if all(p.exists() for p in hp.values()):
                Z = {s: np.load(hp[s]) for s in SPL}; Xh = {s: Z[s]["h"].astype(np.float32) for s in SPL}; yh = {s: np.array([int(K[s][i]["score"] < 1) for i in Z[s]["sample_id"]]) for s in SPL}
                pr = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(np.concatenate([Xh["train"], Xh["validation"]]), np.concatenate([yh["train"], yh["validation"]]))
                r["auroc_probe_test"] = roc_auc_score(yh["test"], pr.predict_proba(Xh["test"])[:, 1]) if 0 < yh["test"].mean() < 1 else np.nan
            rows.append(r); err_by[(d, m)] = {i: int(K["test"][i]["score"] < 1) for i in K["test"]}
            print(d, m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("acc", "auroc_qa_cv", "auroc_probe_test", "gap", "swap_insensitive")}, flush=True)
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "stageB_matrix.csv", index=False)
    # overlap per benchmark
    ov = []; tr_rows = []
    for d in BENCH:
        ms = [m for m in MAIN if (d, m) in err_by]
        if len(ms) < 2: continue
        ids = sorted(set.intersection(*[set(err_by[(d, m)]) for m in ms])); X = np.array([[err_by[(d, m)][i] for m in ms] for i in ids], float)
        ph = np.corrcoef(X.T); n = X.sum(1)
        ov.append(dict(benchmark=d, n_models=len(ms), n_items=len(ids), mean_phi=ph[np.triu_indices(len(ms), 1)].mean(), min_phi=ph[np.triu_indices(len(ms), 1)].min(), all_wrong=(n == len(ms)).mean(), all_wrong_indep=np.prod(X.mean(0)), none_wrong=(n == 0).mean()))
        # transfer: TF-IDF trained on model a (train+val) -> model b test
        S = samples[d]
        cache = {}
        for m in ms:
            K = load_keep(d, m); T = {s: [S[s][i]["question"].lower().split("\noptions:")[0] + " ||| " + key(r["answer_raw"], S[s][i]["question"]) for i, r in K[s].items()] for s in SPL}; Y = {s: np.array([int(r["score"] < 1) for r in K[s].values()]) for s in SPL}
            cache[m] = (T, Y)
        for a in ms:
            Ta, Ya = cache[a]; v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(Ta["train"] + Ta["validation"]); c = LogisticRegression(C=1.0, max_iter=3000).fit(v.transform(Ta["train"] + Ta["validation"]), np.concatenate([Ya["train"], Ya["validation"]]))
            for b in ms:
                Tb, Yb = cache[b]
                if 0 < Yb["test"].mean() < 1: tr_rows.append(dict(benchmark=d, train_model=a, test_model=b, auroc=roc_auc_score(Yb["test"], c.predict_proba(v.transform(Tb["test"]))[:, 1])))
    OV = pd.DataFrame(ov); OV.to_csv(RESULTS / "stageB_overlap.csv", index=False); TR = pd.DataFrame(tr_rows); TR.to_csv(RESULTS / "stageB_transfer.csv", index=False)
    # tables
    out = ["## Stage B — all benchmarks × VLMs\n", "| Benchmark | VLM | n test | acc | AUROC q+a (CV) | q only | a only | probe | log-prob | sampling | perturb. | no-image acc | swap acc | swap-insens. % | acc\\|insens. | acc\\|sens. | % correct insens. | best fixed | oracle | gap | harm mean | selacc@70 | selacc@50 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f = lambda v, d=3: ("--" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{100*v:.1f}" if d == 1 else f"{v:.3f}"))
    for _, r in R.iterrows():
        g = lambda k, d=3: f(r.get(k, np.nan), d)
        out.append(f"| {BN[r.benchmark]} | {NAME[r.model]} | {r.n_test} | {g('acc',1)} | {g('auroc_qa_cv')} | {g('auroc_q_cv')} | {g('auroc_a_cv')} | {g('auroc_probe_test')} | {g('auroc_logprob')} | {g('auroc_sampling')} | {g('auroc_perturb')} | {g('acc_noimage',1)} | {g('acc_swap',1)} | {g('swap_insensitive',1)} | {g('acc_if_insensitive',1)} | {g('acc_if_sensitive',1)} | {g('frac_correct_insensitive',1)} | {g('best_fixed',1)} | {g('oracle',1)} | {g('gap',1)} | {g('harm_mean',1)} | {g('selacc_70',1)} | {g('selacc_50',1)} |")
    out += ["\n### Cross-VLM error overlap per benchmark\n", "| Benchmark | # VLMs | n | mean φ | min φ | all wrong % | all wrong if indep. % | none wrong % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in OV.iterrows(): out.append(f"| {BN[r.benchmark]} | {r.n_models} | {r.n_items} | {r.mean_phi:.2f} | {r.min_phi:.2f} | {100*r.all_wrong:.1f} | {100*r.all_wrong_indep:.1f} | {100*r.none_wrong:.1f} |")
    if not TR.empty:
        out += ["\n### Cross-VLM transfer of the TF-IDF error predictor (AUROC on the target VLM's test errors; diagonal = self)\n"]
        for d in BENCH:
            g = TR[TR.benchmark == d]
            if g.empty: continue
            ms = [m for m in MAIN if m in set(g.train_model)]
            out.append(f"\n**{BN[d]}**\n"); out.append("| train \\ test | " + " | ".join(NAME[m] for m in ms) + " |"); out.append("|---|" + "---:|" * len(ms))
            for a in ms: out.append(f"| {NAME[a]} | " + " | ".join(f"{g[(g.train_model == a) & (g.test_model == b)].auroc.iloc[0]:.2f}" if len(g[(g.train_model == a) & (g.test_model == b)]) else "--" for b in ms) + " |")
    (ROOT / "tables" / "stageB.md").write_text("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    main()
