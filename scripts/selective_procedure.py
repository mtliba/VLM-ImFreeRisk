"""Item D: selective answering as a procedure. For every (benchmark, VLM):
  - predictor p(error) = TF-IDF(question+answer) logistic model fitted on train, calibrated (isotonic) on validation;
  - thresholds chosen on validation for target selective risk r* in {5, 10, 20 %} (answer only if p(error) <= t);
  - report on test: achieved risk, coverage, and whether the target was met; compare to log-prob confidence and (if present) the
    hidden-state probe and sampling consistency under the same procedure;
  - calibration on test (ECE), and cross-benchmark transfer of thresholds (procedure fitted on one benchmark applied to another, same VLM).
Outputs: results/selective_procedure.csv, results/selective_transfer.csv, tables/selective_procedure.md"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonicalize, closed_pred_canonical
from common import DATA, RESULTS, ROOT, read_jsonl
BENCH = ["vqa_rad", "slake", "pathvqa", "pmcvqa", "omnimedvqa"]
MODELS = ["medgemma", "gemma3", "qwen", "qwen3b", "qwen2vl2b", "lingshu7b", "internvl3_8b", "qwen3vl8b"]
MODELS = MODELS + ["qwen25vl32b", "medgemma27b"]
NAME = {"qwen25vl32b": "Qwen2.5-VL-32B", "medgemma27b": "MedGemma-27B", "medgemma": "MedGemma-4B", "gemma3": "Gemma-3-4B", "qwen": "Qwen2.5-VL-7B", "qwen3b": "Qwen2.5-VL-3B", "qwen2vl2b": "Qwen2-VL-2B", "lingshu7b": "Lingshu-7B", "internvl3_8b": "InternVL3-8B", "qwen3vl8b": "Qwen3-VL-8B"}
BN = {"vqa_rad": "VQA-RAD", "slake": "SLAKE", "pathvqa": "PathVQA", "pmcvqa": "PMC-VQA", "omnimedvqa": "OmniMedVQA"}
SPL = ["train", "validation", "test"]; TARGETS = [0.05, 0.10, 0.20]

def key(a, q):
    yn = closed_pred_canonical(a, q); return yn if yn in ("yes", "no") else canonicalize(a)

def ece(p, y, bins=10):
    e = 0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

def load(d, m):
    S = {s: {r["sample_id"]: r for r in read_jsonl(DATA / d / f"{s}.jsonl")} for s in SPL}
    K = {}
    for s in SPL:
        p = RESULTS / "raw" / f"{d}_{m}_{s}_keep.jsonl"
        if not p.exists(): return None
        K[s] = {r["sample_id"]: r for r in read_jsonl(p)}
    out = {}
    for s in SPL:
        ids = list(K[s]); q = [S[s][i]["question"].lower().split("\noptions:")[0] for i in ids]; a = [key(K[s][i]["answer_raw"], S[s][i]["question"]) for i in ids]
        out[s] = dict(ids=ids, txt=[qq + " ||| " + aa for qq, aa in zip(q, a)], y=np.array([int(K[s][i]["score"] < 1) for i in ids]),
                      conf=np.nan_to_num(np.array([K[s][i].get("confidence", np.nan) for i in ids], float), nan=-50))
    hp = {s: RESULTS / "hidden" / f"{d}_{m}_{s}.npz" for s in SPL}
    if all(p.exists() for p in hp.values()):
        for s in SPL:
            z = np.load(hp[s]); pos = {i: j for j, i in enumerate(z["sample_id"])}; out[s]["h"] = z["h"].astype(np.float32)[[pos[i] for i in out[s]["ids"]]]
    for s in ["validation", "test"]:  # sampling self-consistency (J1 added the validation split so it can enter the procedure)
        ps = RESULTS / "sampling" / f"{d}_{m}_{s}.jsonl"
        if ps.exists():
            Sm = {r["sample_id"]: r for r in read_jsonl(ps)}; out[s]["samp"] = np.array([Sm[i]["agree_greedy"] if i in Sm else np.nan for i in out[s]["ids"]])
    return out

def scores(D):
    """return dict signal -> (p_val, p_test) as probability-like error scores (higher = more likely wrong)."""
    sig = {}
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b").fit(D["train"]["txt"]); c = LogisticRegression(C=1.0, max_iter=3000).fit(v.transform(D["train"]["txt"]), D["train"]["y"])
    sig["text prior"] = (c.predict_proba(v.transform(D["validation"]["txt"]))[:, 1], c.predict_proba(v.transform(D["test"]["txt"]))[:, 1])
    sig["log-prob"] = (-D["validation"]["conf"], -D["test"]["conf"])
    if "samp" in D["validation"] and "samp" in D["test"] and np.isfinite(D["validation"]["samp"]).mean() > 0.95 and np.isfinite(D["test"]["samp"]).mean() > 0.95:
        sig["sampling"] = (1 - np.nan_to_num(D["validation"]["samp"], nan=1.0), 1 - np.nan_to_num(D["test"]["samp"], nan=1.0))
    if "h" in D["train"]:
        pr = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(D["train"]["h"], D["train"]["y"])
        sig["hidden probe"] = (pr.predict_proba(D["validation"]["h"])[:, 1], pr.predict_proba(D["test"]["h"])[:, 1])
        both_tr = np.stack([c.predict_proba(v.transform(D["train"]["txt"]))[:, 1], pr.predict_proba(D["train"]["h"])[:, 1]], 1)
        comb = LogisticRegression().fit(both_tr, D["train"]["y"])
        sig["prior + probe"] = (comb.predict_proba(np.stack([sig["text prior"][0], sig["hidden probe"][0]], 1))[:, 1], comb.predict_proba(np.stack([sig["text prior"][1], sig["hidden probe"][1]], 1))[:, 1])
    return sig

def procedure(pv, yv, pt, yt):
    """isotonic calibration on validation, then threshold for each target risk chosen on validation; evaluate on test."""
    iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yv); qv, qt = iso.predict(pv), iso.predict(pt)
    res = {}
    for r in TARGETS:
        ts = np.unique(qv); best = None
        for t in ts:
            m = qv <= t
            if m.sum() >= 20 and yv[m].mean() <= r: best = t if best is None else max(best, t)
        if best is None:
            res[r] = dict(threshold=np.nan, test_cov=0.0, test_risk=np.nan, met=False); continue
        m = qt <= best; res[r] = dict(threshold=best, test_cov=m.mean(), test_risk=yt[m].mean() if m.any() else np.nan, met=bool(m.any() and yt[m].mean() <= r * 1.25))
    return res, qt

def main():
    rows, tr_rows = [], []; fitted = {}
    for d in BENCH:
        for m in MODELS:
            D = load(d, m)
            if D is None: continue
            sig = scores(D); yv, yt = D["validation"]["y"], D["test"]["y"]
            if not (0 < yt.mean() < 1): continue
            for name, (pv, pt) in sig.items():
                res, qt = procedure(pv, yv, pt, yt)
                r = dict(benchmark=d, model=m, signal=name, auroc_test=roc_auc_score(yt, pt), ece_test=ece(qt, yt), base_err=yt.mean())
                for tgt in TARGETS:
                    r[f"cov@{int(tgt*100)}"] = res[tgt]["test_cov"]; r[f"risk@{int(tgt*100)}"] = res[tgt]["test_risk"]; r[f"met@{int(tgt*100)}"] = res[tgt]["met"]
                rows.append(r)
            fitted[(d, m)] = (sig["text prior"], D)
            print(d, m, {k: round(v, 3) for k, v in rows[-len(sig)].items() if isinstance(v, float)}, flush=True)
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "selective_procedure.csv", index=False)
    # cross-benchmark transfer of the whole procedure (text prior fitted+calibrated on benchmark A, applied to benchmark B's test), same VLM
    for m in MODELS:
        for a in BENCH:
            if (a, m) not in fitted: continue
            Da = load(a, m); v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b").fit(Da["train"]["txt"]); c = LogisticRegression(C=1.0, max_iter=3000).fit(v.transform(Da["train"]["txt"]), Da["train"]["y"])
            pv = c.predict_proba(v.transform(Da["validation"]["txt"]))[:, 1]; iso = IsotonicRegression(out_of_bounds="clip").fit(pv, Da["validation"]["y"])
            for b in BENCH:
                if b == a or (b, m) not in fitted: continue
                Db = load(b, m); pt = iso.predict(c.predict_proba(v.transform(Db["test"]["txt"]))[:, 1]); yt = Db["test"]["y"]
                tr_rows.append(dict(model=m, fit_on=a, test_on=b, auroc=roc_auc_score(yt, pt) if 0 < yt.mean() < 1 else np.nan, ece=ece(pt, yt)))
    T = pd.DataFrame(tr_rows); T.to_csv(RESULTS / "selective_transfer.csv", index=False)
    out = ["## Selective answering procedure (thresholds and calibration set on validation only; test = held out)\n",
           "| Benchmark | VLM | signal | AUROC | ECE | base error | cov@5% / risk | cov@10% / risk | cov@20% / risk |", "|---|---|---|---:|---:|---:|---|---|---|"]
    f = lambda c, r, met: f"{100*c:.0f}% / {100*r:.1f}%{'' if met else ' ✗'}" if not np.isnan(r) else "no threshold"
    for _, r in R.iterrows():
        out.append(f"| {BN[r.benchmark]} | {NAME[r.model]} | {r.signal} | {r.auroc_test:.3f} | {r.ece_test:.3f} | {100*r.base_err:.1f}% | " + " | ".join(f(r[f'cov@{t}'], r[f'risk@{t}'], r[f'met@{t}']) for t in [5, 10, 20]) + " |")
    if not T.empty:
        out += ["\n### Cross-benchmark transfer of the fitted procedure (text prior; same VLM)\n", "| VLM | fit on | test on | AUROC | ECE |", "|---|---|---|---:|---:|"]
        for _, r in T.iterrows(): out.append(f"| {NAME[r.model]} | {BN[r.fit_on]} | {BN[r.test_on]} | {r.auroc:.3f} | {r.ece:.3f} |")
    (ROOT / "tables" / "selective_procedure.md").write_text("\n".join(out) + "\n"); print("\n".join(out[:40]))

if __name__ == "__main__":
    main()
