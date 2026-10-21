"""Generate every LaTeX table (paper/tables/*.tex) and figure (paper/figures/*.pdf|png) of the ICASSP draft from results/.
Also computes the 'advanced analysis' quantities: bootstrap CIs of AUROC, per-question-type x model AUROC matrix, calibration (ECE,
reliability), image-sensitivity x correctness breakdown, five-model overlap and #models-wrong distribution, predicted-error histograms."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonicalize, closed_pred_canonical
from common import ACTIONS, DATA, RESULTS, ROOT, read_jsonl
from consensus_stage1 import load

P = ROOT / "paper"; (P / "tables").mkdir(parents=True, exist_ok=True); (P / "figures").mkdir(parents=True, exist_ok=True)
MAIN = [("vqa_rad", "medgemma"), ("vqa_rad", "qwen"), ("slake", "medgemma"), ("slake", "qwen")]
FIVE = ["medgemma", "gemma3", "qwen", "qwen3b", "qwen2vl2b"]
NAME = {"medgemma": "MedGemma-4B", "gemma3": "Gemma-3-4B", "qwen": "Qwen2.5-VL-7B", "qwen3b": "Qwen2.5-VL-3B", "qwen2vl2b": "Qwen2-VL-2B", "lingshu7b": "Lingshu-7B", "internvl3_8b": "InternVL3-8B", "qwen3vl8b": "Qwen3-VL-8B"}
DSN = {"vqa_rad": "VQA-RAD", "slake": "SLAKE"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
rng = np.random.RandomState(0)
M = load(); M["keep_score_b"] = pd.to_numeric(M.keep_score_b)
SAMPLES = {d: {s: {r["sample_id"]: r for r in read_jsonl(DATA / d / f"{s}.jsonl")} for s in ["train", "validation", "test"]} for d in DSN}

def key(a, q):
    yn = closed_pred_canonical(a, q); return yn if yn in ("yes", "no") else canonicalize(a)

def tex(name, body):
    (P / "tables" / f"{name}.tex").write_text(body); print("table", name)

def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

def keep(d, m, s):
    p = RESULTS / "raw" / f"{d}_{m}_{s}_keep.jsonl"; return {r["sample_id"]: r for r in read_jsonl(p)} if p.exists() else None

# ------------------------------------------------------------------ 1. datasets / models tables
st = json.load(open(RESULTS / "dataset_stats.json"))
rows = []
for d in ["vqa_rad", "slake"]:
    for s in ["train", "validation", "test"]:
        x = st[d][s]; rows.append(f"{DSN[d]} & {s} & {x['n_qa']} & {x['n_images']} & {x['closed']} & {x['open']} \\\\")
tex("datasets", "\\begin{tabular}{llrrrr}\\toprule Dataset & Split & QA pairs & Images & Yes/No & Open \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
acc = {}
for d in DSN:
    for m in FIVE:
        K = keep(d, m, "test"); acc[(d, m)] = np.mean([r["score"] for r in K.values()]) if K else np.nan
rows = [f"{NAME[m]} & {'medical' if m == 'medgemma' else 'general'} & {sz} & {100*acc[('vqa_rad', m)]:.1f} & {100*acc[('slake', m)]:.1f} \\\\" for m, sz in zip(FIVE, ["4B", "4B", "7B", "3B", "2B"])]
tex("models", "\\begin{tabular}{llrrr}\\toprule VLM & Domain & Params & VQA-RAD acc. & SLAKE acc. \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

# ------------------------------------------------------------------ 2. main predictability table (+ bootstrap CIs on the test split, official train+val -> test)
E = pd.read_csv(RESULTS / "error_predictability.csv"); EA = E[E.label == "A"]; EB = E[E.label == "B"]
H1 = pd.read_csv(RESULTS / "error_h1_sampling.csv").set_index(["dataset", "model"])
order = [("conf", "Token log-prob (no training)"), ("agree", "Perturbation consistency (R/E/Z)"), ("img", "Image embedding"), ("q", "Question embedding"), ("y0", "Answer embedding"), ("q+y0", "Question + answer embeddings"), ("img+q+y0", "Image + question + answer"),
         ("clip img+q+y0", "Generic CLIP image + question + answer"), ("lookup question only", "Lookup: question"), ("lookup answer only", "Lookup: answer"), ("lookup qtype x answer", "Lookup: question type $\\times$ answer"), ("lookup question x answer", "Lookup: question $\\times$ answer"), ("img+q+y0+agree+conf", "All signals")]
def cell(df, d, m, p, col="auroc_cv"):
    g = df[(df.dataset == d) & (df.model == m) & (df.predictor == p)]; return f"{g[col].iloc[0]:.3f}" if len(g) else "--"
rows = []
for p, lab in order:
    rows.append(lab + " & " + " & ".join(cell(EA, d, m, p) for d, m in MAIN) + " & " + " & ".join(cell(EB, d, m, p) for d, m in MAIN) + " \\\\")
rows.insert(1, "Sampling self-consistency (5$\\times$, test) & " + " & ".join(f"{H1.loc[(d, m), 'auroc_samp_agree']:.3f}" for d, m in MAIN) + " & -- & -- & -- & -- \\\\")
tex("predictability", "\\begin{tabular}{lcccccccc}\\toprule & \\multicolumn{4}{c}{Metric A (original)} & \\multicolumn{4}{c}{Metric B (canonicalised)} \\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-9} Predictor of ``answer is wrong'' & VR-MG & VR-Q & SL-MG & SL-Q & VR-MG & VR-Q & SL-MG & SL-Q \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

# bootstrap CIs of AUROC on the test split for key predictors (official train+val -> test), plus paired differences
def feats(d, m, s, which):
    z = np.load(RESULTS / "features" / f"{d}_{m}_{s}.npz"); ids = list(z["sample_id"]); te = M[(M.dataset == d) & (M.model == m) & (M.split == s)].set_index("sample_id").loc[ids]
    K = te[[f"{a}_key" for a in ACTIONS]].values; agree = np.array([sum(K[i, j] == K[i, 0] for j in range(1, 4)) for i in range(len(te))], float)
    conf = te.base_conf.fillna(te.base_conf.min()).values
    cols = {"img": z["z_img"], "q": z["z_q"], "y0": z["z_y0"], "agree": agree[:, None], "conf": conf[:, None]}
    return np.concatenate([cols[w] for w in which], 1), (te.keep_score.values < 1).astype(int), ids
ci_rows = []; preds_test = {}
for d, m in MAIN:
    P_ = {}
    for name, which in [("Question+answer emb.", ["q", "y0"]), ("Image+question+answer emb.", ["img", "q", "y0"]), ("Image emb.", ["img"]), ("Log-prob", ["conf"]), ("Perturbation consistency", ["agree"]), ("All signals", ["img", "q", "y0", "agree", "conf"])]:
        Xtr, ytr, _ = feats(d, m, "train", which); Xva, yva, _ = feats(d, m, "validation", which); Xte, yte, ids = feats(d, m, "test", which)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
        P_[name] = clf.predict_proba(Xte)[:, 1] if which != ["conf"] else -Xte[:, 0]
        if which == ["agree"]: P_[name] = -Xte[:, 0]
    S = {r["sample_id"]: r for r in read_jsonl(RESULTS / "sampling" / f"{d}_{m}_test.jsonl")}; P_["Sampling consistency"] = -np.array([S[i]["agree_greedy"] for i in ids])
    preds_test[(d, m)] = (P_, yte, ids)
    idx = rng.randint(0, len(yte), (1000, len(yte)))
    for name, p in P_.items():
        b = [roc_auc_score(yte[i], p[i]) for i in idx if 0 < yte[i].mean() < 1]
        dq = [roc_auc_score(yte[i], P_["Question+answer emb."][i]) - roc_auc_score(yte[i], p[i]) for i in idx if 0 < yte[i].mean() < 1]
        ci_rows.append(dict(dataset=d, model=m, predictor=name, auroc=roc_auc_score(yte, p), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), d_lo=np.percentile(dq, 2.5), d_hi=np.percentile(dq, 97.5)))
CI = pd.DataFrame(ci_rows); CI.to_csv(RESULTS / "paper_auroc_bootstrap.csv", index=False)
rows = []
for name in ["Question+answer emb.", "Image+question+answer emb.", "Image emb.", "Log-prob", "Sampling consistency", "Perturbation consistency", "All signals"]:
    cells = []
    for d, m in MAIN:
        g = CI[(CI.dataset == d) & (CI.model == m) & (CI.predictor == name)].iloc[0]
        star = "" if name == "Question+answer emb." else ("$^{*}$" if (g.d_lo > 0 or g.d_hi < 0) else "")
        cells.append(f"{g.auroc:.3f} [{g.lo:.2f}, {g.hi:.2f}]{star}")
    rows.append(name + " & " + " & ".join(cells) + " \\\\")
tex("auroc_ci", "\\begin{tabular}{lcccc}\\toprule Predictor (train+val $\\to$ test) & VR-MG & VR-Q & SL-MG & SL-Q \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

# ------------------------------------------------------------------ 3. hypotheses, probe, variance, counterfactual, twin
PR = pd.read_csv(RESULTS / "stageA_probe.csv"); CF = pd.read_csv(RESULTS / "stageA_counterfactual.csv"); VA = pd.read_csv(RESULTS / "stageA_variance.csv")
rows = []
for d in DSN:
    for m in FIVE:
        g = PR[(PR.dataset == d) & (PR.model == m)]
        if g.empty: continue
        g = g.iloc[0]; rows.append(f"{DSN[d]} & {NAME[m]} & {g.dim} & {g.probe_test:.3f} & {g.probe_cv:.3f} & {g.tfidf_test:.3f} & {g.comb_cv:.3f} \\\\")
tex("probe", "\\begin{tabular}{llrcccc}\\toprule Dataset & VLM & $d$ & Probe (test) & Probe (CV) & q+a$_{\\rm T}$ (test) & Probe + q+a$_{\\rm T}$ (CV) \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
rows = []
for d in DSN:
    for m in FIVE:
        g = CF[(CF.dataset == d) & (CF.model == m)]
        if g.empty: continue
        g = g.iloc[0]; rows.append(f"{DSN[d]} & {NAME[m]} & {100*g.acc:.1f} & {100*g.acc_noimage:.1f} & {100*g.acc_blank:.1f} & {100*g.acc_swap:.1f} & {100*g.same_swap:.1f} & {100*g.same_blank:.1f} & {100*g.same_noimage:.1f} & {100*g.acc_if_insensitive:.1f} & {100*g.acc_if_sensitive:.1f} & {100*g.same_swap*g.acc_if_insensitive/g.acc:.0f} & {esc(g.terciles)} \\\\")
tex("counterfactual", "\\begin{tabular}{llrrrrrrrrrrl}\\toprule & & \\multicolumn{4}{c}{Accuracy with} & \\multicolumn{3}{c}{Answer unchanged (\\%)} & \\multicolumn{2}{c}{Acc.\\ (image) if} & & Swap-insens.\\ rate: \\\\ \\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11} Dataset & VLM & image & none & blank & swap & swap & blank & none & insens. & sens. & \\% correct that are insens. & pred.\\ wrong vs right \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
rows = [f"{DSN[r.dataset]} & {r.n_models} & {r.n_items} & {r.r2_model:.3f} & {r.r2_item:.3f} & {r.r2_both:.3f} & {r.icc_item:.2f} \\\\" for _, r in VA.iterrows()]
tex("variance", "\\begin{tabular}{lrrcccc}\\toprule Dataset & VLMs & Items & $R^2$ model & $R^2$ item & $R^2$ both & Item ICC \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

# five-model overlap + twin
ov_rows = []; phis = {}
for d in DSN:
    E5 = {m: {i: r["score"] < 1 for i, r in keep(d, m, "test").items()} for m in FIVE}
    ids = sorted(set.intersection(*[set(E5[m]) for m in FIVE])); X = np.array([[E5[m][i] for m in FIVE] for i in ids], float)
    phis[d] = np.corrcoef(X.T); n = X.sum(1)
    mg, g3 = X[:, 0].astype(bool), X[:, 1].astype(bool)
    ov_rows.append(dict(dataset=d, n=len(ids), all5=(n == 5).mean(), ge3=(n >= 3).mean(), none=(n == 0).mean(), indep=np.prod(X.mean(0)), mean_phi=phis[d][np.triu_indices(5, 1)].mean(),
                        both=(mg & g3).mean(), fixed=(~mg & g3).mean(), broken=(mg & ~g3).mean(), phi_twin=np.corrcoef(mg, g3)[0, 1], nwrong=np.bincount(n.astype(int), minlength=6) / len(ids)))
OV = pd.DataFrame(ov_rows)
rows = []
for d in DSN:
    ph = phis[d]
    for i, m in enumerate(FIVE):
        rows.append(f"{DSN[d]} & {NAME[m]} & " + " & ".join(("--" if j == i else f"{ph[i, j]:.2f}") for j in range(5)) + " \\\\")
    rows.append("\\midrule")
tex("overlap", "\\begin{tabular}{ll" + "c" * 5 + "}\\toprule Dataset & VLM & " + " & ".join(NAME[m] for m in FIVE) + " \\\\\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\\end{tabular}")
rows = [f"{DSN[r.dataset]} & {r.n} & {100*r.all5:.1f} & {100*r.indep:.1f} & {100*r.ge3:.1f} & {100*r.none:.1f} & {r.mean_phi:.2f} & {r.phi_twin:.2f} & {100*r.both:.1f} & {100*r.fixed:.1f} & {100*r.broken:.1f} \\\\" for _, r in OV.iterrows()]
tex("overlap_summary", "\\begin{tabular}{lrrrrrrrrrr}\\toprule & & \\multicolumn{4}{c}{Items wrong for (\\% of test)} & & \\multicolumn{4}{c}{MedGemma vs.\\ Gemma-3 (\\%)} \\\\ \\cmidrule(lr){3-6}\\cmidrule(lr){8-11} Dataset & $n$ & all 5 & all 5 if indep. & $\\ge$3 & none & mean $\\phi$ & $\\phi$ & both wrong & fixed by med.\\ tuning & broken \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

# ------------------------------------------------------------------ 4. systematic clusters, per-type matrix (advanced), calibration
CL = pd.read_csv(RESULTS / "error_extra_clusters.csv")
rows = [f"{DSN[r.dataset]} & {NAME[r.model]} & {r.total_errors} & {100*r.errors_in_top10_clusters/r.total_errors:.0f} & {100*r.errors_on_always_wrong_pairs/r.total_errors:.0f} & {r.n_always_wrong_pairs} & {100*r.errors_on_pairs_seen_ge3/r.total_errors:.0f} \\\\" for _, r in CL.iterrows()]
tex("clusters", "\\begin{tabular}{llrrrrr}\\toprule Dataset & VLM & Errors (B) & in top-10 (q,a) clusters (\\%) & on always-wrong pairs (\\%) & \\# pairs & on pairs seen $\\ge$3$\\times$ (\\%) \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
# top clusters listing
rows = []
for d, m in MAIN:
    g = M[(M.dataset == d) & (M.model == m)].copy(); g["err"] = (g.keep_score_b < 1).astype(int); g["q"] = g.question.str.lower().str.strip()
    t = g.groupby(["q", "keep_key"]).agg(n=("err", "size"), err=("err", "mean"), refs=("reference_answer", lambda x: "; ".join(f"{k} ({v})" for k, v in x.value_counts().head(2).items()))).reset_index(); t["errors"] = t.n * t.err
    for _, r in t.sort_values("errors", ascending=False).head(4).iterrows():
        rows.append(f"{DSN[d]}/{NAME[m].split('-')[0]} & {esc(r.q[:48])} & {esc(r.keep_key)} & {r.n} & {100*r.err:.0f} & {esc(r.refs[:60])} \\\\")
tex("clusters_examples", "\\begin{tabular}{llllrrl}\\toprule Setting & Question & Model answer & $n$ & Err.\\ \\% & Reference answers (count) \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

# per-question-type x model AUROC (q+a TF-IDF, image-grouped CV) + image-sensitivity per type  (ADVANCED)
adv = []
for d in DSN:
    for m in FIVE:
        Ks = {s: keep(d, m, s) for s in ["train", "validation", "test"]}
        if any(v is None for v in Ks.values()): continue
        txt, y, grp, qt, at, ids_all = [], [], [], [], [], []
        for s in Ks:
            for i, r in Ks[s].items():
                q = SAMPLES[d][s][i]; txt.append(q["question"].lower() + " ||| " + key(r["answer_raw"], q["question"])); y.append(int(r["score"] < 1)); grp.append(q["image_path"]); qt.append(q["question_type"]); at.append(q["answer_type"]); ids_all.append((s, i))
        y = np.array(y); qt = np.array(qt); at = np.array(at); p = np.zeros(len(y))
        for tr, ho in GroupKFold(5).split(txt, y, np.array(grp)):
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([txt[i] for i in tr]); c = LogisticRegression(C=1.0, max_iter=3000).fit(v.transform([txt[i] for i in tr]), y[tr]); p[ho] = c.predict_proba(v.transform([txt[i] for i in ho]))[:, 1]
        # ECE (10 bins) on the CV predictions
        edges = np.linspace(0, 1, 11); ece = sum(((p > lo) & (p <= hi)).mean() * abs(p[(p > lo) & (p <= hi)].mean() - y[(p > lo) & (p <= hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:]) if ((p > lo) & (p <= hi)).any())
        cf = RESULTS / "counterfactual" / f"{d}_{m}_test_swap.jsonl"; SW = {r["sample_id"]: r for r in read_jsonl(cf)} if cf.exists() else {}
        for t in sorted(set(qt)) + ["[CLOSED]", "[OPEN]"]:
            mk = (at == t[1:-1]) if t.startswith("[") else (qt == t)
            if mk.sum() < 30 or not (0 < y[mk].mean() < 1): continue
            sens = np.nan
            if SW:
                ii = [k for k in np.where(mk)[0] if ids_all[k][0] == "test"]
                if ii: sens = np.mean([key(SW[ids_all[k][1]]["answer_raw"], SAMPLES[d]["test"][ids_all[k][1]]["question"]) != key(Ks["test"][ids_all[k][1]]["answer_raw"], SAMPLES[d]["test"][ids_all[k][1]]["question"]) for k in ii])
            adv.append(dict(dataset=d, model=m, qtype=t, n=int(mk.sum()), err=y[mk].mean(), auroc=roc_auc_score(y[mk], p[mk]), swap_sensitive=sens, ece=ece))
ADV = pd.DataFrame(adv); ADV.to_csv(RESULTS / "paper_advanced_pertype.csv", index=False)
for d in DSN:
    g = ADV[ADV.dataset == d]; types = [t for t in g.qtype.unique()]
    rows = []
    for t in types:
        cells = []
        for m in FIVE:
            h = g[(g.model == m) & (g.qtype == t)]
            cells.append(f"{h.auroc.iloc[0]:.2f} / {100*h.err.iloc[0]:.0f} / {100*h.swap_sensitive.iloc[0]:.0f}" if len(h) else "--")
        rows.append(("all " + t[1:-1].lower() + " questions" if t.startswith("[") else esc(t)) + " & " + " & ".join(cells) + " \\\\")
    tex(f"pertype_{d}", "\\begin{tabular}{l" + "c" * 5 + "}\\toprule Question type & " + " & ".join(NAME[m] for m in FIVE) + " \\\\ & \\multicolumn{5}{c}{AUROC / error rate (\\%) / answer changed under image swap (\\%)} \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
rows = [f"{DSN[d]} & " + " & ".join(f"{ADV[(ADV.dataset == d) & (ADV.model == m)].ece.iloc[0]:.3f}" if len(ADV[(ADV.dataset == d) & (ADV.model == m)]) else "--" for m in FIVE) + " \\\\" for d in DSN]
tex("calibration", "\\begin{tabular}{l" + "c" * 5 + "}\\toprule Dataset & " + " & ".join(NAME[m] for m in FIVE) + " \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

# ------------------------------------------------------------------ 5. selective answering
RC = pd.read_csv(RESULTS / "error_extra_risk_coverage.csv"); C2 = pd.read_csv(RESULTS / "error_predictability_coverage.csv"); C2 = C2[C2.label == "A"]
rows = []
for d, m in MAIN:
    g = RC[(RC.dataset == d) & (RC.model == m)].set_index("coverage"); c = C2[(C2.dataset == d) & (C2.model == m)]
    lp = c[c.predictor == "conf"].set_index("coverage").selective_accuracy; ac = c[c.predictor == "agree+conf"].set_index("coverage").selective_accuracy
    rows.append(f"{DSN[d]} & {NAME[m]} & {100*g.loc['1.0','selective_acc']:.1f} & " + " & ".join(f"{100*g.loc[str(cv),'selective_acc']:.1f} [{100*g.loc[str(cv),'ci_low']:.0f},{100*g.loc[str(cv),'ci_high']:.0f}]" for cv in [0.9, 0.8, 0.7, 0.5]) + f" & {100*lp.loc[0.5]:.1f} & {100*ac.loc[0.5]:.1f} & {g.loc['AURC','selective_acc']:.3f} \\\\")
tex("selective", "\\begin{tabular}{llr" + "c" * 4 + "rrr}\\toprule & & & \\multicolumn{4}{c}{Selective accuracy (\\%) at coverage, q+a prior [95\\% CI]} & \\multicolumn{2}{c}{@50\\%, baselines} & \\\\ \\cmidrule(lr){4-7}\\cmidrule(lr){8-9} Dataset & VLM & 100\\% & 90\\% & 80\\% & 70\\% & 50\\% & log-prob & consist.+log-prob & AURC \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")

# ------------------------------------------------------------------ 6. intervention summary (oracle gap / harm) incl. new models; selectors / consensus / verification
O = pd.read_csv(RESULTS / "oracle_analysis.csv"); O = O[O.split == "test"].set_index(["dataset", "model"])
H2 = pd.read_csv(RESULTS / "error_h2_models.csv").set_index(["dataset", "model"])
rows = []
for d in DSN:
    for m in FIVE:
        acts = {a: RESULTS / "raw" / f"{d}_{m}_test_{a}.jsonl" for a in ACTIONS}
        if not all(p.exists() for p in acts.values()): continue
        Sc = {a: {r["sample_id"]: r["score"] for r in read_jsonl(acts[a])} for a in ACTIONS}; ids = sorted(set.intersection(*[set(v) for v in Sc.values()]))
        X = np.array([[Sc[a][i] for a in ACTIONS] for i in ids]); base = X[:, 0]
        harm = [100 * (X[:, j] < base).mean() for j in range(1, 4)]; helpr = [100 * (X[:, j] > base).mean() for j in range(1, 4)]
        rows.append(f"{DSN[d]} & {NAME[m]} & {100*base.mean():.1f} & " + " & ".join(f"{100*X[:, j].mean():.1f}" for j in range(1, 4)) + f" & {100*X.mean(0).max():.1f} & {100*X.max(1).mean():.1f} & {100*(X.max(1).mean()-X.mean(0).max()):.1f} & {100*(X.argmax(1) == 0).mean():.0f} & " + " & ".join(f"{h:.1f}/{hp:.1f}" for h, hp in zip(harm, helpr)) + " \\\\")
tex("interventions", "\\begin{tabular}{llrrrrrrrrccc}\\toprule & & & \\multicolumn{3}{c}{Fixed intervention} & & & & & \\multicolumn{3}{c}{Harm / help (\\% of items)} \\\\ \\cmidrule(lr){4-6}\\cmidrule(lr){11-13} Dataset & VLM & Base & Reprompt & Enhance & Zoom & Best fixed & Oracle & Gap & Oracle=KEEP \\% & R & E & Z \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
# selection-layer summary (Metric A, test, SL-Q emphasised) from earlier reports
R = pd.read_csv(RESULTS / "revised_selector_results.csv"); R = R[R.metric == "A"]; R["config"] = R.config.fillna(""); CS = pd.read_csv(RESULTS / "consensus_stage1.csv"); CS = CS[CS.metric == "A"]
def rr(meth, cfg=""):
    return [R[(R.dataset == d) & (R.model == m) & (R.method == meth) & (R.config == cfg)] for d, m in MAIN]
lines = []
for lab, meth, cfg in [("Base", "Base", ""), ("Best fixed (val)", "Best fixed", ""), ("Question-type router", "Question router", ""), ("Direct $\\Delta Q$ selector", "Direct dQ selector", "per_setting/biomedclip/iqy"), ("Gate + action classifier", "Gate + action classifier", "per_setting/biomedclip/iqy"), ("Gate + utility head", "Gate + utility head", "per_setting/clip/iqy")]:
    gs = rr(meth, cfg); lines.append(lab + " & " + " & ".join(f"{100*g.score.mean():.1f}" for g in gs) + " & " + " & ".join(f"{100*g.harmful_rate.mean():.1f}" if g.harmful_rate.notna().any() else "--" for g in gs) + " & " + " & ".join(f"{100*g.intervention_rate.mean():.1f}" for g in gs) + " \\\\")
for lab, pol in [("Majority vote", "Majority"), ("2-intervention corroboration", "2-intervention corroboration"), ("Gate + corroboration", "Gate + corroboration")]:
    gs = [CS[(CS.dataset == d) & (CS.model == m) & (CS.policy == pol)] for d, m in MAIN]; lines.append(lab + " & " + " & ".join(f"{100*g.score.mean():.1f}" for g in gs) + " & " + " & ".join(f"{100*g.harmful_correction_rate.mean():.1f}" for g in gs) + " & " + " & ".join(f"{100*g.change_rate.mean():.1f}" for g in gs) + " \\\\")
VV = pd.read_csv(RESULTS / "verification_validation_verifiers.csv")
lines.append("Anchored verification (val., any verifier) & " + " & ".join("= base" for _ in MAIN) + " & " + " & ".join("--" for _ in MAIN) + " & " + " & ".join(f"{100*VV[(VV.dataset == d) & (VV.model == m)].correct_rate.max():.1f}" for d, m in MAIN) + " \\\\")
gs = rr("Oracle"); lines.append("Oracle & " + " & ".join(f"{100*g.score.mean():.1f}" for g in gs) + " & 0 & 0 & 0 & 0 & " + " & ".join(f"{100*g.intervention_rate.mean():.1f}" for g in gs) + " \\\\")
tex("selectors", "\\begin{tabular}{lcccccccccccc}\\toprule & \\multicolumn{4}{c}{Score (Metric A)} & \\multicolumn{4}{c}{Harmful fraction of changes (\\%)} & \\multicolumn{4}{c}{Intervention / change rate (\\%)} \\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13} Method & VR-MG & VR-Q & SL-MG & SL-Q & VR-MG & VR-Q & SL-MG & SL-Q & VR-MG & VR-Q & SL-MG & SL-Q \\\\\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}")

# ------------------------------------------------------------------ 7. figures
COL = {"medgemma": "#1f77b4", "gemma3": "#aec7e8", "qwen": "#d62728", "qwen3b": "#ff9896", "qwen2vl2b": "#c49c94"}
# F1 AUROC bars with CIs
fig, ax = plt.subplots(figsize=(7.2, 2.6)); names = ["Log-prob", "Sampling consistency", "Perturbation consistency", "Image emb.", "Question+answer emb.", "Image+question+answer emb.", "All signals"]; cols = ["#999", "#bbb", "#777", "#e07b39", "#1f77b4", "#17becf", "#2ca02c"]
w = 0.115
for j, (nm, c) in enumerate(zip(names, cols)):
    v = [CI[(CI.dataset == d) & (CI.model == m) & (CI.predictor == nm)].iloc[0] for d, m in MAIN]
    ax.bar(np.arange(4) + (j - 3) * w, [x.auroc for x in v], w, yerr=[[x.auroc - x.lo for x in v], [x.hi - x.auroc for x in v]], capsize=1.5, color=c, label=nm)
ax.axhline(0.5, color="k", lw=0.6, ls="--"); ax.set_ylim(0.4, 1.0); ax.set_xticks(range(4)); ax.set_xticklabels(["VQA-RAD\nMedGemma-4B", "VQA-RAD\nQwen2.5-VL-7B", "SLAKE\nMedGemma-4B", "SLAKE\nQwen2.5-VL-7B"]); ax.set_ylabel("Error-detection AUROC (test)")
ax.legend(fontsize=6, ncol=4, loc="upper left", frameon=False); fig.savefig(P / "figures" / "fig_auroc.pdf", bbox_inches="tight"); plt.close(fig)
# F2 overlap heatmaps + #models wrong histogram
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for ax, d in zip(axes[:2], DSN):
    ph = phis[d]; im = ax.imshow(ph, vmin=0.2, vmax=1, cmap="Blues"); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels([NAME[m] for m in FIVE], rotation=35, ha="right", fontsize=6); ax.set_yticklabels([NAME[m] for m in FIVE], fontsize=6)
    for i in range(5):
        for j in range(5): ax.text(j, i, f"{ph[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if ph[i, j] > 0.65 else "k")
    ax.set_title(f"{DSN[d]}: error correlation $\\phi$", fontsize=8)
ax = axes[2]
for k, d in enumerate(DSN):
    r = OV[OV.dataset == d].iloc[0]; ax.bar(np.arange(6) + (k - 0.5) * 0.38, 100 * r.nwrong, 0.38, label=DSN[d], color=["#1f77b4", "#d62728"][k])
    pm = np.array([np.mean([m for m in [np.mean([keep(d, m, "test")[i]["score"] < 1 for i in keep(d, m, "test")]) for m in FIVE]])])
ax.set_xlabel("# of the 5 VLMs that are wrong"); ax.set_ylabel("% of test items"); ax.legend(fontsize=7, frameon=False); ax.set_title("Observed (independent: all-5 $\\approx$ 3\\%)", fontsize=8)
fig.tight_layout(); fig.savefig(P / "figures" / "fig_overlap.pdf", bbox_inches="tight"); plt.close(fig)
# F3 image sensitivity x correctness stacked bars
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
for ax, d in zip(axes, DSN):
    g = CF[CF.dataset == d].set_index("model").loc[FIVE]
    ins_c = g.same_swap * g.acc_if_insensitive; ins_w = g.same_swap * (1 - g.acc_if_insensitive); sen_c = (1 - g.same_swap) * g.acc_if_sensitive; sen_w = (1 - g.same_swap) * (1 - g.acc_if_sensitive)
    x = np.arange(5); b = np.zeros(5)
    for v, lab, c in [(ins_c, "image-insensitive, correct", "#1f77b4"), (ins_w, "image-insensitive, wrong", "#aec7e8"), (sen_c, "image-sensitive, correct", "#2ca02c"), (sen_w, "image-sensitive, wrong", "#98df8a")]:
        ax.bar(x, 100 * v.values, 0.6, bottom=b, color=c, label=lab); b += 100 * v.values
    ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in FIVE], rotation=25, ha="right", fontsize=6.5); ax.set_title(DSN[d], fontsize=8)
axes[0].set_ylabel("% of test items"); axes[0].legend(fontsize=6, frameon=False, loc="lower left"); fig.tight_layout(); fig.savefig(P / "figures" / "fig_counterfactual.pdf", bbox_inches="tight"); plt.close(fig)
# F4 risk-coverage curves with CI bands (q+a) + baselines
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=False)
for ax, (d, m) in zip(axes, MAIN):
    P_, yte, ids = preds_test[(d, m)]
    for nm, c in [("Log-prob", "#999"), ("Sampling consistency", "#bbb"), ("Perturbation consistency", "#777"), ("Question+answer emb.", "#1f77b4"), ("All signals", "#2ca02c")]:
        p = P_[nm]; o = np.argsort(p); covs = np.linspace(0.1, 1, 46); acc_ = [1 - yte[o[:int(cv * len(yte))]].mean() for cv in covs]
        ax.plot(100 * covs, 100 * np.array(acc_), color=c, lw=1.2, label=nm)
        if nm == "Question+answer emb.":
            idx = rng.randint(0, len(yte), (300, len(yte))); band = np.array([[1 - yte[i][np.argsort(p[i])[:int(cv * len(i))]].mean() for cv in covs] for i in idx])
            ax.fill_between(100 * covs, 100 * np.percentile(band, 2.5, 0), 100 * np.percentile(band, 97.5, 0), color=c, alpha=0.15)
    ax.axhline(100 * (1 - yte.mean()), color="k", lw=0.6, ls="--"); ax.set_title(f"{DSN[d]} / {NAME[m]}", fontsize=7); ax.set_xlabel("coverage (%)", fontsize=7); ax.invert_xaxis(); ax.tick_params(labelsize=6)
axes[0].set_ylabel("selective accuracy (%)", fontsize=7); axes[0].legend(fontsize=5, frameon=False); fig.tight_layout(); fig.savefig(P / "figures" / "fig_risk_coverage.pdf", bbox_inches="tight"); plt.close(fig)
# F5 per-type heatmap (AUROC) SLAKE & VQA-RAD
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
for ax, d in zip(axes, DSN):
    g = ADV[(ADV.dataset == d) & (~ADV.qtype.str.startswith("["))]; piv = g.pivot(index="qtype", columns="model", values="auroc")[FIVE]
    im = ax.imshow(piv.values, vmin=0.5, vmax=1, cmap="viridis", aspect="auto"); ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=6); ax.set_xticks(range(5)); ax.set_xticklabels([NAME[m] for m in FIVE], rotation=30, ha="right", fontsize=6)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="w" if v < 0.8 else "k")
    ax.set_title(f"{DSN[d]}: error-detection AUROC by question type", fontsize=7.5)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01); fig.savefig(P / "figures" / "fig_pertype.pdf", bbox_inches="tight"); plt.close(fig)
# F6 predicted-error probability distributions by outcome + reliability diagram (VR-MG, SL-Q)
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
for k, (d, m) in enumerate([("vqa_rad", "medgemma"), ("slake", "qwen")]):
    P_, yte, _ = preds_test[(d, m)]; p = P_["Question+answer emb."]
    ax = axes[2 * k]; ax.hist(p[yte == 0], bins=20, alpha=0.6, color="#2ca02c", label="answer correct", density=True); ax.hist(p[yte == 1], bins=20, alpha=0.6, color="#d62728", label="answer wrong", density=True); ax.set_title(f"{DSN[d]}/{NAME[m]}: predicted P(error)", fontsize=6.5); ax.legend(fontsize=5, frameon=False); ax.tick_params(labelsize=6)
    ax = axes[2 * k + 1]; edges = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (p > lo) & (p <= hi)
        if mk.sum() >= 5: xs.append(p[mk].mean()); ys.append(yte[mk].mean()); ns.append(mk.sum())
    ax.plot([0, 1], [0, 1], "k--", lw=0.6); ax.scatter(xs, ys, s=np.array(ns) / 3, color="#1f77b4"); ax.set_xlabel("predicted P(error)", fontsize=6); ax.set_ylabel("observed error rate", fontsize=6); ax.set_title("reliability", fontsize=6.5); ax.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(P / "figures" / "fig_calibration.pdf", bbox_inches="tight"); plt.close(fig)
# F7 interventions: oracle gap vs harm scatter for 5 models
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for d, mk in zip(DSN, ["o", "s"]):
    for m in FIVE:
        acts = {a: RESULTS / "raw" / f"{d}_{m}_test_{a}.jsonl" for a in ACTIONS}
        if not all(p.exists() for p in acts.values()): continue
        Sc = {a: {r["sample_id"]: r["score"] for r in read_jsonl(acts[a])} for a in ACTIONS}; ids = sorted(set.intersection(*[set(v) for v in Sc.values()])); X = np.array([[Sc[a][i] for a in ACTIONS] for i in ids])
        gap = 100 * (X.max(1).mean() - X.mean(0).max()); harm = 100 * np.mean([(X[:, j] < X[:, 0]).mean() for j in range(1, 4)])
        ax.scatter(harm, gap, marker=mk, color=COL[m], s=28, label=f"{NAME[m]} ({DSN[d]})")
ax.set_xlabel("mean harm of a single intervention (% items)"); ax.set_ylabel("oracle gap over best fixed (pts)"); ax.legend(fontsize=4.5, frameon=False, ncol=2); fig.savefig(P / "figures" / "fig_interventions.pdf", bbox_inches="tight"); plt.close(fig)
print("figures done")
