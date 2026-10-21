"""Stage 2 evaluation: baselines, selector, oracle, selector-specific metrics, bootstrap CIs,
question-type and base-correctness analyses, ablations, qualitative case selection.

Outputs: results/selector_results.csv, results/bootstrap_results.csv, results/ablation_results.csv,
         results/qualitative_cases.json, tables/*.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTIONS, ACTION_NAMES, RESULTS, ROOT, load_matrix  # noqa: E402

TABLES = ROOT / "tables"
DATASETS, MODELS = ["vqa_rad", "slake"], ["medgemma", "qwen"]
PRETTY = {"vqa_rad": "VQA-RAD", "slake": "SLAKE", "medgemma": "MedGemma", "qwen": "Qwen-VL"}
SEEDS = [0, 1, 2]
A = {a: i for i, a in enumerate(ACTIONS)}
rng = np.random.RandomState(0)


def S_of(g):
    return g[[f"{a}_score" for a in ACTIONS]].values


def realised(g, act_idx):
    return S_of(g)[np.arange(len(g)), act_idx]


# ------------------------------------------------------------------ baselines (all fitted on validation only)
def best_fixed_action(val):
    return max(ACTIONS, key=lambda a: val[f"{a}_score"].mean())


def question_router(val, test, fallback):
    """per question_type, the action with the best mean validation score (>=10 val samples)."""
    table = {}
    for qt, g in val.groupby("question_type"):
        if len(g) >= 10:
            table[qt] = max(ACTIONS, key=lambda a: g[f"{a}_score"].mean())
    return np.array([A[table.get(qt, fallback)] for qt in test.question_type]), table


def confidence_router(val, test, fixed):
    """keep if base confidence > tau else apply the best fixed action; tau tuned on validation."""
    cands = np.unique(np.quantile(val.base_conf.dropna(), np.linspace(0, 1, 41)))
    best_tau, best = None, -1
    for tau in cands:
        act = np.where(val.base_conf.values > tau, 0, A[fixed])
        r = realised(val, act).mean()
        if r > best:
            best, best_tau = r, tau
    return np.where(test.base_conf.values > best_tau, 0, A[fixed]), best_tau


def selector_actions(ds, m, feat, encoder, seed, test_ids):
    p = RESULTS / "selector_preds" / f"{ds}_{m}_{feat}_{encoder}_seed{seed}.csv"
    if not p.exists():
        return None, None
    d = pd.read_csv(p)
    d = d[d.split == "test"].set_index("sample_id").loc[test_ids]
    return np.array([A[a] for a in d.selected]), d[[f"pred_dq_{a}" for a in ACTIONS]].values


def metrics(g, act, oracle_act):
    S = S_of(g)
    n = np.arange(len(g))
    sel = S[n, act]
    base = g.base_score.values
    orc = g.best_score.values
    changed = act != 0
    out = dict(score=sel.mean(),
               action_acc=float((act == oracle_act).mean()),
               regret=float((orc - sel).mean()),
               harmful_rate=float((sel[changed] < base[changed]).mean()) if changed.any() else np.nan,
               success_rate=float((sel[changed] > base[changed]).mean()) if changed.any() else np.nan,
               intervene_rate=float(changed.mean()),
               p_keep_given_correct=float((act[base == 1] == 0).mean()) if (base == 1).any() else np.nan,
               p_intervene_given_incorrect=float((act[base < 1] != 0).mean()) if (base < 1).any() else np.nan,
               score_base_correct=float(sel[base == 1].mean()) if (base == 1).any() else np.nan,
               score_base_incorrect=float(sel[base < 1].mean()) if (base < 1).any() else np.nan)
    return out


def bootstrap(arrs, n=1000):
    """arrs: dict name -> per-sample score array (same length). returns mean, CI for each and for pairwise diffs vs best_fixed."""
    N = len(next(iter(arrs.values())))
    idx = rng.randint(0, N, size=(n, N))
    res = {}
    for k, v in arrs.items():
        b = v[idx].mean(1)
        res[k] = (v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5))
    if "best_fixed" in arrs:
        for k, v in arrs.items():
            if k == "best_fixed":
                continue
            d = (v - arrs["best_fixed"])[idx].mean(1)
            res[f"{k}-best_fixed"] = ((v - arrs["best_fixed"]).mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))
    return res


def main():
    M = load_matrix()
    rows, boots, abl, qtype_rows, qual = [], [], [], [], []
    main_table = {}
    for ds in DATASETS:
        for m in MODELS:
            val = M[(M.dataset == ds) & (M.model == m) & (M.split == "validation")].reset_index(drop=True)
            test = M[(M.dataset == ds) & (M.model == m) & (M.split == "test")].reset_index(drop=True)
            if val.empty or test.empty:
                continue
            key = f"{PRETTY[ds]} {PRETTY[m]}"
            oracle_act = np.array([A[a] for a in test.best_action])
            fixed = best_fixed_action(val)
            acts = {"Base": np.zeros(len(test), int),
                    "Best Fixed": np.full(len(test), A[fixed])}
            qr_act, qr_table = question_router(val, test, fixed)
            acts["Question Router"] = qr_act
            cr_act, cr_tau = confidence_router(val, test, fixed)
            acts["Confidence Router"] = cr_act
            per_sample = {"base": test.base_score.values, "best_fixed": realised(test, acts["Best Fixed"]),
                          "question_router": realised(test, qr_act), "confidence_router": realised(test, cr_act),
                          "oracle": test.best_score.values}
            # random: exact expectation over uniform action choice
            rand_score = S_of(test).mean(1)
            per_sample["random"] = rand_score
            for name, act in acts.items():
                mt = metrics(test, act, oracle_act)
                rows.append(dict(dataset=ds, model=m, method=name, seed=None, **mt))
            rows.append(dict(dataset=ds, model=m, method="Random", seed=None, score=rand_score.mean(),
                             action_acc=0.25, regret=float((test.best_score.values - rand_score).mean()), intervene_rate=0.75))
            rows.append(dict(dataset=ds, model=m, method="Oracle", seed=None, **metrics(test, oracle_act, oracle_act)))
            rows[-1]["fixed_action"] = fixed
            # ours + ablations
            for encoder in ["biomedclip", "clip"]:
                for feat in ["q", "iq", "iqy"]:
                    seed_scores, seed_acts = [], []
                    for seed in SEEDS:
                        act, pred = selector_actions(ds, m, feat, encoder, seed, test.sample_id.tolist())
                        if act is None:
                            continue
                        mt = metrics(test, act, oracle_act)
                        seed_scores.append(mt["score"]); seed_acts.append(act)
                        name = "Counterfactual Selector" if (feat == "iqy" and encoder == "biomedclip") else f"ablation:{encoder}:{feat}"
                        rows.append(dict(dataset=ds, model=m, method=name, seed=seed, **mt))
                        if feat == "iqy" and encoder == "biomedclip":
                            per_sample[f"ours_seed{seed}"] = realised(test, act)
                            if seed == 0:
                                test[f"pred_dq"] = list(map(list, pred)); test["selected"] = [ACTIONS[a] for a in act]
                    if seed_scores:
                        abl.append(dict(dataset=ds, model=m, encoder=encoder, feat=feat, mean=np.mean(seed_scores), std=np.std(seed_scores), n_seeds=len(seed_scores)))
            # bootstrap
            if "ours_seed0" in per_sample:
                per_sample["ours"] = np.mean([per_sample[f"ours_seed{s}"] for s in SEEDS if f"ours_seed{s}" in per_sample], 0)
            bt = bootstrap(per_sample)
            for k, (mu, lo, hi) in bt.items():
                boots.append(dict(dataset=ds, model=m, quantity=k, mean=mu, ci_low=lo, ci_high=hi))
            main_table[key] = {k: v[0] for k, v in bt.items()}
            main_table[key]["fixed_action"] = fixed
            main_table[key]["conf_tau"] = cr_tau
            main_table[key]["question_router_table"] = qr_table
            # per question type
            if "selected" in test:
                sel_idx = np.array([A[a] for a in test.selected])
                for qt, g in test.groupby("question_type"):
                    ii = g.index.values
                    qtype_rows.append(dict(dataset=ds, model=m, question_type=qt, n=len(g),
                                           base=g.base_score.mean(), best_fixed=realised(test, acts["Best Fixed"])[ii].mean(),
                                           ours=realised(test, sel_idx)[ii].mean(), oracle=g.best_score.mean(),
                                           ours_keep_rate=float((sel_idx[ii] == 0).mean())))
                for at, g in test.groupby("answer_type"):
                    ii = g.index.values
                    qtype_rows.append(dict(dataset=ds, model=m, question_type=f"[{at}]", n=len(g),
                                           base=g.base_score.mean(), best_fixed=realised(test, acts["Best Fixed"])[ii].mean(),
                                           ours=realised(test, sel_idx)[ii].mean(), oracle=g.best_score.mean(),
                                           ours_keep_rate=float((sel_idx[ii] == 0).mean())))
                # qualitative cases
                def pick(mask, tag, k=1):
                    c = test[mask]
                    for _, r in c.head(k).iterrows():
                        qual.append(dict(case=tag, dataset=ds, model=m, sample_id=r.sample_id, image_path=r.image_path,
                                         question=r.question, ground_truth=r.reference_answer, base_answer=r.base_answer,
                                         answers={a: r[f"{a}_answer"] for a in ACTIONS}, true_dq={a: r[f"dq_{a}"] for a in ACTIONS},
                                         pred_dq=dict(zip(ACTIONS, r.pred_dq)), selected=r.selected, oracle=r.best_action,
                                         selected_score=float(r[f"{r.selected}_score"]), base_score=float(r.base_score)))
                sel_score = realised(test, sel_idx)
                for a in ["zoom", "reprompt", "enhance"]:
                    others = [b for b in ACTIONS[1:] if b != a]
                    mask = (test.best_action == a) & (test[f"{a}_score"] == 1) & (test.base_score < 1) & np.logical_and.reduce([test[f"{b}_score"] < 1 for b in others]) & (test.selected == a)
                    pick(mask, f"{ACTION_NAMES[a]} is best (and selected)")
                    if not mask.any():
                        pick((test.best_action == a) & (test.base_score < 1), f"{ACTION_NAMES[a]} is best")
                pick((test.base_score == 1) & (test.selected == "keep") & np.logical_or.reduce([test[f"{b}_score"] < 1 for b in ACTIONS[1:]]), "KEEP is best (and selected)")
                pick((sel_score < test.base_score), "Selector failure (intervention hurt)")
                pick((test.base_score == 0) & (test.best_score == 0), "Every intervention fails")

    R = pd.DataFrame(rows); R.to_csv(RESULTS / "selector_results.csv", index=False)
    B = pd.DataFrame(boots); B.to_csv(RESULTS / "bootstrap_results.csv", index=False)
    pd.DataFrame(abl).to_csv(RESULTS / "ablation_results.csv", index=False)
    pd.DataFrame(qtype_rows).to_csv(RESULTS / "question_type_results.csv", index=False)
    json.dump(qual, open(RESULTS / "qualitative_cases.json", "w"), indent=1, default=str)
    json.dump(main_table, open(RESULTS / "main_table_numbers.json", "w"), indent=1, default=str)

    # ---------------- tables
    cols = list(main_table)
    order = [("Base", "base"), ("Random action", "random"), ("Best fixed", "best_fixed"), ("Question-type router", "question_router"),
             ("Confidence router", "confidence_router"), ("**Counterfactual selector (ours)**", "ours"), ("Oracle", "oracle")]
    t = ["| Method | " + " | ".join(cols) + " | Avg. |", "|---|" + "---:|" * (len(cols) + 1)]
    for name, k in order:
        vals = [main_table[c].get(k, np.nan) for c in cols]
        t.append(f"| {name} | " + " | ".join(f"{100*v:.1f}" for v in vals) + f" | {100*np.nanmean(vals):.1f} |")
    ours_std = R[R.method == "Counterfactual Selector"].groupby(["dataset", "model"]).score.std()
    t.append("\nSelector row = mean over seeds 0,1,2; per-seed std: " + ", ".join(f"{PRETTY[d]} {PRETTY[m]}: {100*s:.2f}" for (d, m), s in ours_std.items()))
    t.append("\nBest fixed action chosen on validation: " + ", ".join(f"{c}: {ACTION_NAMES[main_table[c]['fixed_action']]}" for c in cols))
    (TABLES / "stage2_main_table.md").write_text("\n".join(t) + "\n")
    print("\n".join(t))

    # selector-specific metrics
    t = ["| Dataset / Model | Action-selection acc. | Mean regret | Headroom recovery | Harmful intervention rate | Successful intervention rate | Intervention rate | P(KEEP | base correct) | P(intervene | base wrong) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cols:
        ds, m = [k for k in DATASETS if PRETTY[k] == c.split()[0]][0], [k for k in MODELS if PRETTY[k] == c.split()[1]][0]
        g = R[(R.dataset == ds) & (R.model == m) & (R.method == "Counterfactual Selector")]
        if g.empty:
            continue
        mt = main_table[c]
        rec = (mt["ours"] - mt["best_fixed"]) / (mt["oracle"] - mt["best_fixed"]) if mt["oracle"] > mt["best_fixed"] else np.nan
        t.append(f"| {c} | {100*g.action_acc.mean():.1f} | {100*g.regret.mean():.2f} | {100*rec:.1f}% | {100*g.harmful_rate.mean():.1f} | {100*g.success_rate.mean():.1f} | {100*g.intervene_rate.mean():.1f} | {100*g.p_keep_given_correct.mean():.1f} | {100*g.p_intervene_given_incorrect.mean():.1f} |")
    t.append("\nSame metrics for the baselines:")
    t.append("| Dataset / Model | Method | Action acc. | Regret | Harmful rate | Success rate | Intervention rate |")
    t.append("|---|---|---:|---:|---:|---:|---:|")
    for c in cols:
        ds, m = [k for k in DATASETS if PRETTY[k] == c.split()[0]][0], [k for k in MODELS if PRETTY[k] == c.split()[1]][0]
        for meth in ["Best Fixed", "Question Router", "Confidence Router", "Oracle"]:
            g = R[(R.dataset == ds) & (R.model == m) & (R.method == meth)]
            if g.empty:
                continue
            g = g.iloc[0]
            t.append(f"| {c} | {meth} | {100*g.action_acc:.1f} | {100*g.regret:.2f} | {100*g.harmful_rate:.1f} | {100*g.success_rate:.1f} | {100*g.intervene_rate:.1f} |")
    (TABLES / "stage2_selector_metrics.md").write_text("\n".join(t) + "\n")
    print("\n".join(t))

    # bootstrap table
    t = ["| Dataset / Model | Quantity | Mean | 95% CI |", "|---|---|---:|---:|"]
    for _, r in B.iterrows():
        t.append(f"| {PRETTY[r.dataset]} {PRETTY[r.model]} | {r.quantity} | {100*r['mean']:.2f} | [{100*r.ci_low:.2f}, {100*r.ci_high:.2f}] |")
    (TABLES / "stage2_bootstrap.md").write_text("\n".join(t) + "\n")

    # ablation table
    Ab = pd.DataFrame(abl)
    if not Ab.empty:
        t = ["| Encoder | Features | " + " | ".join(cols) + " | Avg. |", "|---|---|" + "---:|" * (len(cols) + 1)]
        for enc in ["biomedclip", "clip"]:
            for feat, lab in [("q", "question only (A1)"), ("iq", "image + question (A2)"), ("iqy", "image + question + initial answer (A3)")]:
                vals = []
                for c in cols:
                    ds, m = [k for k in DATASETS if PRETTY[k] == c.split()[0]][0], [k for k in MODELS if PRETTY[k] == c.split()[1]][0]
                    g = Ab[(Ab.dataset == ds) & (Ab.model == m) & (Ab.encoder == enc) & (Ab.feat == feat)]
                    vals.append((g["mean"].iloc[0], g["std"].iloc[0]) if len(g) else (np.nan, np.nan))
                if all(np.isnan(v[0]) for v in vals):
                    continue
                t.append(f"| {enc} | {lab} | " + " | ".join(f"{100*mu:.1f} ± {100*sd:.1f}" for mu, sd in vals) + f" | {100*np.nanmean([v[0] for v in vals]):.1f} |")
        (TABLES / "stage2_ablations.md").write_text("\n".join(t) + "\n")
        print("\n".join(t))

    # question-type table
    Q = pd.DataFrame(qtype_rows)
    if not Q.empty:
        t = ["| Dataset / Model | Question type | n | Base | Best fixed | Ours (seed 0) | Oracle | Ours KEEP rate |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in Q.iterrows():
            t.append(f"| {PRETTY[r.dataset]} {PRETTY[r.model]} | {r.question_type} | {r.n} | {100*r.base:.1f} | {100*r.best_fixed:.1f} | {100*r.ours:.1f} | {100*r.oracle:.1f} | {100*r.ours_keep_rate:.0f} |")
        (TABLES / "stage2_question_types.md").write_text("\n".join(t) + "\n")


if __name__ == "__main__":
    main()
