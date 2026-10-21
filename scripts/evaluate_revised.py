"""Evaluation of the revised (gated) selectors against all baselines, under Metric A (original) and Metric B (canonicalised).

Primary gated configuration (regime, encoder, features) is chosen by mean VALIDATION score over the four settings.
Outputs (never overwriting the original results):
  results/revised_selector_results.csv, revised_gate_metrics.csv, revised_bootstrap.csv, revised_ablations.csv,
  revised_semantic_eval.csv, revised_error_taxonomy.csv, revised_repairs.csv, manual_review_candidates.csv
  tables/revised_*.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonicalize  # noqa: E402
from common import ACTIONS, ACTION_NAMES, RESULTS, ROOT, normalize_answer  # noqa: E402

TABLES = ROOT / "tables"
SETTINGS = [(d, m) for d in ["vqa_rad", "slake"] for m in ["medgemma", "qwen"]]
PRETTY = {"vqa_rad": "VQA-RAD", "slake": "SLAKE", "medgemma": "MG", "qwen": "Qwen"}
A = {a: i for i, a in enumerate(ACTIONS)}
SEEDS = [0, 1, 2]
rng = np.random.RandomState(0)


def load_all():
    M = pd.read_csv(RESULTS / "intervention_matrix_canonical.csv", keep_default_na=False)
    for c in M.columns:
        if c.endswith(("_score", "_score_b", "_conf", "_improvement")) or c.startswith("dq_"):
            M[c] = pd.to_numeric(M[c], errors="coerce")
    return M


def S_of(g, metric):
    suf = "" if metric == "A" else "_b"
    return g[[f"{a}_score{suf}" for a in ACTIONS]].values


def realised(S, a):
    return S[np.arange(len(a)), a]


def stats(S, a, oracle_a):
    r = realised(S, a); base = S[:, 0]; orc = S.max(1); ch = a != 0
    return dict(score=r.mean(), intervention_rate=ch.mean(),
                harmful_rate=float((r[ch] < base[ch]).mean()) if ch.any() else 0.0,
                successful_rate=float((r[ch] > base[ch]).mean()) if ch.any() else 0.0,
                regret=float((orc - r).mean()), action_acc=float((a == oracle_a).mean()),
                false_intervention=float((a[oracle_a == 0] != 0).mean()) if (oracle_a == 0).any() else np.nan,
                missed_opportunity=float((a[oracle_a != 0] == 0).mean()) if (oracle_a != 0).any() else np.nan)


# ------------------------------------------------------------------ baselines fitted on validation only
def best_fixed(Sv):
    return int(np.argmax(Sv.mean(0)))


def question_router(val, test, Sv, fixed):
    tab = {}
    for qt, g in val.groupby("question_type"):
        if len(g) >= 10:
            tab[qt] = int(np.argmax(Sv[g.index.values].mean(0)))
    return np.array([tab.get(q, fixed) for q in test.question_type])


def confidence_router(val, test, Sv, fixed):
    cands = np.unique(np.quantile(val.base_conf.dropna(), np.linspace(0, 1, 41)))
    best = (-1, None)
    for t in cands:
        r = realised(Sv, np.where(val.base_conf.values > t, 0, fixed)).mean()
        if r > best[0]:
            best = (r, t)
    return np.where(test.base_conf.values > best[1], 0, fixed)


def direct_actions(ds, m, seed, ids):
    p = RESULTS / "selector_preds" / f"{ds}_{m}_iqy_biomedclip_seed{seed}.csv"
    d = pd.read_csv(p); d = d[d.split == "test"].set_index("sample_id").loc[ids]
    return np.array([A[a] for a in d.selected])


def gated_preds(regime, enc, feat, seed):
    p = RESULTS / "revised_preds" / f"{regime}_{enc}_{feat}_seed{seed}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, keep_default_na=False)
    for c in ["eps", "tau", "gate_prob"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def ece(p, y, bins=10):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def bootstrap_diff(x, y, n=1000):
    idx = rng.randint(0, len(x), (n, len(x)))
    d = (x - y)[idx].mean(1)
    return float((x - y).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    M = load_all()
    configs = [(r, e, f) for r in ["per_setting", "pooled"] for e, f in [("biomedclip", "q"), ("biomedclip", "iq"), ("biomedclip", "iqy"), ("clip", "iqy")]]
    data = {}
    for ds, m in SETTINGS:
        va = M[(M.dataset == ds) & (M.model == m) & (M.split == "validation")].reset_index(drop=True)
        te = M[(M.dataset == ds) & (M.model == m) & (M.split == "test")].reset_index(drop=True)
        data[(ds, m)] = (va, te)

    # ---------------- pick primary gated config on validation (mean over settings, metric A, per head)
    val_tab = {}
    for (r, e, f) in configs:
        for head in ["cls", "util"]:
            vals = []
            for seed in SEEDS:
                P = gated_preds(r, e, f, seed)
                if P is None:
                    continue
                for ds, m in SETTINGS:
                    va = data[(ds, m)][0]
                    q = P[(P.dataset == ds) & (P.model == m) & (P.split == "validation") & (P["head"] == head)].set_index("sample_id").loc[va.sample_id]
                    vals.append(realised(S_of(va, "A"), np.array([A[a] for a in q.selected])).mean())
            if vals:
                val_tab[(r, e, f, head)] = np.mean(vals)
    primary = {h: max([k for k in val_tab if k[3] == h], key=lambda k: val_tab[k]) for h in ["cls", "util"]}
    print("validation-selected primary configs:", primary, {k: round(v, 4) for k, v in val_tab.items()})

    results, gate_rows, boots, semantic, taxonomy, repairs, cand = [], [], [], [], [], [], []
    per_sample = {}
    for ds, m in SETTINGS:
        va, te = data[(ds, m)]
        key = f"{PRETTY[ds]} {PRETTY[m]}"
        for metric in ["A", "B"]:
            Sv, St = S_of(va, metric), S_of(te, metric)
            oracle_a = St.argmax(1)
            fixed = best_fixed(Sv)
            acts = {"Base": np.zeros(len(te), int), "Best fixed": np.full(len(te), fixed),
                    "Question router": question_router(va, te, Sv, fixed), "Confidence router": confidence_router(va, te, Sv, fixed)}
            per_sample[(ds, m, metric, "base")] = St[:, 0]
            per_sample[(ds, m, metric, "best_fixed")] = realised(St, acts["Best fixed"])
            per_sample[(ds, m, metric, "question_router")] = realised(St, acts["Question router"])
            for name, a in acts.items():
                results.append(dict(dataset=ds, model=m, metric=metric, method=name, config="", seed=None, fixed_action=ACTIONS[fixed] if name == "Best fixed" else "", **stats(St, a, oracle_a)))
            results.append(dict(dataset=ds, model=m, metric=metric, method="Random action", config="", seed=None, score=St.mean(1).mean(), intervention_rate=0.75,
                                regret=float((St.max(1) - St.mean(1)).mean()), action_acc=0.25))
            results.append(dict(dataset=ds, model=m, metric=metric, method="Oracle", config="", seed=None, **stats(St, oracle_a, oracle_a)))
            # direct dQ selector (original)
            dsc = []
            for seed in SEEDS:
                a = direct_actions(ds, m, seed, te.sample_id.tolist())
                results.append(dict(dataset=ds, model=m, metric=metric, method="Direct dQ selector", config="per_setting/biomedclip/iqy", seed=seed, **stats(St, a, oracle_a)))
                dsc.append(realised(St, a))
            per_sample[(ds, m, metric, "direct")] = np.mean(dsc, 0)
            # gated selectors, all configs
            for (r, e, f) in configs:
                for head in ["cls", "util"]:
                    name = "Gate + action classifier" if head == "cls" else "Gate + utility head"
                    sc = []
                    for seed in SEEDS:
                        P = gated_preds(r, e, f, seed)
                        if P is None:
                            continue
                        q = P[(P.dataset == ds) & (P.model == m) & (P.split == "test") & (P["head"] == head)].set_index("sample_id").loc[te.sample_id]
                        a = np.array([A[x] for x in q.selected])
                        results.append(dict(dataset=ds, model=m, metric=metric, method=name, config=f"{r}/{e}/{f}", seed=seed, eps=q.eps.iloc[0], tau=q.tau.iloc[0], **stats(St, a, oracle_a)))
                        sc.append(realised(St, a))
                        if (r, e, f, head) == primary[head] and metric == "A":
                            te[f"sel_{head}_s{seed}"] = a
                    if sc:
                        per_sample[(ds, m, metric, f"{head}|{r}/{e}/{f}")] = np.mean(sc, 0)
            # bootstrap for primary configs
            for head in ["cls", "util"]:
                r, e, f, _ = primary[head]
                ours = per_sample.get((ds, m, metric, f"{head}|{r}/{e}/{f}"))
                if ours is None:
                    continue
                for other in ["base", "best_fixed", "question_router", "direct"]:
                    mu, lo, hi = bootstrap_diff(ours, per_sample[(ds, m, metric, other)])
                    boots.append(dict(dataset=ds, model=m, metric=metric, head=head, comparison=f"gated-{other}", mean=mu, ci_low=lo, ci_high=hi, significant=(lo > 0) or (hi < 0)))
                oracle = St.max(1)
                idx = rng.randint(0, len(te), (1000, len(te)))
                for nm, arr in [("gated", ours), ("base", St[:, 0]), ("best_fixed", per_sample[(ds, m, metric, "best_fixed")]), ("question_router", per_sample[(ds, m, metric, "question_router")]),
                                ("direct", per_sample[(ds, m, metric, "direct")]), ("oracle", oracle)]:
                    b = arr[idx].mean(1)
                    boots.append(dict(dataset=ds, model=m, metric=metric, head=head, comparison=f"abs-{nm}", mean=arr.mean(), ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5), significant=np.nan))
            # error taxonomy (metric-specific)
            base_ok = St[:, 0] == 1
            hurt = (St[:, 1:] < St[:, :1]).any(1); help_ = (St[:, 1:] > St[:, :1]).any(1)
            taxonomy.append(dict(dataset=ds, model=m, metric=metric, n=len(te),
                                 A_base_correct_no_harm=int((base_ok & ~hurt).sum()), D_base_correct_some_harm=int((base_ok & hurt).sum()),
                                 B_base_wrong_repairable=int((~base_ok & help_).sum()), C_base_wrong_unrepairable=int((~base_ok & ~help_).sum())))

        # ---------------- gate quality (eps = 0 gate of every config, metric A labels) + decision-level gate metrics of the primary
        St = S_of(te, "A"); b0 = (St[:, 1:].max(1) > St[:, 0]).astype(int)
        for (r, e, f) in configs:
            for seed in SEEDS:
                P = gated_preds(r, e, f, seed)
                if P is None:
                    continue
                g = P[(P.dataset == ds) & (P.model == m) & (P.split == "test") & (P["head"] == "gate_eps0")].set_index("sample_id").loc[te.sample_id]
                p = g.gate_prob.values.astype(float)
                rec = dict(dataset=ds, model=m, config=f"{r}/{e}/{f}", seed=seed, pos_rate=b0.mean(), auroc=roc_auc_score(b0, p), auprc=average_precision_score(b0, p), ece=ece(p, b0))
                for head in ["cls", "util"]:
                    q = P[(P.dataset == ds) & (P.model == m) & (P.split == "test") & (P["head"] == head)].set_index("sample_id").loc[te.sample_id]
                    pred = (q.selected.values != "keep").astype(int)
                    tp = ((pred == 1) & (b0 == 1)).sum(); fn = ((pred == 0) & (b0 == 1)).sum(); fp = ((pred == 1) & (b0 == 0)).sum(); tn = ((pred == 0) & (b0 == 0)).sum()
                    rec_, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
                    rec.update({f"{head}_tau": q.tau.iloc[0], f"{head}_eps": q.eps.iloc[0], f"{head}_recall": rec_, f"{head}_specificity": spec, f"{head}_precision": prec,
                                f"{head}_f1": 2 * prec * rec_ / max(prec + rec_, 1e-9), f"{head}_balanced_acc": (rec_ + spec) / 2,
                                f"{head}_false_intervention": fp / max(fp + tn, 1), f"{head}_missed_opportunity": fn / max(fn + tp, 1), f"{head}_pred_rate": pred.mean()})
                gate_rows.append(rec)

        # ---------------- semantic-eval (seed 0) + repair analysis (all seeds, deduplicated) on the primary classifier head
        r, e, f, _ = primary["cls"]
        SA, SB = S_of(te, "A"), S_of(te, "B")
        for seed in SEEDS:
            col = f"sel_cls_s{seed}"
            if col not in te:
                continue
            a = te[col].values
            rA, rB = realised(SA, a), realised(SB, a)
            succA, harmA = rA > SA[:, 0], rA < SA[:, 0]
            lex_only = np.array([canonicalize(te.loc[i, "base_answer"]) == canonicalize(te.loc[i, f"{ACTIONS[a[i]]}_answer"]) for i in range(len(te))])
            if seed == 0:
                semantic.append(dict(dataset=ds, model=m, n=len(te), oracle_action_changed_A_vs_B=int((SA.argmax(1) != SB.argmax(1)).sum()),
                                     gated_successes_A=int(succA.sum()), gated_successes_lexical_only=int((succA & lex_only).sum()), gated_successes_not_success_under_B=int((succA & ~(rB > SB[:, 0])).sum()),
                                     gated_harms_A=int(harmA.sum()), gated_harms_semantically_equivalent=int((harmA & (rB >= SB[:, 0])).sum()),
                                     decisions_changed_A_vs_B_oracle=int((SA.argmax(1) != SB.argmax(1)).sum()),
                                     oracle_gap_A=float((SA.max(1) - SA[:, best_fixed(S_of(va, "A"))]).mean()), oracle_gap_B=float((SB.max(1) - SB[:, best_fixed(S_of(va, "B"))]).mean())))
            for i in np.where(succA | harmA)[0]:
                act = ACTIONS[a[i]]; row = te.loc[i]
                if lex_only[i]:
                    rt = "lexical_only"
                elif act == "reprompt":
                    rt = "prompt_format"
                elif row.answer_type == "CLOSED":
                    rt = "visual_presence_flip"
                else:
                    rt = "visual_localization_or_attribute"
                repairs.append(dict(sample_id=row.sample_id, dataset=ds, vlm=m, seed=seed, outcome="success" if succA[i] else "harm", question=row.question, reference=row.reference_answer,
                                    base_answer=row.base_answer, selected_action=act, selected_answer=row[f"{act}_answer"], base_score=float(SA[i, 0]), selected_score=float(rA[i]),
                                    base_score_b=float(SB[i, 0]), selected_score_b=float(rB[i]), repair_type=rt, lexical_only=bool(lex_only[i]), question_type=row.question_type,
                                    image_path=row.image_path))

    R = pd.DataFrame(results); R.to_csv(RESULTS / "revised_selector_results.csv", index=False)
    G = pd.DataFrame(gate_rows); G.to_csv(RESULTS / "revised_gate_metrics.csv", index=False)
    B = pd.DataFrame(boots); B.to_csv(RESULTS / "revised_bootstrap.csv", index=False)
    pd.DataFrame(semantic).to_csv(RESULTS / "revised_semantic_eval.csv", index=False)
    pd.DataFrame(taxonomy).to_csv(RESULTS / "revised_error_taxonomy.csv", index=False)
    Rp = pd.DataFrame(repairs); Rp.to_csv(RESULTS / "revised_repairs.csv", index=False)
    json.dump({h: list(primary[h]) for h in primary}, open(RESULTS / "revised_primary_config.json", "w"))

    # manual-review candidates: 50 successes + 50 harms, spread over settings (seed 0, primary classifier head)
    if not Rp.empty:
        parts = []
        for outcome in ["success", "harm"]:
            sub = Rp[Rp.outcome == outcome].sort_values("seed").drop_duplicates(["sample_id", "vlm", "selected_action"])
            # round-robin over settings so every setting is represented, seed-0 decisions first
            groups = [g.sample(frac=1, random_state=0).sort_values("seed").to_dict("records") for _, g in sub.groupby(["dataset", "vlm"])]
            picked = []
            while len(picked) < 50 and any(groups):
                for g in groups:
                    if g and len(picked) < 50:
                        picked.append(g.pop(0))
            parts.append(pd.DataFrame(picked))
        pd.concat(parts).to_csv(RESULTS / "manual_review_candidates.csv", index=False)

    # ---------------- ablations table (score A + gate AUROC/AUPRC per config, mean ± std over seeds)
    ab = []
    for (r, e, f) in configs:
        for head in ["cls", "util"]:
            for ds, m in SETTINGS:
                g = R[(R.dataset == ds) & (R.model == m) & (R.metric == "A") & (R.config == f"{r}/{e}/{f}") & (R.method == ("Gate + action classifier" if head == "cls" else "Gate + utility head"))]
                gg = G[(G.dataset == ds) & (G.model == m) & (G.config == f"{r}/{e}/{f}")]
                if g.empty:
                    continue
                ab.append(dict(regime=r, encoder=e, feat=f, head=head, dataset=ds, model=m, score_mean=g.score.mean(), score_std=g.score.std(), harm_mean=g.harmful_rate.mean(), rate_mean=g.intervention_rate.mean(),
                               auroc=gg.auroc.mean(), auprc=gg.auprc.mean(), val_score=val_tab.get((r, e, f, head), np.nan)))
    Ab = pd.DataFrame(ab); Ab.to_csv(RESULTS / "revised_ablations.csv", index=False)
    write_tables(R, G, B, Ab, pd.DataFrame(semantic), pd.DataFrame(taxonomy), primary, val_tab)


def fmt(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{d}f}"


def write_tables(R, G, B, Ab, Sem, Tax, primary, val_tab):
    cols = [f"{PRETTY[d]} {PRETTY[m]}" for d, m in SETTINGS]
    out = []
    pc, pu = primary["cls"], primary["util"]
    out.append(f"Primary configs (validation-selected): classifier = {pc[0]}/{pc[1]}/{pc[2]} (mean val {100*val_tab[pc]:.2f}), utility = {pu[0]}/{pu[1]}/{pu[2]} (mean val {100*val_tab[pu]:.2f})\n")
    for metric in ["A", "B"]:
        out.append(f"\n### Main table — Metric {metric} ({'original' if metric == 'A' else 'canonicalised'}), test, mean answer score ×100\n")
        out.append("| Method | " + " | ".join(cols) + " | Average |"); out.append("|---|" + "---:|" * (len(cols) + 1))
        rows = [("Base", "Base", ""), ("Random action", "Random action", ""), ("Best fixed", "Best fixed", ""), ("Question router", "Question router", ""), ("Confidence router", "Confidence router", ""),
                ("Direct ΔQ selector", "Direct dQ selector", "per_setting/biomedclip/iqy"), ("**Gate + action classifier**", "Gate + action classifier", f"{pc[0]}/{pc[1]}/{pc[2]}"),
                ("**Gate + utility head**", "Gate + utility head", f"{pu[0]}/{pu[1]}/{pu[2]}"), ("Oracle", "Oracle", "")]
        for label, meth, cfg in rows:
            vals = []
            for d, m in SETTINGS:
                g = R[(R.dataset == d) & (R.model == m) & (R.metric == metric) & (R.method == meth) & (R.config == cfg)]
                vals.append(g.score.mean() if len(g) else np.nan)
            extra = ""
            if meth == "Best fixed":
                extra = " (" + ", ".join(R[(R.dataset == d) & (R.model == m) & (R.metric == metric) & (R.method == meth)].fixed_action.iloc[0].upper() for d, m in SETTINGS) + ")"
            out.append(f"| {label}{extra} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(np.nanmean(vals))} |")
    # harm/benefit
    out.append("\n### Harm / benefit (Metric A, test; gated rows = mean over seeds)\n")
    out.append("| Setting | Method | Score | Intervention rate | Successful | Harmful | Regret | Headroom recovery | False intervention | Missed opportunity |"); out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d, m in SETTINGS:
        g = R[(R.dataset == d) & (R.model == m) & (R.metric == "A")]
        bf = g[g.method == "Best fixed"].score.iloc[0]; orc = g[g.method == "Oracle"].score.iloc[0]
        for label, meth, cfg in [("Base", "Base", ""), ("Best fixed", "Best fixed", ""), ("Question router", "Question router", ""), ("Confidence router", "Confidence router", ""),
                                 ("Direct ΔQ", "Direct dQ selector", "per_setting/biomedclip/iqy"), ("Gate+classifier", "Gate + action classifier", f"{pc[0]}/{pc[1]}/{pc[2]}"),
                                 ("Gate+utility", "Gate + utility head", f"{pu[0]}/{pu[1]}/{pu[2]}"), ("Oracle", "Oracle", "")]:
            h = g[(g.method == meth) & (g.config == cfg)]
            if h.empty:
                continue
            rec = (h.score.mean() - bf) / (orc - bf) if orc > bf else np.nan
            out.append(f"| {PRETTY[d]} {PRETTY[m]} | {label} | {fmt(h.score.mean())} | {fmt(h.intervention_rate.mean())} | {fmt(h.successful_rate.mean())} | {fmt(h.harmful_rate.mean())} | {fmt(h.regret.mean(), 2)} | {fmt(rec)} | {fmt(h.false_intervention.mean())} | {fmt(h.missed_opportunity.mean())} |")
    # bootstrap
    out.append("\n### Paired bootstrap (1000 resamples), gated − other, ×100 [95% CI]; * = CI excludes 0\n")
    out.append("| Setting | Metric | Head | − Base | − Best fixed | − Question router | − Direct ΔQ |"); out.append("|---|---|---|---:|---:|---:|---:|")
    for d, m in SETTINGS:
        for metric in ["A", "B"]:
            for head in ["cls", "util"]:
                g = B[(B.dataset == d) & (B.model == m) & (B.metric == metric) & (B["head"] == head)]
                cells = []
                for other in ["base", "best_fixed", "question_router", "direct"]:
                    r = g[g.comparison == f"gated-{other}"]
                    cells.append(f"{100*r['mean'].iloc[0]:+.2f} [{100*r.ci_low.iloc[0]:+.2f}, {100*r.ci_high.iloc[0]:+.2f}]{'*' if r.significant.iloc[0] else ''}" if len(r) else "n/a")
                out.append(f"| {PRETTY[d]} {PRETTY[m]} | {metric} | {'classifier' if head == 'cls' else 'utility'} | " + " | ".join(cells) + " |")
    # gate metrics
    out.append("\n### Gate quality (eps=0 gate, test labels = 'some intervention strictly improves', metric A; mean over seeds) and decision-level rates of the classifier head\n")
    out.append("| Setting | Config | Positive rate | AUROC | AUPRC | ECE | Recall | Specificity | Precision | F1 | Balanced acc | Predicted intervention rate | False intervention | Missed opportunity |"); out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cfg in sorted(G.config.unique()):
        for d, m in SETTINGS:
            g = G[(G.dataset == d) & (G.model == m) & (G.config == cfg)]
            if g.empty:
                continue
            out.append(f"| {PRETTY[d]} {PRETTY[m]} | {cfg} | {fmt(g.pos_rate.mean())} | {g.auroc.mean():.3f} | {g.auprc.mean():.3f} | {g.ece.mean():.3f} | {fmt(g.cls_recall.mean())} | {fmt(g.cls_specificity.mean())} | {fmt(g.cls_precision.mean())} | {fmt(g.cls_f1.mean())} | {fmt(g.cls_balanced_acc.mean())} | {fmt(g.cls_pred_rate.mean())} | {fmt(g.cls_false_intervention.mean())} | {fmt(g.cls_missed_opportunity.mean())} |")
    # ablations
    out.append("\n### Ablations (Metric A test score mean ± std over seeds; gate AUROC / AUPRC; mean validation score used for selection)\n")
    out.append("| Regime | Encoder | Features | Head | " + " | ".join(cols) + " | Avg test | Avg val | Avg harm | Avg rate | Avg AUROC | Avg AUPRC |"); out.append("|---|---|---|---|" + "---:|" * (len(cols) + 6))
    for (r, e, f, head), g in Ab.groupby(["regime", "encoder", "feat", "head"], sort=False):
        cells = []
        for d, m in SETTINGS:
            h = g[(g.dataset == d) & (g.model == m)]
            cells.append(f"{100*h.score_mean.iloc[0]:.1f} ± {100*h.score_std.iloc[0]:.1f}" if len(h) else "n/a")
        out.append(f"| {r} | {e} | {f} | {head} | " + " | ".join(cells) + f" | {fmt(g.score_mean.mean())} | {fmt(g.val_score.mean())} | {fmt(g.harm_mean.mean())} | {fmt(g.rate_mean.mean())} | {g.auroc.mean():.3f} | {g.auprc.mean():.3f} |")
    # taxonomy + semantic
    out.append("\n### Error taxonomy (test)\n")
    out.append("| Setting | Metric | n | A: base correct, no intervention harms | D: base correct, ≥1 intervention harms | B: base wrong, repairable | C: base wrong, unrepairable |"); out.append("|---|---|---:|---:|---:|---:|---:|")
    for _, t in Tax.iterrows():
        n = t.n
        out.append(f"| {PRETTY[t.dataset]} {PRETTY[t.model]} | {t.metric} | {n} | {t.A_base_correct_no_harm} ({100*t.A_base_correct_no_harm/n:.1f}%) | {t.D_base_correct_some_harm} ({100*t.D_base_correct_some_harm/n:.1f}%) | {t.B_base_wrong_repairable} ({100*t.B_base_wrong_repairable/n:.1f}%) | {t.C_base_wrong_unrepairable} ({100*t.C_base_wrong_unrepairable/n:.1f}%) |")
    out.append("\n### Semantic evaluation analysis (primary classifier head, seed 0)\n")
    out.append("| Setting | oracle action changed A→B | gated successes (A) | of which lexical-only | of which not a success under B | gated harms (A) | of which semantically equivalent under B | oracle gap A | oracle gap B |"); out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, s in Sem.iterrows():
        out.append(f"| {PRETTY[s.dataset]} {PRETTY[s.model]} | {s.oracle_action_changed_A_vs_B} | {s.gated_successes_A} | {s.gated_successes_lexical_only} | {s.gated_successes_not_success_under_B} | {s.gated_harms_A} | {s.gated_harms_semantically_equivalent} | {fmt(s.oracle_gap_A)} | {fmt(s.oracle_gap_B)} |")
    (TABLES / "revised_results.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
