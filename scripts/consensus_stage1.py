"""Stage 1 — zero-cost consensus / corroboration policies on the existing KEEP/REPROMPT/ENHANCE/ZOOM answers.

Canonical answer key = canonicalised answer (evaluation/canonicalization_rules.yaml), with yes/no-equivalent
phrases mapped to yes/no. Policies P0..P7 (+ extra: direct-dQ selector + corroboration). Both metrics.
Outputs: results/consensus_stage1.csv, results/consensus_stage1_bootstrap.csv, results/consensus_stage1_decisions.csv,
         tables/consensus_stage1.md
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonicalize, closed_pred_canonical  # noqa: E402
from common import ACTIONS, RESULTS, ROOT  # noqa: E402

TABLES = ROOT / "tables"
SETTINGS = [(d, m) for d in ["vqa_rad", "slake"] for m in ["medgemma", "qwen"]]
PRETTY = {"vqa_rad": "VR", "slake": "SL", "medgemma": "MG", "qwen": "Q"}
INTERV = ACTIONS[1:]
SEEDS = [0, 1, 2]
rng = np.random.RandomState(0)
PRIMARY = json.load(open(RESULTS / "revised_primary_config.json"))["cls"][:3]


def key(ans, q):
    yn = closed_pred_canonical(ans, q)
    return yn if yn in ("yes", "no") else canonicalize(ans)


def load():
    M = pd.read_csv(RESULTS / "intervention_matrix_canonical.csv", keep_default_na=False)
    for c in M.columns:
        if c.endswith(("_score", "_score_b", "_conf")) or c.startswith("dq_"):
            M[c] = pd.to_numeric(M[c], errors="coerce")
    for a in ACTIONS:
        M[f"{a}_key"] = [key(x, q) for x, q in zip(M[f"{a}_answer"], M.question)]
    return M


def rep_action(keys, target):
    """representative action producing canonical answer `target` (KEEP preferred, then R, E, Z)."""
    for i, k in enumerate(keys):
        if k == target:
            return i
    return 0


# ---------------------------------------------------------------- policies: return chosen action index per sample
def p_majority(keys):
    c = Counter(keys)
    best = max(c.values())
    cands = [k for k, v in c.items() if v == best]
    target = keys[0] if keys[0] in cands else cands[0]
    return rep_action(keys, target)


def p_two_corroboration(keys):
    c = Counter(keys[1:])
    for k, v in c.items():
        if v >= 2 and k != keys[0]:
            return rep_action(keys, k)
    return 0


def p_keep_anchored(keys):
    c = Counter(keys)
    alt = [(v, k) for k, v in c.items() if k != keys[0]]
    if alt and max(alt)[0] > c[keys[0]]:
        return rep_action(keys, max(alt)[1])
    return 0


def p_unanimous(keys):
    if keys[1] == keys[2] == keys[3] and keys[1] != keys[0]:
        return 1
    return 0


def p_selector_corroborated(keys, sel_action):
    if sel_action == 0:
        return 0
    cand = keys[sel_action]
    if cand == keys[0]:
        return 0
    if any(keys[j] == cand for j in range(1, 4) if j != sel_action):
        return sel_action
    return 0


def gated_actions(ds, m, seed, ids, regime="per_setting", enc="biomedclip", feat="iqy", head="cls"):
    P = pd.read_csv(RESULTS / "revised_preds" / f"{regime}_{enc}_{feat}_seed{seed}.csv", keep_default_na=False)
    P = P[(P.dataset == ds) & (P.model == m) & (P.split == "test") & (P["head"] == head)].set_index("sample_id").loc[ids]
    return np.array([ACTIONS.index(a) for a in P.selected])


def direct_actions(ds, m, seed, ids):
    d = pd.read_csv(RESULTS / "selector_preds" / f"{ds}_{m}_iqy_biomedclip_seed{seed}.csv")
    d = d[d.split == "test"].set_index("sample_id").loc[ids]
    return np.array([ACTIONS.index(a) for a in d.selected])


def evaluate(te, act, metric):
    suf = "" if metric == "A" else "_b"
    S = te[[f"{a}_score{suf}" for a in ACTIONS]].values
    K = te[[f"{a}_key" for a in ACTIONS]].values
    r = S[np.arange(len(act)), act]; base = S[:, 0]
    changed = np.array([K[i, act[i]] != K[i, 0] for i in range(len(act))])
    return dict(score=r.mean(), action_rate=float((act != 0).mean()), change_rate=float(changed.mean()),
                harmful_correction_rate=float((r[changed] < base[changed]).mean()) if changed.any() else 0.0,
                successful_correction_rate=float((r[changed] > base[changed]).mean()) if changed.any() else 0.0,
                abs_harmed=float((r < base).mean()), abs_corrected=float((r > base).mean()),
                regret=float((S.max(1) - r).mean())), r


def main():
    M = load()
    rows, boots, decisions = [], [], []
    for ds, m in SETTINGS:
        va = M[(M.dataset == ds) & (M.model == m) & (M.split == "validation")].reset_index(drop=True)
        te = M[(M.dataset == ds) & (M.model == m) & (M.split == "test")].reset_index(drop=True)
        K = te[[f"{a}_key" for a in ACTIONS]].values
        ids = te.sample_id.tolist()
        policies = {
            "Base": [np.zeros(len(te), int)],
            "Majority": [np.array([p_majority(list(k)) for k in K])],
            "2-intervention corroboration": [np.array([p_two_corroboration(list(k)) for k in K])],
            "KEEP-anchored majority": [np.array([p_keep_anchored(list(k)) for k in K])],
            "Unanimous correction": [np.array([p_unanimous(list(k)) for k in K])],
        }
        gated = [gated_actions(ds, m, s, ids) for s in SEEDS]
        direct = [direct_actions(ds, m, s, ids) for s in SEEDS]
        policies["Current gated selector"] = gated
        policies["Direct dQ selector"] = direct
        p2 = policies["2-intervention corroboration"][0]
        policies["Gate + corroboration"] = [np.where(g == 0, 0, p2) for g in gated]
        policies["Selector + corroboration"] = [np.array([p_selector_corroborated(list(K[i]), int(g[i])) for i in range(len(te))]) for g in gated]
        policies["Direct dQ + corroboration"] = [np.array([p_selector_corroborated(list(K[i]), int(g[i])) for i in range(len(te))]) for g in direct]
        for metric in ["A", "B"]:
            suf = "" if metric == "A" else "_b"
            Sv = va[[f"{a}_score{suf}" for a in ACTIONS]].values
            fixed = int(np.argmax(Sv.mean(0)))
            pol = dict(policies)
            pol["Best fixed"] = [np.full(len(te), fixed)]
            St = te[[f"{a}_score{suf}" for a in ACTIONS]].values
            pol["Oracle"] = [St.argmax(1)]
            per = {}
            for name, acts in pol.items():
                rs = []
                for si, a in enumerate(acts):
                    stats, r = evaluate(te, a, metric)
                    rows.append(dict(dataset=ds, model=m, metric=metric, policy=name, seed=si if len(acts) > 1 else None,
                                     fixed_action=ACTIONS[fixed] if name == "Best fixed" else "", **stats))
                    rs.append(r)
                    if metric == "A" and si == 0:
                        for i in range(len(te)):
                            decisions.append(dict(sample_id=ids[i], dataset=ds, model=m, policy=name, action=ACTIONS[a[i]], answer_key=K[i, a[i]], keep_key=K[i, 0],
                                                  score_a=float(St[i, a[i]]), base_a=float(St[i, 0])))
                per[name] = np.mean(rs, 0)
            idx = rng.randint(0, len(te), (1000, len(te)))
            for name in pol:
                for ref in ["Base", "Best fixed", "Current gated selector"]:
                    if name == ref:
                        continue
                    d = per[name] - per[ref]
                    b = d[idx].mean(1)
                    boots.append(dict(dataset=ds, model=m, metric=metric, policy=name, reference=ref, diff=float(d.mean()),
                                      ci_low=float(np.percentile(b, 2.5)), ci_high=float(np.percentile(b, 97.5))))
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "consensus_stage1.csv", index=False)
    B = pd.DataFrame(boots); B.to_csv(RESULTS / "consensus_stage1_bootstrap.csv", index=False)
    pd.DataFrame(decisions).to_csv(RESULTS / "consensus_stage1_decisions.csv", index=False)
    write_tables(R, B)


def fmt(x, d=1):
    return f"{100*x:.{d}f}"


def write_tables(R, B):
    order = ["Base", "Best fixed", "Direct dQ selector", "Current gated selector", "Majority", "2-intervention corroboration", "KEEP-anchored majority",
             "Unanimous correction", "Gate + corroboration", "Selector + corroboration", "Direct dQ + corroboration", "Oracle"]
    cols = [f"{PRETTY[d]}-{PRETTY[m]}" for d, m in SETTINGS]
    out = []
    for metric in ["A", "B"]:
        out.append(f"\n### Stage-1 policies — Metric {metric} ({'original' if metric == 'A' else 'canonicalised'}), test, mean answer score ×100 (learned policies = mean over 3 seeds)\n")
        out.append("| Policy | " + " | ".join(cols) + " | Avg |"); out.append("|---|" + "---:|" * 5)
        for p in order:
            vals = [R[(R.dataset == d) & (R.model == m) & (R.metric == metric) & (R.policy == p)].score.mean() for d, m in SETTINGS]
            extra = ""
            if p == "Best fixed":
                extra = " (" + "/".join(R[(R.dataset == d) & (R.model == m) & (R.metric == metric) & (R.policy == p)].fixed_action.iloc[0].upper() for d, m in SETTINGS) + ")"
            out.append(f"| {p}{extra} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(np.mean(vals))} |")
        out.append(f"\n### Safety metrics — Metric {metric} (changed = output canonical answer ≠ KEEP canonical answer)\n")
        out.append("| Setting | Policy | Score | Action ≠ KEEP | Answer changed | Successful correction | Harmful correction | Abs. corrected | Abs. harmed | Regret |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for d, m in SETTINGS:
            for p in order:
                g = R[(R.dataset == d) & (R.model == m) & (R.metric == metric) & (R.policy == p)]
                out.append(f"| {PRETTY[d]}-{PRETTY[m]} | {p} | {fmt(g.score.mean())} | {fmt(g.action_rate.mean())} | {fmt(g.change_rate.mean())} | {fmt(g.successful_correction_rate.mean())} | "
                           f"{fmt(g.harmful_correction_rate.mean())} | {fmt(g.abs_corrected.mean())} | {fmt(g.abs_harmed.mean())} | {fmt(g.regret.mean(), 2)} |")
    out.append("\n### Paired bootstrap (1000), policy − reference, ×100 [95% CI]; * = CI excludes 0\n")
    out.append("| Setting | Metric | Policy | − Base | − Best fixed | − Current gated selector |"); out.append("|---|---|---|---:|---:|---:|")
    for d, m in SETTINGS:
        for metric in ["A", "B"]:
            for p in order:
                if p in ("Base",):
                    continue
                cells = []
                for ref in ["Base", "Best fixed", "Current gated selector"]:
                    g = B[(B.dataset == d) & (B.model == m) & (B.metric == metric) & (B.policy == p) & (B.reference == ref)]
                    cells.append("—" if g.empty else f"{100*g['diff'].iloc[0]:+.2f} [{100*g.ci_low.iloc[0]:+.2f}, {100*g.ci_high.iloc[0]:+.2f}]{'*' if (g.ci_low.iloc[0] > 0 or g.ci_high.iloc[0] < 0) else ''}")
                out.append(f"| {PRETTY[d]}-{PRETTY[m]} | {metric} | {p} | " + " | ".join(cells) + " |")
    (TABLES / "consensus_stage1.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
