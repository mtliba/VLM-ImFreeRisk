"""Evaluate anchored-verification outputs (results/verification_raw) on a split.

python scripts/evaluate_verification.py --split validation   -> hard-stop check on SLAKE/Qwen validation
python scripts/evaluate_verification.py --split test         -> main tables, safety, agreement, bootstrap, review candidates
Outputs: results/verification_{split}.csv, results/verification_{split}_agreement.csv, results/verification_{split}_bootstrap.csv,
         results/verification_{split}_decisions.csv, tables/verification_{split}.md
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonicalize, closed_pred_canonical  # noqa: E402
from common import ACTIONS, RESULTS, ROOT, read_jsonl  # noqa: E402

TABLES = ROOT / "tables"
SETTINGS = [(d, m) for d in ["vqa_rad", "slake"] for m in ["medgemma", "qwen"]]
PRETTY = {"vqa_rad": "VR", "slake": "SL", "medgemma": "MG", "qwen": "Q"}
VERIFIERS = ["global", "enhance", "zoom"]
SEEDS = [0, 1, 2]
rng = np.random.RandomState(0)
GATED_SCORES_A = {("vqa_rad", "medgemma"): 63.1, ("vqa_rad", "qwen"): 58.8, ("slake", "medgemma"): 67.4, ("slake", "qwen"): 62.8}


def key(ans, q):
    yn = closed_pred_canonical(ans, q)
    return yn if yn in ("yes", "no") else canonicalize(ans)


def load_setting(ds, m, split):
    M = pd.read_csv(RESULTS / "intervention_matrix_canonical.csv", keep_default_na=False)
    for c in M.columns:
        if c.endswith(("_score", "_score_b", "_conf")) or c.startswith("dq_"):
            M[c] = pd.to_numeric(M[c], errors="coerce")
    te = M[(M.dataset == ds) & (M.model == m) & (M.split == split)].reset_index(drop=True)
    V = {}
    for v in VERIFIERS:
        p = RESULTS / "verification_raw" / f"{ds}_{m}_{split}_{v}.jsonl"
        if not p.exists():
            return None, None
        V[v] = {r["sample_id"]: r for r in read_jsonl(p)}
    keep_ids = [s for s in te.sample_id if all(s in V[v] for v in VERIFIERS)]
    te = te[te.sample_id.isin(keep_ids)].reset_index(drop=True)
    for v in VERIFIERS:
        te[f"v_{v}_decision"] = [V[v][s]["decision"] for s in te.sample_id]
        te[f"v_{v}_answer"] = [V[v][s]["final_answer"] for s in te.sample_id]
        te[f"v_{v}_raw"] = [V[v][s]["output_raw"] for s in te.sample_id]
        te[f"v_{v}_score"] = [V[v][s]["score"] for s in te.sample_id]
        te[f"v_{v}_score_b"] = [V[v][s]["score_b"] for s in te.sample_id]
        te[f"v_{v}_key"] = [key(V[v][s]["final_answer"], q) for s, q in zip(te.sample_id, te.question)]
    te["keep_key"] = [key(a, q) for a, q in zip(te.keep_answer, te.question)]
    return te, V


# ---------------------------------------------------------------- policies: return index into candidate list [keep, global, enhance, zoom]
def proposals(row):
    """verifier -> canonical corrected key, only for CORRECT decisions that differ from KEEP."""
    return {v: row[f"v_{v}_key"] for v in VERIFIERS if row[f"v_{v}_decision"] == "CORRECT" and row[f"v_{v}_key"] != row["keep_key"]}


def pol_single(row, v):
    return VERIFIERS.index(v) + 1 if row[f"v_{v}_decision"] == "CORRECT" else 0


def pol_any(row):
    P = proposals(row)
    if not P:
        return 0
    if len(set(P.values())) == 1:
        return VERIFIERS.index(next(iter(P))) + 1
    return 0


def pol_agree(row, k):
    P = proposals(row)
    c = Counter(P.values())
    for ans, n in c.items():
        if n >= k:
            return VERIFIERS.index([v for v in VERIFIERS if P.get(v) == ans][0]) + 1
    return 0


def agreement_pattern(row):
    P = proposals(row)
    n = len(P); d = len(set(P.values()))
    if n == 0:
        return "no correction proposed"
    if n == 1:
        return "1 verifier proposes"
    if n == 2:
        return "2 propose, agree" if d == 1 else "2 propose, disagree"
    return "3 propose, all agree" if d == 1 else ("3 propose, 2 agree" if d == 2 else "3 propose, all disagree")


def scores_of(te, metric):
    suf = "" if metric == "A" else "_b"
    return np.stack([te[f"keep_score{suf}"].values] + [te[f"v_{v}_score{suf}"].values for v in VERIFIERS], 1).astype(float)


def keys_of(te):
    return np.stack([te.keep_key.values] + [te[f"v_{v}_key"].values for v in VERIFIERS], 1)


def stats(S, K, a):
    r = S[np.arange(len(a)), a]; base = S[:, 0]
    changed = np.array([K[i, a[i]] != K[i, 0] for i in range(len(a))])
    return dict(score=r.mean(), correction_rate=float(changed.mean()),
                precision=float((r[changed] > base[changed]).mean()) if changed.any() else np.nan,
                harm_among=float((r[changed] < base[changed]).mean()) if changed.any() else np.nan,
                abs_corrected=float((r > base).mean()), abs_harmed=float((r < base).mean()), regret=float((S.max(1) - r).mean()),
                mean_delta=float((r - base).mean())), r


def prev_actions(ds, m, split, ids):
    """previous gated selector (3 seeds) and direct-dQ selector actions on the ORIGINAL 4-action matrix."""
    if split != "test":
        return None, None
    g, d = [], []
    for s in SEEDS:
        P = pd.read_csv(RESULTS / "revised_preds" / f"per_setting_biomedclip_iqy_seed{s}.csv", keep_default_na=False)
        P = P[(P.dataset == ds) & (P.model == m) & (P.split == "test") & (P["head"] == "cls")].set_index("sample_id").loc[ids]
        g.append(np.array([ACTIONS.index(x) for x in P.selected]))
        D = pd.read_csv(RESULTS / "selector_preds" / f"{ds}_{m}_iqy_biomedclip_seed{s}.csv"); D = D[D.split == "test"].set_index("sample_id").loc[ids]
        d.append(np.array([ACTIONS.index(x) for x in D.selected]))
    return g, d


def main(split):
    rows, agree_rows, boots, decisions, verifier_rows = [], [], [], [], []
    hardstop = {}
    for ds, m in SETTINGS:
        te, V = load_setting(ds, m, split)
        if te is None:
            print("missing", ds, m, split); continue
        ids = te.sample_id.tolist()
        K = keys_of(te)
        acts = {"Base": np.zeros(len(te), int)}
        for v in VERIFIERS:
            acts[f"Verify {v.capitalize()}"] = np.array([pol_single(r, v) for _, r in te.iterrows()])
        acts["Any verifier"] = np.array([pol_any(r) for _, r in te.iterrows()])
        acts["2-verifier corroboration"] = np.array([pol_agree(r, 2) for _, r in te.iterrows()])
        acts["3-verifier corroboration"] = np.array([pol_agree(r, 3) for _, r in te.iterrows()])
        # per-verifier diagnostics
        for v in VERIFIERS:
            dec = te[f"v_{v}_decision"].values
            SA = scores_of(te, "A"); a = acts[f"Verify {v.capitalize()}"]
            r = SA[np.arange(len(a)), a]; base = SA[:, 0]
            corr = dec == "CORRECT"; changed = np.array([K[i, a[i]] != K[i, 0] for i in range(len(a))])
            verifier_rows.append(dict(dataset=ds, model=m, split=split, verifier=v, n=len(te), keep_rate=float((dec == "KEEP").mean()), correct_rate=float(corr.mean()),
                                      valid_rate=float((dec != "INVALID").mean()), correct_but_same_answer=float((corr & ~changed).mean()),
                                      precision_given_correct=float((r[corr] > base[corr]).mean()) if corr.any() else np.nan,
                                      harm_given_correct=float((r[corr] < base[corr]).mean()) if corr.any() else np.nan,
                                      precision_given_changed=float((r[changed] > base[changed]).mean()) if changed.any() else np.nan,
                                      harm_given_changed=float((r[changed] < base[changed]).mean()) if changed.any() else np.nan,
                                      abs_corrected=float((r > base).mean()), abs_harmed=float((r < base).mean()), mean_delta=float((r - base).mean())))
        # previous methods on test
        gated, direct = prev_actions(ds, m, split, ids)
        per = {}
        for metric in ["A", "B"]:
            S = scores_of(te, metric)
            suf = "" if metric == "A" else "_b"
            S4 = te[[f"{a}_score{suf}" for a in ACTIONS]].values.astype(float)
            pol = dict(acts)
            pol["Oracle verifier"] = S.argmax(1)
            for name, a in pol.items():
                st, r = stats(S, K, a)
                rows.append(dict(dataset=ds, model=m, split=split, metric=metric, method=name, **st))
                per[(metric, name)] = r
                if metric == "A":
                    for i in range(len(te)):
                        decisions.append(dict(sample_id=ids[i], dataset=ds, model=m, method=name, chosen=(["keep"] + VERIFIERS)[a[i]], answer=([te.keep_answer[i]] + [te[f"v_{v}_answer"][i] for v in VERIFIERS])[a[i]],
                                              score_a=float(S[i, a[i]]), base_a=float(S[i, 0]), pattern=agreement_pattern(te.iloc[i])))
            # previous baselines (4-action matrix)
            va = pd.read_csv(RESULTS / "intervention_matrix_canonical.csv", keep_default_na=False)
            va = va[(va.dataset == ds) & (va.model == m) & (va.split == "validation")]
            Sv = va[[f"{a}_score{suf}" for a in ACTIONS]].apply(pd.to_numeric).values
            fixed = int(np.argmax(Sv.mean(0)))
            r_fixed = S4[:, fixed]; per[(metric, "Previous best fixed")] = r_fixed
            K4 = np.stack([te[f"{a}_key"].values if f"{a}_key" in te else np.array([key(x, q) for x, q in zip(te[f"{a}_answer"], te.question)]) for a in ACTIONS], 1)
            st, _ = stats(S4, K4, np.full(len(te), fixed)); rows.append(dict(dataset=ds, model=m, split=split, metric=metric, method="Previous best fixed", fixed_action=ACTIONS[fixed], **st))
            if gated is not None:
                rg = np.mean([S4[np.arange(len(te)), g] for g in gated], 0); per[(metric, "Previous gated selector")] = rg
                for g in gated:
                    st, _ = stats(S4, K4, g); rows.append(dict(dataset=ds, model=m, split=split, metric=metric, method="Previous gated selector", **st))
                rd = np.mean([S4[np.arange(len(te)), g] for g in direct], 0); per[(metric, "Previous direct dQ selector")] = rd
                for g in direct:
                    st, _ = stats(S4, K4, g); rows.append(dict(dataset=ds, model=m, split=split, metric=metric, method="Previous direct dQ selector", **st))
                per[(metric, "Oracle (4 original actions)")] = S4.max(1)
                rows.append(dict(dataset=ds, model=m, split=split, metric=metric, method="Oracle (4 original actions)", score=S4.max(1).mean()))
            # bootstrap
            idx = rng.randint(0, len(te), (1000, len(te)))
            for name in ["2-verifier corroboration", "3-verifier corroboration", "Any verifier", "Verify Global", "Verify Enhance", "Verify Zoom"]:
                for ref in ["Base", "Previous best fixed", "Previous gated selector", "Previous direct dQ selector"]:
                    if (metric, ref) not in per:
                        continue
                    d = per[(metric, name)] - per[(metric, ref)]; b = d[idx].mean(1)
                    boots.append(dict(dataset=ds, model=m, split=split, metric=metric, method=name, reference=ref, diff=float(d.mean()), ci_low=float(np.percentile(b, 2.5)), ci_high=float(np.percentile(b, 97.5))))
        # agreement analysis (metric A): accepted answer = the (majority) proposed correction
        SA = scores_of(te, "A")
        for i, r in te.iterrows():
            pat = agreement_pattern(r)
            if pat == "no correction proposed":
                continue
            P = proposals(r)
            top = Counter(P.values()).most_common(1)[0][0]
            j = VERIFIERS.index([v for v in VERIFIERS if P.get(v) == top][0]) + 1
            agree_rows.append(dict(dataset=ds, model=m, split=split, pattern=pat, base=SA[i, 0], new=SA[i, j]))
        # hard-stop numbers (validation, metric A)
        if split == "validation":
            SA = scores_of(te, "A"); base = SA[:, 0].mean()
            for name in ["2-verifier corroboration", "3-verifier corroboration"]:
                st, _ = stats(SA, K, acts[name])
                hardstop[(ds, m, name)] = dict(precision=st["precision"], harm=st["harm_among"], coverage=st["correction_rate"], score=st["score"], base=base)
    R = pd.DataFrame(rows); R.to_csv(RESULTS / f"verification_{split}.csv", index=False)
    pd.DataFrame(verifier_rows).to_csv(RESULTS / f"verification_{split}_verifiers.csv", index=False)
    AG = pd.DataFrame(agree_rows)
    if not AG.empty:
        AG["better"] = AG.new > AG.base; AG["worse"] = AG.new < AG.base; AG["delta"] = AG.new - AG.base
        AGs = AG.groupby(["dataset", "model", "pattern"]).agg(n=("delta", "size"), precision=("better", "mean"), harm=("worse", "mean"), mean_delta=("delta", "mean")).reset_index()
        AGs.to_csv(RESULTS / f"verification_{split}_agreement.csv", index=False)
    else:
        AGs = pd.DataFrame()
    pd.DataFrame(boots).to_csv(RESULTS / f"verification_{split}_bootstrap.csv", index=False)
    pd.DataFrame(decisions).to_csv(RESULTS / f"verification_{split}_decisions.csv", index=False)
    write_tables(split, R, pd.DataFrame(verifier_rows), AGs, pd.DataFrame(boots), hardstop)


def fmt(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{d}f}"


def write_tables(split, R, VR, AG, B, hardstop):
    cols = [f"{PRETTY[d]}-{PRETTY[m]}" for d, m in SETTINGS]
    out = [f"## Anchored verification — {split} split\n", "### Per-verifier diagnostics (Metric A)\n",
           "| Setting | Verifier | n | KEEP rate | CORRECT rate | valid-format rate | CORRECT but same canonical answer | P(improve \\| CORRECT) | P(harm \\| CORRECT) | P(improve \\| changed) | P(harm \\| changed) | abs. corrected | abs. harmed | mean Δscore |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in VR.iterrows():
        out.append(f"| {PRETTY[r.dataset]}-{PRETTY[r.model]} | {r.verifier} | {r.n} | {fmt(r.keep_rate)} | {fmt(r.correct_rate)} | {fmt(r.valid_rate)} | {fmt(r.correct_but_same_answer)} | {fmt(r.precision_given_correct)} | {fmt(r.harm_given_correct)} | {fmt(r.precision_given_changed)} | {fmt(r.harm_given_changed)} | {fmt(r.abs_corrected)} | {fmt(r.abs_harmed)} | {100*r.mean_delta:+.2f} |")
    order = ["Base", "Previous best fixed", "Previous direct dQ selector", "Previous gated selector", "Verify Global", "Verify Enhance", "Verify Zoom", "Any verifier", "2-verifier corroboration", "3-verifier corroboration", "Oracle verifier", "Oracle (4 original actions)"]
    for metric in ["A", "B"]:
        out.append(f"\n### Policies — Metric {metric}, {split}, mean answer score ×100\n")
        out.append("| Method | " + " | ".join(cols) + " | Avg |"); out.append("|---|" + "---:|" * 5)
        for p in order:
            vals = [R[(R.dataset == d) & (R.model == m) & (R.metric == metric) & (R.method == p)].score.mean() for d, m in SETTINGS]
            if all(np.isnan(v) for v in vals):
                continue
            out.append(f"| {p} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(np.nanmean(vals))} |")
        out.append(f"\n### Safety — Metric {metric} (changed = output canonical answer ≠ initial canonical answer)\n")
        out.append("| Setting | Method | Score | Correction rate | Correction precision | Harm among corrections | Abs. corrected | Abs. harmed | Regret |"); out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for d, m in SETTINGS:
            for p in order:
                g = R[(R.dataset == d) & (R.model == m) & (R.metric == metric) & (R.method == p)]
                if g.empty or "correction_rate" not in g or g.correction_rate.isna().all():
                    continue
                out.append(f"| {PRETTY[d]}-{PRETTY[m]} | {p} | {fmt(g.score.mean())} | {fmt(g.correction_rate.mean())} | {fmt(g.precision.mean())} | {fmt(g.harm_among.mean())} | {fmt(g.abs_corrected.mean())} | {fmt(g.abs_harmed.mean())} | {fmt(g.regret.mean(), 2)} |")
    if not AG.empty:
        out.append("\n### Agreement analysis (Metric A; accepted answer = majority proposed correction)\n")
        out.append("| Setting | Pattern | n | Correction precision | Harmful fraction | Mean Δscore |"); out.append("|---|---|---:|---:|---:|---:|")
        for _, r in AG.iterrows():
            out.append(f"| {PRETTY[r.dataset]}-{PRETTY[r.model]} | {r.pattern} | {r.n} | {fmt(r.precision)} | {fmt(r.harm)} | {100*r.mean_delta:+.2f} |")
    if not B.empty:
        out.append("\n### Paired bootstrap (1000), method − reference ×100 [95% CI]; * = CI excludes 0\n")
        out.append("| Setting | Metric | Method | − Base | − Prev. best fixed | − Prev. gated | − Prev. direct ΔQ |"); out.append("|---|---|---|---:|---:|---:|---:|")
        for d, m in SETTINGS:
            for metric in ["A", "B"]:
                for p in ["2-verifier corroboration", "3-verifier corroboration", "Any verifier", "Verify Global", "Verify Enhance", "Verify Zoom"]:
                    cells = []
                    for ref in ["Base", "Previous best fixed", "Previous gated selector", "Previous direct dQ selector"]:
                        g = B[(B.dataset == d) & (B.model == m) & (B.metric == metric) & (B.method == p) & (B.reference == ref)]
                        cells.append("—" if g.empty else f"{100*g['diff'].iloc[0]:+.2f} [{100*g.ci_low.iloc[0]:+.2f}, {100*g.ci_high.iloc[0]:+.2f}]{'*' if (g.ci_low.iloc[0] > 0 or g.ci_high.iloc[0] < 0) else ''}")
                    out.append(f"| {PRETTY[d]}-{PRETTY[m]} | {metric} | {p} | " + " | ".join(cells) + " |")
    if hardstop:
        out.append("\n### Validation hard-stop check (SLAKE/Qwen: precision ≥ 60 %, harm < 5 %, ≥ 50 % of gated gain (62.8 − 56.4 on test; validation gated gain taken from the revised run log), coverage ≥ 3 %)\n")
        out.append("| Setting | Policy | Correction precision | Harm among corrections | Coverage | Score | Base | Gain over base |"); out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for (d, m, p), h in hardstop.items():
            out.append(f"| {PRETTY[d]}-{PRETTY[m]} | {p} | {fmt(h['precision'])} | {fmt(h['harm'])} | {fmt(h['coverage'])} | {fmt(h['score'])} | {fmt(h['base'])} | {100*(h['score']-h['base']):+.2f} |")
    (TABLES / f"verification_{split}.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation")
    main(ap.parse_args().split)
