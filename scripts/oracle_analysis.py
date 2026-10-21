"""Stage 1: build the intervention matrix and the oracle / GO-NO-GO analysis.

Inputs : results/raw/{dataset}_{model}_{split}_{action}.jsonl
Outputs: results/intervention_matrix.csv   (one row per sample x model, all splits)
         results/oracle_analysis.csv       (per dataset/model/split summary)
         tables/stage1_oracle_table.md, tables/stage1_action_distribution.md,
         tables/stage1_complementarity.md, tables/stage1_harm.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTIONS, ACTION_NAMES, DATA, RESULTS, ROOT, canonical_qtype, read_jsonl  # noqa: E402

TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)
DATASETS = ["vqa_rad", "slake"]
MODELS = ["medgemma", "qwen"]
SPLITS = ["train", "validation", "test"]
PRETTY = {"vqa_rad": "VQA-RAD", "slake": "SLAKE", "medgemma": "MedGemma-4B", "qwen": "Qwen2.5-VL-7B"}


def build_matrix():
    rows = []
    for ds in DATASETS:
        for split in SPLITS:
            p = DATA / ds / f"{split}.jsonl"
            if not p.exists():
                continue
            samples = {s["sample_id"]: s for s in read_jsonl(p)}
            for m in MODELS:
                per_action = {}
                for a in ACTIONS:
                    rp = RESULTS / "raw" / f"{ds}_{m}_{split}_{a}.jsonl"
                    if rp.exists():
                        per_action[a] = {r["sample_id"]: r for r in read_jsonl(rp)}
                if len(per_action) < len(ACTIONS):
                    continue
                ids = set(samples)
                for a in ACTIONS:
                    ids &= set(per_action[a])
                for sid in sorted(ids):
                    s = samples[sid]
                    r = dict(sample_id=sid, dataset=ds, model=m, split=split,
                             question=s["question"], reference_answer=s["answer"],
                             answer_type=s["answer_type"], question_type=canonical_qtype(s["question_type"]),
                             image_path=s["image_path"])
                    for a in ACTIONS:
                        r[f"{a}_answer"] = per_action[a][sid]["answer_raw"]
                        r[f"{a}_score"] = per_action[a][sid]["score"]
                        r[f"{a}_conf"] = per_action[a][sid]["confidence"]
                    r["base_answer"] = r["keep_answer"]
                    r["base_score"] = r["keep_score"]
                    r["base_conf"] = r["keep_conf"]
                    scores = np.array([r[f"{a}_score"] for a in ACTIONS])
                    # oracle: best action; ties broken in favour of KEEP (index 0 = argmax first)
                    best = int(np.argmax(scores))
                    r["best_action"] = ACTIONS[best]
                    r["best_score"] = float(scores[best])
                    r["oracle_improvement"] = float(scores[best] - r["base_score"])
                    for a in ACTIONS:
                        r[f"dq_{a}"] = float(r[f"{a}_score"] - r["base_score"])
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "intervention_matrix.csv", index=False)
    return df


def fmt(x):
    return f"{100*x:.1f}"


def analyse(df, split="test"):
    d = df[df.split == split]
    summary, t1, t2, t3, t4 = [], [], [], [], []
    t1.append("| Dataset | VLM | n | Base | Reprompt | Enhance | Zoom | Best Fixed (action) | Oracle | Oracle Gap |")
    t1.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    t2.append("| Dataset / Model | KEEP | REPROMPT | ENHANCE | ZOOM | any-intervention-strictly-better |")
    t2.append("|---|---:|---:|---:|---:|---:|")
    t4.append("| Dataset / Model | action | P(worse than base) | P(better than base) | P(unchanged) | mean dQ |")
    t4.append("|---|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        for m in MODELS:
            g = d[(d.dataset == ds) & (d.model == m)]
            if g.empty:
                continue
            means = {a: g[f"{a}_score"].mean() for a in ACTIONS}
            fixed_best = max(ACTIONS, key=lambda a: means[a])
            oracle = g["best_score"].mean()
            gap = oracle - means[fixed_best]
            closed = g[g.answer_type == "CLOSED"]
            open_ = g[g.answer_type == "OPEN"]
            rec = dict(dataset=ds, model=m, split=split, n=len(g),
                       n_closed=len(closed), n_open=len(open_),
                       **{f"{a}_mean": means[a] for a in ACTIONS},
                       **{f"{a}_closed_acc": closed[f"{a}_score"].mean() for a in ACTIONS},
                       **{f"{a}_open_f1": open_[f"{a}_score"].mean() for a in ACTIONS},
                       best_fixed=fixed_best, best_fixed_mean=means[fixed_best],
                       oracle_mean=oracle, oracle_gap=gap)
            # oracle action distribution (ties -> KEEP)
            dist = g["best_action"].value_counts(normalize=True).reindex(ACTIONS).fillna(0)
            for a in ACTIONS:
                rec[f"oracle_frac_{a}"] = dist[a]
            rec["frac_any_improves"] = float((g["oracle_improvement"] > 0).mean())
            # harm
            for a in ACTIONS[1:]:
                rec[f"harm_{a}"] = float((g[f"{a}_score"] < g["base_score"]).mean())
                rec[f"help_{a}"] = float((g[f"{a}_score"] > g["base_score"]).mean())
            summary.append(rec)
            t1.append(f"| {PRETTY[ds]} | {PRETTY[m]} | {len(g)} | {fmt(means['keep'])} | {fmt(means['reprompt'])} | "
                      f"{fmt(means['enhance'])} | {fmt(means['zoom'])} | {fmt(means[fixed_best])} ({ACTION_NAMES[fixed_best]}) | "
                      f"{fmt(oracle)} | **{fmt(gap)}** |")
            t2.append(f"| {PRETTY[ds]} / {PRETTY[m]} | " + " | ".join(fmt(dist[a]) for a in ACTIONS) +
                      f" | {fmt(rec['frac_any_improves'])} |")
            # complementarity
            imp = {a: (g[f"{a}_score"] > g["base_score"]) for a in ACTIONS[1:]}
            hurt = {a: (g[f"{a}_score"] < g["base_score"]) for a in ACTIONS[1:]}
            t3.append(f"\n**{PRETTY[ds]} / {PRETTY[m]}** (n={len(g)}, base-wrong = {(g.base_score<1).sum()})\n")
            t3.append("| | improved by REPROMPT | improved by ENHANCE | improved by ZOOM |")
            t3.append("|---|---:|---:|---:|")
            for a in ACTIONS[1:]:
                t3.append(f"| improved by {ACTION_NAMES[a]} | " + " | ".join(str(int((imp[a] & imp[b]).sum())) for b in ACTIONS[1:]) + " |")
            only = {a: int((imp[a] & ~np.logical_or.reduce([imp[b] for b in ACTIONS[1:] if b != a])).sum()) for a in ACTIONS[1:]}
            allhurt = int(np.logical_and.reduce([hurt[a] for a in ACTIONS[1:]]).sum())
            anyimp = int(np.logical_or.reduce([imp[a] for a in ACTIONS[1:]]).sum())
            t3.append(f"\nimproved by exactly one action: " + ", ".join(f"{ACTION_NAMES[a]} only = {only[a]}" for a in ACTIONS[1:]) +
                      f"; improved by >=1 action = {anyimp}; every intervention hurts = {allhurt}; "
                      f"unrepaired base errors = {int(((g.base_score<1) & ~np.logical_or.reduce([imp[a] for a in ACTIONS[1:]])).sum())}")
            for a in ACTIONS[1:]:
                t4.append(f"| {PRETTY[ds]} / {PRETTY[m]} | {ACTION_NAMES[a]} | {fmt(rec[f'harm_{a}'])} | {fmt(rec[f'help_{a}'])} | "
                          f"{fmt(1-rec[f'harm_{a}']-rec[f'help_{a}'])} | {100*g[f'dq_{a}'].mean():+.2f} |")
    return summary, t1, t2, t3, t4


if __name__ == "__main__":
    df = build_matrix()
    print("intervention matrix:", df.groupby(["dataset", "model", "split"]).size().to_dict())
    all_summ = []
    for split in SPLITS:
        if (df.split == split).any():
            s, t1, t2, t3, t4 = analyse(df, split)
            all_summ += s
            if split == "test":
                (TABLES / "stage1_oracle_table.md").write_text("\n".join(t1) + "\n")
                (TABLES / "stage1_action_distribution.md").write_text("\n".join(t2) + "\n")
                (TABLES / "stage1_complementarity.md").write_text("\n".join(t3) + "\n")
                (TABLES / "stage1_harm.md").write_text("\n".join(t4) + "\n")
                print("\n".join(t1)); print(); print("\n".join(t2)); print(); print("\n".join(t3)); print(); print("\n".join(t4))
    pd.DataFrame(all_summ).to_csv(RESULTS / "oracle_analysis.csv", index=False)
