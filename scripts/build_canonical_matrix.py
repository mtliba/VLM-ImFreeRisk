"""Add Metric-B (canonicalised) scores to the intervention matrix -> results/intervention_matrix_canonical.csv
(original file is left untouched). Columns: {action}_score_b, base_score_b, best_action_b, best_score_b, dq_{a}_b."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import score_canonical  # noqa: E402
from common import ACTIONS, RESULTS, load_matrix  # noqa: E402

df = load_matrix()
for a in ACTIONS:
    df[f"{a}_score_b"] = [score_canonical(p, r, q) for p, r, q in zip(df[f"{a}_answer"], df.reference_answer, df.question)]
df["base_score_b"] = df["keep_score_b"]
S = df[[f"{a}_score_b" for a in ACTIONS]].values
df["best_action_b"] = [ACTIONS[i] for i in S.argmax(1)]
df["best_score_b"] = S.max(1)
for a in ACTIONS:
    df[f"dq_{a}_b"] = df[f"{a}_score_b"] - df["base_score_b"]
df.to_csv(RESULTS / "intervention_matrix_canonical.csv", index=False)
t = df[df.split == "test"]
print("| Setting | Base A | Base B | Best fixed A | Best fixed B | Oracle A | Oracle B | Gap A | Gap B | oracle-KEEP A | oracle-KEEP B |")
for (d, m), g in t.groupby(["dataset", "model"]):
    ma = {a: g[f"{a}_score"].mean() for a in ACTIONS}; mb = {a: g[f"{a}_score_b"].mean() for a in ACTIONS}
    print(f"| {d}/{m} | {100*ma['keep']:.1f} | {100*mb['keep']:.1f} | {100*max(ma.values()):.1f} | {100*max(mb.values()):.1f} | {100*g.best_score.mean():.1f} | {100*g.best_score_b.mean():.1f} | "
          f"{100*(g.best_score.mean()-max(ma.values())):.1f} | {100*(g.best_score_b.mean()-max(mb.values())):.1f} | {100*(g.best_action=='keep').mean():.1f} | {100*(g.best_action_b=='keep').mean():.1f} |")
