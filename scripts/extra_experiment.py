"""Follow-up experiment (offline, existing predictions/features only).

(i)  VQA-RAD cross-fitted gate: 5-fold GroupKFold (by image) over train+validation (1,793 QA). Per fold a gate
     (eps = 0, class-weighted BCE) and an action classifier are trained for a FIXED number of epochs (no checkpoint
     selection), giving out-of-fold gate AUROC/AUPRC; the 5 fold models are ensembled (mean probability) on the
     official test split, with tau chosen on the out-of-fold predictions (harm cap 5 %, as before).
     Answers: is the VQA-RAD gate failure a data-size / split-noise problem?
(ii) Harm-aware tau: on the existing gated predictions (results/revised_preds, primary classifier config) choose
     tau on validation with a stricter cap (validation harm <= 3 %) and abstain (never intervene) if no tau meets it.
     Answers: can SLAKE harm be brought under 5 % on test without losing the gain?
Outputs: results/extra_cv_vqarad.csv, results/extra_harm_aware.csv, tables/extra_experiment.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTIONS, RESULTS, ROOT, load_matrix  # noqa: E402
from train_gated import FEATS, Head, T, DEV, decide, realised, harm_rate  # noqa: E402

TABLES = ROOT / "tables"
SEEDS = [0, 1, 2]
TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)
A = {a: i for i, a in enumerate(ACTIONS)}


def load_feats(ds, m, split, feat, M):
    z = np.load(RESULTS / "features" / f"{ds}_{m}_{split}.npz")
    ids = list(z["sample_id"])
    X = np.concatenate([z[k] for k in FEATS[feat]], 1).astype(np.float32)
    g = M[(M.dataset == ds) & (M.model == m) & (M.split == split)].set_index("sample_id").loc[ids]
    S = g[[f"{a}_score" for a in ACTIONS]].values.astype(np.float32)
    return ids, X, S, g.image_path.values


def fit_fixed(model, X, Y, loss_fn, epochs, lr=1e-3, bs=128, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = T(X); sid = T(np.zeros(len(X)), torch.long)
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss_fn(model(Xt[idx], sid[idx]), Y[idx]).backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def pred(model, X):
    return model(T(X), T(np.zeros(len(X)), torch.long)).cpu().numpy()


def tau_rule(S, gp, act, cap):
    rows = []
    for tau in TAUS:
        a = decide(gp, act, tau)
        rows.append(dict(tau=tau, score=realised(S, a).mean(), harm=harm_rate(S, a), rate=float((a != 0).mean())))
    ok = [r for r in rows if r["harm"] <= cap]
    if not ok:
        return None, rows  # abstain
    return max(ok, key=lambda r: r["score"])["tau"], rows


def experiment_cv(M, out_rows):
    for m in ["medgemma", "qwen"]:
        for feat in ["q", "iq", "iqy"]:
            ids_tr, Xtr, Str, img_tr = load_feats("vqa_rad", m, "train", feat, M)
            ids_va, Xva, Sva, img_va = load_feats("vqa_rad", m, "validation", feat, M)
            ids_te, Xte, Ste, _ = load_feats("vqa_rad", m, "test", feat, M)
            X = np.concatenate([Xtr, Xva]); S = np.concatenate([Str, Sva]); groups = np.concatenate([img_tr, img_va])
            b = (S[:, 1:].max(1) > S[:, 0]).astype(np.float32)
            for seed in SEEDS:
                oof = np.zeros(len(X)); oof_act = np.zeros(len(X), int)
                te_gp, te_logits = np.zeros(len(Xte)), np.zeros((len(Xte), 3))
                fold_auc = []
                for k, (tr_idx, ho_idx) in enumerate(GroupKFold(n_splits=5).split(X, b, groups)):
                    pos_w = float((1 - b[tr_idx]).sum() / max(b[tr_idx].sum(), 1))
                    gate = Head(X.shape[1], 1).to(DEV)
                    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=DEV))
                    fit_fixed(gate, X[tr_idx], T(b[tr_idx])[:, None], bce, epochs=60, seed=seed + k)
                    p_ho = 1 / (1 + np.exp(-pred(gate, X[ho_idx])[:, 0])); oof[ho_idx] = p_ho
                    fold_auc.append(roc_auc_score(b[ho_idx], p_ho) if b[ho_idx].sum() else np.nan)
                    te_gp += (1 / (1 + np.exp(-pred(gate, Xte)[:, 0]))) / 5
                    pos = tr_idx[b[tr_idx] == 1]
                    cls = Head(X.shape[1], 3).to(DEV)
                    fit_fixed(cls, X[pos], T(S[pos, 1:].argmax(1), torch.long), nn.CrossEntropyLoss(), epochs=60, seed=seed + k)
                    oof_act[ho_idx] = pred(cls, X[ho_idx]).argmax(1)
                    te_logits += pred(cls, Xte) / 5
                auroc, auprc = roc_auc_score(b, oof), average_precision_score(b, oof)
                tau, rows = tau_rule(S, oof, oof_act, cap=0.05)
                te_act = te_logits.argmax(1)
                if tau is None:
                    a_te = np.zeros(len(Xte), int)
                else:
                    a_te = decide(te_gp, te_act, tau)
                b_te = (Ste[:, 1:].max(1) > Ste[:, 0]).astype(int)
                out_rows.append(dict(model=m, feat=feat, seed=seed, n_trainval=len(X), n_pos=int(b.sum()), oof_auroc=auroc, oof_auprc=auprc,
                                     fold_auroc_mean=np.nanmean(fold_auc), fold_auroc_min=np.nanmin(fold_auc), fold_auroc_max=np.nanmax(fold_auc),
                                     test_gate_auroc=roc_auc_score(b_te, te_gp), test_gate_auprc=average_precision_score(b_te, te_gp),
                                     tau=tau if tau is not None else "abstain", test_score=float(realised(Ste, a_te).mean()), test_base=float(Ste[:, 0].mean()),
                                     test_harm=harm_rate(Ste, a_te), test_rate=float((a_te != 0).mean()),
                                     oof_score_at_tau=[r for r in rows if tau is not None and r["tau"] == tau][0]["score"] if tau is not None else float(S[:, 0].mean())))
                print(out_rows[-1], flush=True)


def experiment_harm_aware(M, out_rows):
    import json
    prim = json.load(open(RESULTS / "revised_primary_config.json"))["cls"]
    r, e, f = prim[:3]
    for ds in ["vqa_rad", "slake"]:
        for m in ["medgemma", "qwen"]:
            va = M[(M.dataset == ds) & (M.model == m) & (M.split == "validation")].set_index("sample_id")
            te = M[(M.dataset == ds) & (M.model == m) & (M.split == "test")].set_index("sample_id")
            Sv, St = [g[[f"{a}_score" for a in ACTIONS]].values.astype(np.float32) for g in (va, te)]
            for seed in SEEDS:
                P = pd.read_csv(RESULTS / "revised_preds" / f"{r}_{e}_{f}_seed{seed}.csv", keep_default_na=False)
                P = P[(P.dataset == ds) & (P.model == m) & (P["head"] == "cls")]
                pv = P[P.split == "validation"].set_index("sample_id").loc[va.index]; pt = P[P.split == "test"].set_index("sample_id").loc[te.index]
                gv, gt = pv.gate_prob.astype(float).values, pt.gate_prob.astype(float).values
                av, at = np.array([A[x] - 1 for x in pv.action_argmax]), np.array([A[x] - 1 for x in pt.action_argmax])
                for cap in [0.05, 0.03, 0.02]:
                    tau, rows = tau_rule(Sv, gv, av, cap)
                    a_te = np.zeros(len(te), int) if tau is None else decide(gt, at, tau)
                    out_rows.append(dict(dataset=ds, model=m, seed=seed, cap=cap, tau=tau if tau is not None else "abstain",
                                         val_harm_at_tau=[x for x in rows if tau is not None and x["tau"] == tau][0]["harm"] if tau is not None else 0.0,
                                         test_score=float(realised(St, a_te).mean()), test_base=float(St[:, 0].mean()), test_harm=harm_rate(St, a_te), test_rate=float((a_te != 0).mean())))


def main():
    M = load_matrix()
    cv, ha = [], []
    experiment_cv(M, cv)
    experiment_harm_aware(M, ha)
    CV = pd.DataFrame(cv); CV.to_csv(RESULTS / "extra_cv_vqarad.csv", index=False)
    HA = pd.DataFrame(ha); HA.to_csv(RESULTS / "extra_harm_aware.csv", index=False)
    out = ["### (i) VQA-RAD 5-fold cross-fitted gate (train+validation, GroupKFold by image; mean ± std over 3 seeds)\n",
           "| VLM | Features | n train+val (pos) | OOF gate AUROC | OOF AUPRC | per-fold AUROC range | test gate AUROC (5-model ensemble) | test AUPRC | tau | test score | base | harm | intervention rate |",
           "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (m, feat), g in CV.groupby(["model", "feat"], sort=False):
        taus = ", ".join(str(t) for t in g.tau)
        out.append(f"| {m} | {feat} | {g.n_trainval.iloc[0]} ({g.n_pos.iloc[0]}) | {g.oof_auroc.mean():.3f} ± {g.oof_auroc.std():.3f} | {g.oof_auprc.mean():.3f} ± {g.oof_auprc.std():.3f} | "
                   f"{g.fold_auroc_min.min():.2f}–{g.fold_auroc_max.max():.2f} | {g.test_gate_auroc.mean():.3f} ± {g.test_gate_auroc.std():.3f} | {g.test_gate_auprc.mean():.3f} | {taus} | "
                   f"{100*g.test_score.mean():.1f} ± {100*g.test_score.std():.1f} | {100*g.test_base.iloc[0]:.1f} | {100*g.test_harm.mean():.1f} | {100*g.test_rate.mean():.1f} |")
    out += ["\n### (ii) Harm-aware tau on the primary gated classifier (validation cap; abstain = never intervene if no tau meets the cap; mean over 3 seeds)\n",
            "| Setting | cap | tau (seeds) | val harm at tau | test score | base | test harm | intervention rate |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for (ds, m, cap), g in HA.groupby(["dataset", "model", "cap"], sort=False):
        out.append(f"| {ds}/{m} | {cap:.0%} | {', '.join(str(t) for t in g.tau)} | {100*g.val_harm_at_tau.mean():.1f} | {100*g.test_score.mean():.1f} ± {100*g.test_score.std():.1f} | {100*g.test_base.iloc[0]:.1f} | {100*g.test_harm.mean():.1f} | {100*g.test_rate.mean():.1f} |")
    (TABLES / "extra_experiment.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
