"""Revised two-stage selective intervention model (trained on the existing predictions/features).

Stage 1  gate  g(x) = P(intervention beneficial | x), b = 1[max_{a!=KEEP} Q_a > Q_KEEP + eps], class-weighted BCE.
Stage 2a action classifier h(x) = P(a* | x) over {REPROMPT, ENHANCE, ZOOM}, 3-way CE on samples with b = 1.
Stage 2b utility head  u(x) -> dQ for the three interventions, SmoothL1 on all training samples.
Decision: KEEP if g(x) < tau else argmax over the three interventions.

eps in {0, .05, .1} and tau in {.1..9} are selected on validation only (per setting), objective:
max validation score subject to validation harmful-intervention rate <= HARM_CAP (else max score).
Regimes: per-setting (one model per dataset x VLM) and pooled (all four, with learned dataset / VLM embeddings).
Outputs: results/revised_preds/{regime}_{encoder}_{feat}_seed{s}.csv, results/revised_train_log.csv, results/revised_tradeoff.csv
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTIONS, RESULTS, load_matrix  # noqa: E402

FEATS = {"q": ["z_q"], "iq": ["z_img", "z_q"], "iqy": ["z_img", "z_q", "z_y0"]}
EPSS = [0.0, 0.05, 0.10]
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HARM_CAP = 0.05
SETTINGS = [(d, m) for d in ["vqa_rad", "slake"] for m in ["medgemma", "qwen"]]
SID = {s: i for i, s in enumerate(SETTINGS)}
INTERV = ACTIONS[1:]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    def __init__(self, d, out, h=256, p=0.2, pooled=False):
        super().__init__()
        self.pooled = pooled
        if pooled:
            self.emb_ds, self.emb_vlm = nn.Embedding(2, 8), nn.Embedding(2, 8)
            d += 16
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Linear(h, out))

    def forward(self, x, sid=None):
        if self.pooled:
            x = torch.cat([x, self.emb_ds(sid // 2), self.emb_vlm(sid % 2)], 1)
        return self.net(x)


def load_split(ds, m, split, feat, enc_dir, M):
    z = np.load(RESULTS / enc_dir / f"{ds}_{m}_{split}.npz")
    ids = list(z["sample_id"])
    X = np.concatenate([z[k] for k in FEATS[feat]], 1).astype(np.float32)
    g = M[(M.dataset == ds) & (M.model == m) & (M.split == split)].set_index("sample_id")
    keep = [i for i, s in enumerate(ids) if s in g.index]
    ids = [ids[i] for i in keep]
    g = g.loc[ids]
    S = g[[f"{a}_score" for a in ACTIONS]].values.astype(np.float32)
    return dict(ids=ids, X=X[keep], S=S, sid=np.full(len(ids), SID[(ds, m)]))


def cat(parts):
    return dict(ids=sum([p["ids"] for p in parts], []), X=np.concatenate([p["X"] for p in parts]),
                S=np.concatenate([p["S"] for p in parts]), sid=np.concatenate([p["sid"] for p in parts]))


def T(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=DEV)


def fit(model, X, sid, Y, loss_fn, lr, seed, val_fn, epochs=200, bs=128, wd=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt, St, Yt = T(X), T(sid, torch.long), Y
    best = (-np.inf, None, -1)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(Xt[idx], St[idx]), Yt[idx]).backward()
            opt.step()
        if ep % 5 == 4:
            model.eval()
            v = val_fn(model)
            if v > best[0]:
                best = (v, {k: t.detach().clone() for k, t in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); model.eval()
    return best[0], best[2]


@torch.no_grad()
def predict(model, X, sid):
    return model(T(X), T(sid, torch.long)).cpu().numpy()


def decide(gate_p, act, tau):
    a = np.where(gate_p >= tau, act + 1, 0)
    return a


def realised(S, a):
    return S[np.arange(len(a)), a]


def harm_rate(S, a):
    ch = a != 0
    if not ch.any():
        return 0.0
    r = realised(S, a)
    return float((r[ch] < S[ch, 0]).mean())


def pick_tau(S, gate_p, act):
    """validation objective: max score s.t. harmful rate <= HARM_CAP; else max score."""
    rows = []
    for tau in TAUS:
        a = decide(gate_p, act, tau)
        rows.append(dict(tau=tau, score=realised(S, a).mean(), harm=harm_rate(S, a), rate=float((a != 0).mean())))
    ok = [r for r in rows if r["harm"] <= HARM_CAP] or rows
    best = max(ok, key=lambda r: r["score"])
    return best["tau"], rows


def train_regime(regime, encoder, feat, seed, M, log, tradeoff, preds_out):
    enc_dir = "features" if encoder == "biomedclip" else "features_clip"
    data = {(d, m): {sp: load_split(d, m, sp, feat, enc_dir, M) for sp in ["train", "validation", "test"]} for d, m in SETTINGS}
    groups = [[s] for s in SETTINGS] if regime == "per_setting" else [SETTINGS]
    pooled = regime == "pooled"
    out_rows = []
    for grp in groups:
        tr, va = cat([data[s]["train"] for s in grp]), cat([data[s]["validation"] for s in grp])
        d = tr["X"].shape[1]
        per_eps = {}
        for eps in EPSS:
            b_tr = (tr["S"][:, 1:].max(1) > tr["S"][:, 0] + eps).astype(np.float32)
            b_va = (va["S"][:, 1:].max(1) > va["S"][:, 0] + eps).astype(np.float32)
            pos_w = float((1 - b_tr).sum() / max(b_tr.sum(), 1))
            # ---- gate (lr chosen on validation AUPRC)
            best_g = (-1, None, None)
            for lr in [1e-3, 3e-4]:
                gate = Head(d, 1, pooled=pooled).to(DEV)
                bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=DEV))
                auprc, ep = fit(gate, tr["X"], tr["sid"], T(b_tr)[:, None], lambda o, y: bce(o, y), lr, seed,
                                lambda mdl: average_precision_score(b_va, predict(mdl, va["X"], va["sid"])[:, 0]) if b_va.sum() else 0.0)
                if auprc > best_g[0]:
                    best_g = (auprc, gate, dict(lr=lr, epoch=ep))
            gate = best_g[1]
            # ---- action classifier on beneficial training samples (target = best intervention)
            pos = b_tr == 1
            y_cls = tr["S"][pos, 1:].argmax(1)
            vpos = b_va == 1
            v_cls = va["S"][vpos, 1:].argmax(1)
            cls = Head(d, 3, pooled=pooled).to(DEV)
            ce = nn.CrossEntropyLoss()
            cls_acc, _ = fit(cls, tr["X"][pos], tr["sid"][pos], T(y_cls, torch.long), lambda o, y: ce(o, y), 1e-3, seed,
                             lambda mdl: float((predict(mdl, va["X"][vpos], va["sid"][vpos]).argmax(1) == v_cls).mean()) if vpos.any() else 0.0)
            # ---- utility head on all training samples (dQ of the three interventions)
            dq_tr = tr["S"][:, 1:] - tr["S"][:, :1]
            util = Head(d, 3, pooled=pooled).to(DEV)
            sl1 = nn.SmoothL1Loss(beta=0.1)
            util_v, _ = fit(util, tr["X"], tr["sid"], T(dq_tr), lambda o, y: sl1(o, y), 1e-3, seed,
                            lambda mdl: float(realised(va["S"][vpos], predict(mdl, va["X"][vpos], va["sid"][vpos]).argmax(1) + 1).mean()) if vpos.any() else 0.0)
            per_eps[eps] = dict(gate=gate, cls=cls, util=util, pos_w=pos_w, gate_auprc=best_g[0], gate_cfg=best_g[2], cls_acc=cls_acc, util_v=util_v,
                                pos_rate_train=float(b_tr.mean()), pos_rate_val=float(b_va.mean()))
        # ---- per-setting selection of (eps, tau) on that setting's validation split, then test predictions
        for s in grp:
            vd, td = data[s]["validation"], data[s]["test"]
            choice = {}
            for head in ["cls", "util"]:
                best = (-np.inf, None)
                for eps, P in per_eps.items():
                    gp = 1 / (1 + np.exp(-predict(P["gate"], vd["X"], vd["sid"])[:, 0]))
                    act = predict(P[head], vd["X"], vd["sid"]).argmax(1)
                    tau, rows = pick_tau(vd["S"], gp, act)
                    sc = [r for r in rows if r["tau"] == tau][0]
                    key = sc["score"] if sc["harm"] <= HARM_CAP else sc["score"] - 1  # prefer eps that meets the cap
                    for r in rows:
                        tradeoff.append(dict(regime=regime, encoder=encoder, feat=feat, seed=seed, dataset=s[0], model=s[1], head=head, eps=eps, split="validation", **r))
                    if key > best[0]:
                        best = (key, dict(eps=eps, tau=tau, val_score=sc["score"], val_harm=sc["harm"], val_rate=sc["rate"]))
                choice[head] = best[1]
            rec = dict(regime=regime, encoder=encoder, feat=feat, seed=seed, dataset=s[0], model=s[1])
            for head in ["cls", "util"]:
                c = choice[head]; P = per_eps[c["eps"]]
                rec.update({f"{head}_eps": c["eps"], f"{head}_tau": c["tau"], f"{head}_val_score": c["val_score"], f"{head}_val_harm": c["val_harm"],
                            f"{head}_gate_auprc_pooledval": P["gate_auprc"], f"{head}_gate_lr": P["gate_cfg"]["lr"], f"{head}_pos_w": P["pos_w"],
                            f"{head}_pos_rate_train": P["pos_rate_train"]})
                for split, D in [("validation", vd), ("test", td)]:
                    gp = 1 / (1 + np.exp(-predict(P["gate"], D["X"], D["sid"])[:, 0]))
                    logits = predict(P[head], D["X"], D["sid"])
                    act = logits.argmax(1)
                    a = decide(gp, act, c["tau"])
                    if split == "test":
                        rec[f"{head}_test_score"] = float(realised(D["S"], a).mean()); rec[f"{head}_test_harm"] = harm_rate(D["S"], a); rec[f"{head}_test_rate"] = float((a != 0).mean())
                        for tau in TAUS:
                            aa = decide(gp, act, tau)
                            tradeoff.append(dict(regime=regime, encoder=encoder, feat=feat, seed=seed, dataset=s[0], model=s[1], head=head, eps=c["eps"], split="test",
                                                 tau=tau, score=float(realised(D["S"], aa).mean()), harm=harm_rate(D["S"], aa), rate=float((aa != 0).mean())))
                    for i, sid_ in enumerate(D["ids"]):
                        out_rows.append(dict(sample_id=sid_, dataset=s[0], model=s[1], split=split, head=head, eps=c["eps"], tau=c["tau"], gate_prob=float(gp[i]),
                                             action_argmax=INTERV[act[i]], selected=ACTIONS[a[i]],
                                             **{f"out_{x}": float(logits[i, j]) for j, x in enumerate(INTERV)}))
            # gate probabilities under eps = 0 for the gate-quality metrics (fixed reference labelling)
            P0 = per_eps[0.0]
            for split, D in [("validation", vd), ("test", td)]:
                gp0 = 1 / (1 + np.exp(-predict(P0["gate"], D["X"], D["sid"])[:, 0]))
                for i, sid_ in enumerate(D["ids"]):
                    out_rows.append(dict(sample_id=sid_, dataset=s[0], model=s[1], split=split, head="gate_eps0", eps=0.0, tau=np.nan, gate_prob=float(gp0[i]),
                                         action_argmax="", selected="", out_reprompt=np.nan, out_enhance=np.nan, out_zoom=np.nan))
            log.append(rec)
            print(f"{regime}/{encoder}/{feat}/seed{seed} {s[0]}/{s[1]}: cls eps={rec['cls_eps']} tau={rec['cls_tau']} val={rec['cls_val_score']:.4f} test={rec['cls_test_score']:.4f} "
                  f"harm={rec['cls_test_harm']:.3f} rate={rec['cls_test_rate']:.2f} | util eps={rec['util_eps']} tau={rec['util_tau']} test={rec['util_test_score']:.4f} harm={rec['util_test_harm']:.3f}", flush=True)
    pd.DataFrame(out_rows).to_csv(preds_out / f"{regime}_{encoder}_{feat}_seed{seed}.csv", index=False)


def main(a):
    M = load_matrix()
    preds_out = RESULTS / "revised_preds"
    preds_out.mkdir(exist_ok=True)
    log, tradeoff = [], []
    for regime, encoder, feat, seed in itertools.product(a.regimes, a.encoders, a.feats, a.seeds):
        if encoder == "clip" and feat != "iqy":
            continue
        train_regime(regime, encoder, feat, seed, M, log, tradeoff, preds_out)
    for name, rows in [("revised_train_log.csv", log), ("revised_tradeoff.csv", tradeoff)]:
        p = RESULTS / name
        df = pd.DataFrame(rows)
        if p.exists() and not a.fresh:
            old = pd.read_csv(p)
            key = ["regime", "encoder", "feat", "seed"]
            new_keys = set(map(tuple, df[key].drop_duplicates().values))
            old = old[~old[key].apply(tuple, axis=1).isin(new_keys)]
            df = pd.concat([old, df])
        df.to_csv(p, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["per_setting", "pooled"])
    ap.add_argument("--encoders", nargs="+", default=["biomedclip", "clip"])
    ap.add_argument("--feats", nargs="+", default=["q", "iq", "iqy"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fresh", action="store_true")
    main(ap.parse_args())
