"""Train the counterfactual utility selector: MLP on frozen BiomedCLIP features -> predicted dQ per action.

Hyper-parameters (hidden, lr, dropout) and the intervention margin tau are selected on the
validation split only (criterion: realised mean score of the selected action). Seeds 0,1,2.
Outputs results/selector_preds/{dataset}_{model}_{feat}_{encoder}_seed{s}.csv  (val + test predictions)
        results/selector_train_log.csv
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTIONS, RESULTS, load_matrix  # noqa: E402

FEATS = {"q": ["z_q"], "iq": ["z_img", "z_q"], "iqy": ["z_img", "z_q", "z_y0"]}
TAUS = [0.0, 0.02, 0.05, 0.1]


class MLP(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Linear(h, len(ACTIONS)))

    def forward(self, x):
        return self.net(x)


def load(ds, m, split, feat, enc_dir, matrix):
    z = np.load(RESULTS / enc_dir / f"{ds}_{m}_{split}.npz")
    ids = list(z["sample_id"])
    X = np.concatenate([z[k] for k in FEATS[feat]], axis=1).astype(np.float32)
    g = matrix[(matrix.dataset == ds) & (matrix.model == m) & (matrix.split == split)].set_index("sample_id")
    keep = [i for i, s in enumerate(ids) if s in g.index]
    ids = [ids[i] for i in keep]
    X = X[keep]
    g = g.loc[ids]
    Y = g[[f"dq_{a}" for a in ACTIONS]].values.astype(np.float32)
    S = g[[f"{a}_score" for a in ACTIONS]].values.astype(np.float32)
    return ids, X, Y, S


def select(pred, tau):
    """argmax over predicted dQ, but intervene only if best predicted gain > tau (KEEP otherwise)."""
    p = pred.copy()
    p[:, 0] = 0.0
    a = p.argmax(1)
    a[p.max(1) <= tau] = 0
    return a


def realised(S, a):
    return float(S[np.arange(len(a)), a].mean())


def train_one(Xtr, Ytr, Xva, Sva, h, lr, p, seed, epochs=300, bs=128, wd=1e-4, dev="cuda"):
    torch.manual_seed(seed); np.random.seed(seed)
    net = MLP(Xtr.shape[1], h, p).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    Xt, Yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
    Xv = torch.tensor(Xva, device=dev)
    best = (-1, None, None, None)
    n = len(Xt)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad():
                pv = net(Xv).cpu().numpy()
            for tau in TAUS:
                r = realised(Sva, select(pv, tau))
                if r > best[0]:
                    best = (r, ep, tau, {k: v.detach().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[3])
    net.eval()
    return net, best[0], best[1], best[2]


def main(datasets, models, feats, encoder, seeds):
    matrix = load_matrix()
    enc_dir = "features" if encoder == "biomedclip" else "features_clip"
    out = RESULTS / "selector_preds"
    out.mkdir(exist_ok=True)
    log = []
    grid = list(itertools.product([256, 512], [1e-3, 3e-4], [0.1, 0.3]))
    for ds in datasets:
        for m in models:
            for feat in feats:
                try:
                    tr = load(ds, m, "train", feat, enc_dir, matrix)
                    va = load(ds, m, "validation", feat, enc_dir, matrix)
                    te = load(ds, m, "test", feat, enc_dir, matrix)
                except FileNotFoundError as e:
                    print("skip", ds, m, feat, e); continue
                # hyper-parameter search on validation with seed 0
                best_cfg, best_val = None, -1
                for h, lr, p in grid:
                    _, r, ep, tau = train_one(tr[1], tr[2], va[1], va[3], h, lr, p, seed=0)
                    log.append(dict(dataset=ds, model=m, feat=feat, encoder=encoder, phase="search", hidden=h, lr=lr, dropout=p, seed=0, val_score=r, epoch=ep, tau=tau))
                    if r > best_val:
                        best_val, best_cfg = r, (h, lr, p)
                h, lr, p = best_cfg
                print(f"{ds}/{m}/{feat}/{encoder}: best cfg hidden={h} lr={lr} drop={p} val={best_val:.4f}", flush=True)
                for seed in seeds:
                    net, r, ep, tau = train_one(tr[1], tr[2], va[1], va[3], h, lr, p, seed=seed)
                    with torch.no_grad():
                        pv = net(torch.tensor(va[1], device="cuda")).cpu().numpy()
                        pt = net(torch.tensor(te[1], device="cuda")).cpu().numpy()
                    rows = []
                    for split, ids, pred in [("validation", va[0], pv), ("test", te[0], pt)]:
                        sel = select(pred, tau)
                        for i, sid in enumerate(ids):
                            rows.append(dict(sample_id=sid, split=split, **{f"pred_dq_{a}": float(pred[i, j]) for j, a in enumerate(ACTIONS)},
                                             selected=ACTIONS[sel[i]]))
                    pd.DataFrame(rows).to_csv(out / f"{ds}_{m}_{feat}_{encoder}_seed{seed}.csv", index=False)
                    log.append(dict(dataset=ds, model=m, feat=feat, encoder=encoder, phase="final", hidden=h, lr=lr, dropout=p, seed=seed, val_score=r, epoch=ep, tau=tau,
                                    test_score=realised(te[3], select(pt, tau))))
                    print(f"  seed {seed}: val={r:.4f} tau={tau} ep={ep} test={log[-1]['test_score']:.4f}", flush=True)
    lp = RESULTS / "selector_train_log.csv"
    df = pd.DataFrame(log)
    if lp.exists():
        old = pd.read_csv(lp)
        old = old[~((old.encoder == encoder) & old.feat.isin(feats) & old.dataset.isin(datasets) & old.model.isin(models))]
        df = pd.concat([old, df])
    df.to_csv(lp, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["vqa_rad", "slake"])
    ap.add_argument("--models", nargs="+", default=["medgemma", "qwen"])
    ap.add_argument("--feats", nargs="+", default=["q", "iq", "iqy"])
    ap.add_argument("--encoder", default="biomedclip", choices=["biomedclip", "clip"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    main(a.datasets, a.models, a.feats, a.encoder, a.seeds)
