"""Baseline methods evaluated under the same folds, inputs and tuning budget as LayerGap.

All baselines receive gene/protein-level blocks; within each training fold the k most variable
features per block are selected (unsupervised, training samples only). Missing blocks at test
time are filled with the training mean (0 after within-cohort z-scoring); this imputation step is
part of each baseline's measured inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class _BlockInput:
    """Shared block handling: in-fold variance filter, scaling, mean-imputation of missing blocks."""

    def __init__(self, blocks=("G", "T", "P"), k=2000):
        self.blocks, self.k = blocks, k

    def fit_blocks(self, D, idx):
        self.cols, self.scalers = {}, {}
        for b in self.blocks:
            X = D[b].reindex(idx)
            X = X.loc[:, X.notna().all()]
            if b != "G" and X.shape[1] > self.k:
                X = X[X.var().sort_values(ascending=False).index[: self.k]]
            self.cols[b] = list(X.columns)
            self.scalers[b] = StandardScaler().fit(X.values)
        return self

    def block_matrix(self, D, idx, b):
        if b not in D:
            return np.zeros((len(idx), len(self.cols[b])))
        X = D[b].reindex(index=idx, columns=self.cols[b])
        Xs = self.scalers[b].transform(X.fillna(pd.Series(self.scalers[b].mean_, index=self.cols[b])).values)
        return np.nan_to_num(Xs)

    def concat(self, D, idx):
        return np.hstack([self.block_matrix(D, idx, b) for b in self.blocks])


def _inner_select(make, grid, X, y, folds=3, seed=0):
    if len(grid) == 1:
        return grid[0]
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = []
    for g in grid:
        s = []
        for a, b in skf.split(X, y):
            m = make(g).fit(X[a], y[a])
            s.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
        scores.append(np.mean(s))
    return grid[int(np.argmax(scores))]


class ElasticNetLR(_BlockInput):
    name = "EN-concat"

    def __init__(self, blocks=("G", "T", "P"), k=2000, grid=(0.02, 0.1, 0.5), seed=0):
        super().__init__(blocks, k)
        self.grid, self.seed = grid, seed

    def _make(self, C):
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=C, max_iter=3000, tol=1e-3)

    def fit(self, D, idx, y):
        self.fit_blocks(D, idx)
        X = self.concat(D, idx)
        C = _inner_select(self._make, self.grid, X, y, seed=self.seed)
        self.model = self._make(C).fit(X, y)
        return self

    def predict(self, D, idx):
        return self.model.predict_proba(self.concat(D, idx))[:, 1]


class RandomForest(_BlockInput):
    name = "RF-concat"

    def __init__(self, blocks=("G", "T", "P"), k=2000, grid=("sqrt", 0.05), seed=0):
        super().__init__(blocks, k)
        self.grid, self.seed = grid, seed

    def _make(self, mf):
        return RandomForestClassifier(n_estimators=500, max_features=mf, min_samples_leaf=2, n_jobs=2,
                                      random_state=self.seed, class_weight="balanced")

    def fit(self, D, idx, y):
        self.fit_blocks(D, idx)
        X = self.concat(D, idx)
        self.model = self._make(_inner_select(self._make, self.grid, X, y, seed=self.seed)).fit(X, y)
        return self

    def predict(self, D, idx):
        return self.model.predict_proba(self.concat(D, idx))[:, 1]


class XGBoost(_BlockInput):
    name = "XGB-concat"

    def __init__(self, blocks=("G", "T", "P"), k=2000, grid=(2, 4), seed=0):
        super().__init__(blocks, k)
        self.grid, self.seed = grid, seed

    def _make(self, depth):
        return XGBClassifier(n_estimators=300, max_depth=depth, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.3, n_jobs=2, random_state=self.seed, verbosity=0)

    def fit(self, D, idx, y):
        self.fit_blocks(D, idx)
        X = self.concat(D, idx)
        self.model = self._make(_inner_select(self._make, self.grid, X, y, seed=self.seed)).fit(X, y)
        return self

    def predict(self, D, idx):
        return self.model.predict_proba(self.concat(D, idx))[:, 1]


class LateFusionLR(_BlockInput):
    name = "Late-fusion LR"

    def __init__(self, blocks=("G", "T", "P"), k=2000, grid=(0.001, 0.01, 0.1), seed=0):
        super().__init__(blocks, k)
        self.grid, self.seed = grid, seed

    def _make(self, C):
        return LogisticRegression(C=C, max_iter=3000)

    def fit(self, D, idx, y):
        self.fit_blocks(D, idx)
        self.models = {}
        for b in self.blocks:
            X = self.block_matrix(D, idx, b)
            self.models[b] = self._make(_inner_select(self._make, self.grid, X, y, seed=self.seed)).fit(X, y)
        return self

    def predict(self, D, idx):
        ps, w = np.zeros(len(idx)), np.zeros(len(idx))
        for b, m in self.models.items():
            if b in D:
                avail = D[b].reindex(idx).notna().any(axis=1).values.astype(float)
                ps += avail * m.predict_proba(self.block_matrix(D, idx, b))[:, 1]
                w += avail
        return np.where(w > 0, ps / np.maximum(w, 1), 0.5)


class MBPLSDA(_BlockInput):
    """Multi-block PLS-DA in the spirit of DIABLO: blocks scaled by 1/sqrt(p_b), PLS on the
    class indicator, logistic read-out on the latent components."""
    name = "MB-PLS-DA"

    def __init__(self, blocks=("G", "T", "P"), k=2000, grid=(1, 2, 3), seed=0):
        super().__init__(blocks, k)
        self.grid, self.seed = grid, seed

    def _weighted(self, D, idx):
        return np.hstack([self.block_matrix(D, idx, b) / np.sqrt(max(len(self.cols[b]), 1)) for b in self.blocks])

    def fit(self, D, idx, y):
        self.fit_blocks(D, idx)
        X = self._weighted(D, idx)

        class _M:
            def __init__(s, nc):
                s.nc = nc

            def fit(s, X, y):
                s.pls = PLSRegression(n_components=s.nc, scale=False).fit(X, y)
                s.lr = LogisticRegression(max_iter=2000).fit(s.pls.transform(X), y)
                return s

            def predict_proba(s, X):
                return s.lr.predict_proba(s.pls.transform(X))

        self.model = _M(_inner_select(lambda nc: _M(nc), self.grid, X, y, seed=self.seed)).fit(X, y)
        return self

    def predict(self, D, idx):
        return self.model.predict_proba(self._weighted(D, idx))[:, 1]


# ----------------------------------------------------------------------------- deep baselines
def _torch_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _cos_adj(Xa, Xb, k):
    """Cosine-similarity adjacency (rows Xa, cols Xb) keeping edges above the threshold that gives
    on average k neighbours per node, as in MOGONET."""
    A = F.normalize(Xa, dim=1) @ F.normalize(Xb, dim=1).T
    thr = torch.sort(A.flatten(), descending=True).values[min(A.numel() - 1, k * A.shape[0])]
    A = torch.where(A >= thr, A, torch.zeros_like(A))
    return A


def _norm_adj(A):
    A = A + torch.eye(A.shape[0])
    d = A.sum(1)
    return A / d[:, None]


class _GCN(nn.Module):
    def __init__(self, din, h=(200, 100), drop=0.5):
        super().__init__()
        self.w1, self.w2 = nn.Linear(din, h[0]), nn.Linear(h[0], h[1])
        self.cls = nn.Linear(h[1], 2)
        self.drop = drop

    def forward(self, X, A):
        h = F.leaky_relu(A @ self.w1(X), 0.25)
        h = F.dropout(h, self.drop, self.training)
        h = F.leaky_relu(A @ self.w2(h), 0.25)
        return self.cls(h)


class _VCDN(nn.Module):
    def __init__(self, n_views, h=16):
        super().__init__()
        self.n_views = n_views
        self.net = nn.Sequential(nn.Linear(2 ** n_views, h), nn.LeakyReLU(0.25), nn.Linear(h, 2))

    def forward(self, logits):
        ps = [torch.sigmoid(l) if False else F.softmax(l, 1) for l in logits]
        x = ps[0]
        for p in ps[1:]:
            x = torch.einsum("ni,nj->nij", x, p).reshape(x.shape[0], -1)
        return self.net(x)


class MOGONET(_BlockInput):
    """Re-implementation of MOGONET (Wang et al., 2021): one GCN per omics on a sample-similarity
    graph plus a View Correlation Discovery Network for cross-omics fusion."""
    name = "MOGONET"

    def __init__(self, blocks=("G", "T", "P"), k=500, k_adj=10, epochs_pre=200, epochs=300, lr=1e-3,
                 grid=(1e-4, 1e-3), seed=0):
        super().__init__(blocks, k)
        self.k_adj, self.epochs_pre, self.epochs, self.lr, self.grid, self.seed = k_adj, epochs_pre, epochs, lr, grid, seed

    def _train(self, Xs, y, wd):
        _torch_seed(self.seed)
        yt = torch.tensor(y, dtype=torch.long)
        As = [_norm_adj(_cos_adj(X, X, self.k_adj)) for X in Xs]
        gcns = [_GCN(X.shape[1]) for X in Xs]
        vcdn = _VCDN(len(Xs))
        for g, X, A in zip(gcns, Xs, As):
            opt = torch.optim.Adam(g.parameters(), lr=self.lr, weight_decay=wd)
            for _ in range(self.epochs_pre):
                opt.zero_grad(); F.cross_entropy(g(X, A), yt).backward(); opt.step()
        params = [p for g in gcns for p in g.parameters()] + list(vcdn.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=wd)
        for _ in range(self.epochs):
            opt.zero_grad()
            outs = [g(X, A) for g, X, A in zip(gcns, Xs, As)]
            loss = sum(F.cross_entropy(o, yt) for o in outs) + F.cross_entropy(vcdn(outs), yt)
            loss.backward(); opt.step()
        return gcns, vcdn

    def _infer(self, gcns, vcdn, Xtr, Xte):
        outs = []
        for g, Xa, Xb in zip(gcns, Xtr, Xte):
            Xall = torch.cat([Xa, Xb])
            n = Xa.shape[0]
            A = torch.zeros(Xall.shape[0], Xall.shape[0])
            A[:n, :n] = _cos_adj(Xa, Xa, self.k_adj)
            cross = _cos_adj(Xb, Xa, self.k_adj)
            A[n:, :n], A[:n, n:] = cross, cross.T
            g.eval()
            with torch.no_grad():
                outs.append(g(Xall, _norm_adj(A))[n:])
        with torch.no_grad():
            return F.softmax(vcdn(outs), 1)[:, 1].numpy()

    def _tensors(self, D, idx):
        return [torch.tensor(self.block_matrix(D, idx, b), dtype=torch.float32) for b in self.blocks]

    def fit(self, D, idx, y):
        self.fit_blocks(D, idx)
        Xs = self._tensors(D, idx)
        y = np.asarray(y)
        if len(self.grid) > 1:
            skf = StratifiedKFold(3, shuffle=True, random_state=self.seed)
            sc = []
            for wd in self.grid:
                s = []
                for a, b in skf.split(Xs[0], y):
                    gc, vc = self._train([X[a] for X in Xs], y[a], wd)
                    s.append(roc_auc_score(y[b], self._infer(gc, vc, [X[a] for X in Xs], [X[b] for X in Xs])))
                sc.append(np.mean(s))
            wd = self.grid[int(np.argmax(sc))]
        else:
            wd = self.grid[0]
        self.Xtr = Xs
        self.gcns, self.vcdn = self._train(Xs, y, wd)
        return self

    def predict(self, D, idx):
        return self._infer(self.gcns, self.vcdn, self.Xtr, self._tensors(D, idx))


def _ds_combine(a1, a2):
    """Reduced Dempster-Shafer combination of two Dirichlet opinions (Han et al., TMC)."""
    K = 2
    S1, S2 = a1.sum(1, keepdim=True), a2.sum(1, keepdim=True)
    b1, b2 = (a1 - 1) / S1, (a2 - 1) / S2
    u1, u2 = K / S1, K / S2
    bb = torch.einsum("ni,nj->nij", b1, b2)
    conflict = bb.sum((1, 2)) - torch.diagonal(bb, dim1=1, dim2=2).sum(1)
    scale = (1 - conflict).unsqueeze(1)
    b = (b1 * b2 + b1 * u2 + b2 * u1) / scale
    u = (u1 * u2) / scale
    S = K / u
    return b * S + 1


def _edl_loss(alpha, y, epoch, anneal):
    S = alpha.sum(1, keepdim=True)
    yo = F.one_hot(y, 2).float()
    A = (yo * (torch.digamma(S) - torch.digamma(alpha))).sum(1)
    at = yo + (1 - yo) * alpha
    St = at.sum(1, keepdim=True)
    kl = (torch.lgamma(St).squeeze(1) - torch.lgamma(torch.tensor(2.0)) - torch.lgamma(at).sum(1)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St))).sum(1))
    return (A + min(1.0, epoch / anneal) * kl).mean()


class TMC(_BlockInput):
    """Re-implementation of Trusted Multi-view Classification (Han et al., IEEE TPAMI 2023):
    per-view evidential networks combined with the reduced Dempster-Shafer rule."""
    name = "TMC"

    def __init__(self, blocks=("G", "T", "P"), k=500, hidden=64, epochs=300, lr=1e-3, anneal=50,
                 grid=(1e-4, 1e-3), seed=0):
        super().__init__(blocks, k)
        self.hidden, self.epochs, self.lr, self.anneal, self.grid, self.seed = hidden, epochs, lr, anneal, grid, seed

    def _nets(self, dims):
        return nn.ModuleList([nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(), nn.Dropout(0.3),
                                            nn.Linear(self.hidden, 2), nn.Softplus()) for d in dims])

    def _combine(self, nets, Xs):
        alphas = [n(X) + 1 for n, X in zip(nets, Xs)]
        a = alphas[0]
        for b in alphas[1:]:
            a = _ds_combine(a, b)
        return alphas, a

    def _train(self, Xs, y, wd):
        _torch_seed(self.seed)
        nets = self._nets([X.shape[1] for X in Xs])
        opt = torch.optim.Adam(nets.parameters(), lr=self.lr, weight_decay=wd)
        yt = torch.tensor(y, dtype=torch.long)
        for ep in range(1, self.epochs + 1):
            nets.train(); opt.zero_grad()
            alphas, a = self._combine(nets, Xs)
            loss = sum(_edl_loss(x, yt, ep, self.anneal) for x in alphas) + _edl_loss(a, yt, ep, self.anneal)
            loss.backward(); opt.step()
        return nets

    def _infer(self, nets, Xs):
        nets.eval()
        with torch.no_grad():
            _, a = self._combine(nets, Xs)
            return (a[:, 1] / a.sum(1)).numpy()

    def _tensors(self, D, idx):
        return [torch.tensor(self.block_matrix(D, idx, b), dtype=torch.float32) for b in self.blocks]

    def fit(self, D, idx, y):
        self.fit_blocks(D, idx)
        Xs = self._tensors(D, idx)
        y = np.asarray(y)
        skf = StratifiedKFold(3, shuffle=True, random_state=self.seed)
        sc = []
        for wd in self.grid:
            s = []
            for a, b in skf.split(Xs[0], y):
                n = self._train([X[a] for X in Xs], y[a], wd)
                s.append(roc_auc_score(y[b], self._infer(n, [X[b] for X in Xs])))
            sc.append(np.mean(s))
        self.nets = self._train(Xs, y, self.grid[int(np.argmax(sc))])
        return self

    def predict(self, D, idx):
        return self._infer(self.nets, self._tensors(D, idx))


def all_baselines(blocks=("G", "T", "P"), seed=0):
    rna_only = ElasticNetLR(blocks=("T",), seed=seed)
    rna_only.name = "EN-RNA only"
    return [rna_only, ElasticNetLR(blocks, seed=seed), RandomForest(blocks, seed=seed), XGBoost(blocks, seed=seed),
            LateFusionLR(blocks, seed=seed), MBPLSDA(blocks, seed=seed), MOGONET(blocks, seed=seed),
            TMC(blocks, seed=seed)]
