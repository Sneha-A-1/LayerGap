"""LayerGap: pathway-constrained cross-layer uncoupling with precision-weighted fusion.

Components (each can be switched off for ablation):
  1. Pathway-constrained layer scores  (t_p, p_p, m_p, g_p) computed on genes measured in both
     RNA and protein, so that transcript and protein summaries of a pathway refer to the same genes.
  2. Cascade uncoupling features       u_p = mean_g (P_g - b_g T_g)  and  v_p = m_p - c_p p_p,
     where b_g and c_p are estimated on training samples only.
  3. Precision-weighted adaptive fusion of block-level predictors; sample-specific precision is
     the inverse bootstrap variance of each block's logit, and missing blocks receive zero weight.
  4. Mechanism-aware pathway prioritisation (MAP score) combining effect size, bootstrap sign
     stability and cross-layer sign concordance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BLOCK_ORDER = ["G", "T", "P", "U", "M", "V"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class PathwayEncoder:
    """Turns gene/protein/metabolite matrices into pathway-level block features."""

    def __init__(self, pathways, min_genes=5, min_mets=2, use_uncoupling=True, pathway_level=True,
                 genomic_cols=None):
        self.pathways = pathways
        self.min_genes = min_genes
        self.min_mets = min_mets
        self.use_uncoupling = use_uncoupling
        self.pathway_level = pathway_level
        self.genomic_cols = genomic_cols

    def fit(self, D, idx):
        T, P = D.get("T"), D.get("P")
        self.shared = sorted(set(T.columns) & set(P.columns))
        tr = [i for i in idx if i in T.index and i in P.index]
        Tt, Pt = T.loc[tr, self.shared], P.loc[tr, self.shared]
        # gene-wise transcript->protein slope on z-scored data (= Pearson r), shrunk towards 0
        r = ((Tt - Tt.mean()) * (Pt - Pt.mean())).mean() / (Tt.std(ddof=0) * Pt.std(ddof=0) + 1e-9)
        n = len(tr)
        self.beta = (r * n / (n + 10.0)).fillna(0.0)
        sh = set(self.shared)
        self.gene_sets = {k: [g for g in v["genes"] if g in sh] for k, v in self.pathways.items()}
        self.gene_sets = {k: v for k, v in self.gene_sets.items() if len(v) >= self.min_genes}
        if not self.pathway_level:  # ablation: gene-level blocks restricted to pathway genes
            genes = sorted(set().union(*self.gene_sets.values()))
            var = Tt[genes].var().sort_values(ascending=False)
            self.gene_level = list(var.index[:400])
        M = D.get("M")
        self.met_sets = {}
        if M is not None:
            mcols = set(M.columns)
            for k, v in self.pathways.items():
                ms = [m for m in v["mets"] if m in mcols]
                if len(ms) >= self.min_mets and k in self.gene_sets:
                    self.met_sets[k] = ms
            measured = set(M.index[M.notna().any(axis=1)])
            trm = [i for i in tr if i in measured]
            self.gamma = {}
            if len(trm) >= 10:
                mp = self._met_scores(M.loc[trm])
                pp = self._mean_scores(P.loc[trm], {k: self.gene_sets[k] for k in self.met_sets})
                for k in self.met_sets:
                    a, b = mp[k], pp[k]
                    rr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
                    self.gamma[k] = rr * len(trm) / (len(trm) + 10.0)
        return self

    def _mean_scores(self, X, sets):
        return pd.DataFrame({k: X[v].mean(axis=1) for k, v in sets.items()}, index=X.index)

    def _met_scores(self, M):
        return pd.DataFrame({k: M[v].mean(axis=1) for k, v in self.met_sets.items()}, index=M.index)

    def transform(self, D, idx):
        """Return {block: DataFrame(idx x features)} with NaN rows for missing layers."""
        out = {}
        G = D.get("G")
        if G is not None:
            g = G.reindex(idx)
            mut = D.get("MUT")
            if mut is not None and self.pathway_level:
                # pathway mutation burden = mutated pathway genes / pathway size; genes absent from the
                # mutation matrix of a sequenced sample are unmutated; unsequenced samples stay NaN
                sequenced = pd.Index(idx).isin(mut.index)
                mm = mut.reindex(idx)
                burden = {}
                for k, v in self.pathways.items():
                    cols = [c for c in v["genes"] if c in mm.columns]
                    val = mm[cols].fillna(0).sum(axis=1) / len(v["genes"]) if cols else pd.Series(0.0, index=idx)
                    burden["mb:" + k] = val.where(sequenced, np.nan)
                g = pd.concat([g, pd.DataFrame(burden, index=idx)], axis=1)
            out["G"] = g
        T, P = D.get("T"), D.get("P")
        if not self.pathway_level:
            if T is not None:
                out["T"] = T.reindex(idx)[[c for c in self.gene_level if c in T.columns]].reindex(columns=self.gene_level)
            if P is not None:
                out["P"] = P.reindex(idx)[[c for c in self.gene_level if c in P.columns]].reindex(columns=self.gene_level)
            return out
        if T is not None:
            Tx = T.reindex(idx)
            avail = {k: [g for g in v if g in Tx.columns] for k, v in self.gene_sets.items()}
            out["T"] = pd.DataFrame({k: Tx[v].mean(axis=1) if len(v) >= self.min_genes else np.nan
                                     for k, v in avail.items()}, index=idx)
        if P is not None:
            Px = P.reindex(idx)
            out["P"] = self._mean_scores(Px, self.gene_sets)
            if self.use_uncoupling and T is not None:
                Tx = T.reindex(idx)[self.shared]
                R = Px[self.shared] - Tx * self.beta.values
                out["U"] = self._mean_scores(R, self.gene_sets)
        M = D.get("M")
        if M is not None and self.met_sets:
            mp = self._met_scores(M.reindex(idx))
            out["M"] = mp
            if self.use_uncoupling and self.gamma and P is not None:
                pp = self._mean_scores(P.reindex(idx), {k: self.gene_sets[k] for k in self.met_sets})
                out["V"] = pd.DataFrame({k: mp[k] - self.gamma[k] * pp[k] for k in self.gamma}, index=idx)
        return out


class BlockModel:
    """Bootstrap ensemble of L2-logistic models for one block; gives logit mean and variance."""

    def __init__(self, Cs=(0.01, 0.1, 1.0), n_boot=20, inner_folds=4, seed=0):
        self.Cs, self.n_boot, self.inner_folds, self.seed = Cs, n_boot, inner_folds, seed

    def _choose_C(self, X, y):
        if len(self.Cs) == 1:
            return self.Cs[0]
        skf = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        best, best_auc = self.Cs[0], -1
        for C in self.Cs:
            aucs = []
            for a, b in skf.split(X, y):
                m = LogisticRegression(C=C, max_iter=2000).fit(X[a], y[a])
                aucs.append(roc_auc_score(y[b], m.decision_function(X[b])))
            if np.mean(aucs) > best_auc:
                best, best_auc = C, np.mean(aucs)
        return best

    def fit(self, X, y, C=None):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.C = C if C is not None else self._choose_C(Xs, y)
        rng = np.random.default_rng(self.seed)
        self.models = []
        n = len(y)
        for _ in range(self.n_boot):
            for _try in range(10):
                b = rng.integers(0, n, n)
                if len(np.unique(y[b])) == 2:
                    break
            self.models.append(LogisticRegression(C=self.C, max_iter=2000).fit(Xs[b], y[b]))
        self.coef_ = np.array([m.coef_[0] for m in self.models])
        return self

    def predict_logit(self, X):
        Xs = self.scaler.transform(X)
        Z = np.array([m.decision_function(Xs) for m in self.models])
        return Z.mean(axis=0), Z.var(axis=0)


class LayerGap:
    def __init__(self, pathways, blocks=("G", "T", "P", "U", "M", "V"), min_genes=5, min_mets=2,
                 n_boot=20, n_boot_inner=10, tau=0.1, Cs=(0.01, 0.1, 1.0), inner_folds=4,
                 use_uncoupling=True, use_precision=True, pathway_level=True, stack_l2=1.0, seed=0):
        self.pathways = pathways
        self.blocks = blocks
        self.min_genes, self.min_mets = min_genes, min_mets
        self.n_boot, self.n_boot_inner, self.tau, self.Cs = n_boot, n_boot_inner, tau, Cs
        self.inner_folds = inner_folds
        self.use_uncoupling, self.use_precision, self.pathway_level = use_uncoupling, use_precision, pathway_level
        self.stack_l2, self.seed = stack_l2, seed

    # ----------------------------------------------------------------- helpers
    max_missing_features = 0.2  # a block is used for a sample if <=20% of its features are missing

    @staticmethod
    def _complete(Xdf):
        ok = Xdf.notna().all(axis=1).values
        return ok

    def _usable(self, Xdf, b):
        """Rows with at most max_missing_features missing; remaining gaps get the training mean."""
        frac = Xdf.isna().mean(axis=1).values
        ok = frac <= self.max_missing_features
        X = Xdf.fillna(pd.Series(self.block_models[b].scaler.mean_, index=Xdf.columns))
        return X, ok

    def _prepare_block(self, Xdf, train=False):
        if train:
            keep = Xdf.notna().mean() > 0.0
            Xdf = Xdf.loc[:, keep]
            var = Xdf.var()
            self._cols[self._cur] = list(var.index[var > 1e-12])
        return Xdf.reindex(columns=self._cols[self._cur])

    def _precision(self, b, var):
        if not self.use_precision:
            return np.ones_like(var)
        return (1.0 / (var + self.tau)) / self.ref_prec[b]

    # ----------------------------------------------------------------- fit / predict
    def fit(self, D, idx, y):
        idx = list(idx)
        y = pd.Series(y, index=idx).astype(int)
        self.encoder = PathwayEncoder(self.pathways, self.min_genes, self.min_mets, self.use_uncoupling,
                                      self.pathway_level).fit(D, idx)
        feats = self.encoder.transform(D, idx)
        self.active = [b for b in self.blocks if b in feats]
        self._cols = {}
        self.block_models, self.ref_prec = {}, {}
        oof_z = pd.DataFrame(np.nan, index=idx, columns=self.active)
        oof_v = pd.DataFrame(np.nan, index=idx, columns=self.active)
        for b in self.active:
            self._cur = b
            Xdf = self._prepare_block(feats[b], train=True)
            ok = self._complete(Xdf)
            if Xdf.shape[1] == 0 or ok.sum() < 20 or len(np.unique(y.values[ok])) < 2:
                continue
            X, yy = Xdf.values[ok], y.values[ok]
            bm = BlockModel(self.Cs, self.n_boot, self.inner_folds, self.seed).fit(X, yy)
            self.block_models[b] = bm
            # out-of-fold block logits for the stacker
            skf = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed + 1)
            sub_idx = np.array(idx)[ok]
            for a, c in skf.split(X, yy):
                m = BlockModel((bm.C,), self.n_boot_inner, self.inner_folds, self.seed).fit(X[a], yy[a], C=bm.C)
                z, v = m.predict_logit(X[c])
                oof_z.loc[sub_idx[c], b] = z
                oof_v.loc[sub_idx[c], b] = v
            self.ref_prec[b] = np.nanmedian(1.0 / (oof_v[b].values + self.tau))
        self.active = [b for b in self.active if b in self.block_models]
        Xs = self._stack_matrix(oof_z[self.active], oof_v[self.active])
        self.alpha, self.bias = self._fit_stacker(Xs, y.values)
        self.feats_train_ = feats
        return self

    def _stack_matrix(self, Z, V):
        S = np.zeros(Z.shape)
        for j, b in enumerate(Z.columns):
            z, v = Z[b].values, V[b].values
            miss = np.isnan(z)
            pi = self._precision(b, np.where(miss, 0, v))
            S[:, j] = np.where(miss, 0.0, pi * np.nan_to_num(z))
        return S

    def _fit_stacker(self, S, y):
        k = S.shape[1]

        def loss(w):
            z = w[0] + S @ w[1:]
            p = _sigmoid(z)
            ll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            return ll + self.stack_l2 * 0.5 * np.sum(w[1:] ** 2) / len(y)

        w0 = np.r_[0.0, np.full(k, 1.0 / max(k, 1))]
        res = minimize(loss, w0, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * k)
        return res.x[1:], res.x[0]

    def block_outputs(self, D, idx):
        feats = self.encoder.transform(D, list(idx))
        Z = pd.DataFrame(np.nan, index=idx, columns=self.active)
        V = pd.DataFrame(np.nan, index=idx, columns=self.active)
        for b in self.active:
            if b not in feats:
                continue
            self._cur = b
            Xdf, ok = self._usable(self._prepare_block(feats[b]), b)
            if ok.sum() == 0:
                continue
            z, v = self.block_models[b].predict_logit(Xdf.values[ok])
            Z.loc[np.array(idx)[ok], b] = z
            V.loc[np.array(idx)[ok], b] = v
        return Z, V

    def predict(self, D, idx, return_details=False):
        idx = list(idx)
        Z, V = self.block_outputs(D, idx)
        S = self._stack_matrix(Z, V)
        logit = self.bias + S @ self.alpha
        prob = _sigmoid(logit)
        if not return_details:
            return prob
        pis = np.column_stack([np.where(np.isnan(Z[b]), 0, self._precision(b, np.nan_to_num(V[b].values)))
                               for b in self.active])
        unc = np.sqrt(np.nansum((self.alpha * pis) ** 2 * np.nan_to_num(V.values), axis=1))
        contrib = pd.DataFrame(S * self.alpha, index=idx, columns=self.active)
        return {"prob": prob, "logit": logit, "uncertainty": unc, "contrib": contrib,
                "n_layers": Z.notna().sum(axis=1).values}

    # ----------------------------------------------------------------- interpretation
    def map_scores(self):
        """Mechanism-aware pathway prioritisation."""
        rows = {}
        for b in self.active:
            if b == "G":
                continue
            bm = self.block_models[b]
            cols = self._cols[b]
            W = bm.coef_
            wbar = W.mean(axis=0)
            stab = np.abs(np.sign(W).mean(axis=0))
            for j, c in enumerate(cols):
                rows.setdefault(c, {})[b] = (wbar[j], stab[j])
        out = []
        a = dict(zip(self.active, self.alpha))
        for pw, d in rows.items():
            eff = {b: a[b] * abs(w) * s for b, (w, s) in d.items()}
            sgn = {b: np.sign(w) for b, (w, s) in d.items()}
            tot = sum(eff.values())
            if tot <= 0:
                continue
            # T, P and M share the direction of the pathway state; U and V are deviations and are
            # counted as supporting evidence when they agree with the protein-level direction.
            conc = abs(sum(sgn[b] * eff[b] for b in eff)) / tot
            out.append({"pathway": pw, "MAP": tot * conc, "evidence": tot, "concordance": conc,
                        **{f"w_{b}": d[b][0] for b in d}, **{f"stab_{b}": d[b][1] for b in d},
                        "n_layers": len(d)})
        return pd.DataFrame(out).sort_values("MAP", ascending=False).reset_index(drop=True)

    def block_weights(self):
        mean_pi = {b: 1.0 for b in self.active}
        return pd.Series(self.alpha, index=self.active)
