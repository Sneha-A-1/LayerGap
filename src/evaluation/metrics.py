"""Evaluation metrics and statistical tests."""
import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             matthews_corrcoef, roc_auc_score)

THRESHOLD = 0.5  # pre-specified decision threshold for all methods


def classification_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    yhat = (p >= THRESHOLD).astype(int)
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "BalAcc": balanced_accuracy_score(y, yhat), "MCC": matthews_corrcoef(y, yhat),
            "Brier": brier_score_loss(y, p)}


def bootstrap_ci(y, p, fn=roc_auc_score, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) == 2:
            vals.append(fn(y[b], p[b]))
    return np.percentile(vals, [2.5, 97.5])


def paired_bootstrap_p(y, p1, p2, fn=roc_auc_score, n=2000, seed=0):
    """Two-sided bootstrap p-value for fn(p1) - fn(p2) on the same samples."""
    rng = np.random.default_rng(seed)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d0 = fn(y, p1) - fn(y, p2)
    ds = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) == 2:
            ds.append(fn(y[b], p1[b]) - fn(y[b], p2[b]))
    ds = np.array(ds)
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return d0, max(p, 1.0 / len(ds))


def corrected_resampled_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected repeated cross-validation t-test for paired fold scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return t, p


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def harrell_c(time, event, risk):
    """Harrell's concordance index (higher risk -> shorter time)."""
    from lifelines.utils import concordance_index
    return concordance_index(time, -np.asarray(risk), event)
