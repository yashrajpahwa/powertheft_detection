"""Day-level features from today's 24h profile vs. yesterday and a 7-day history mean.

Everything is simple arithmetic over <= 8 days x 24 h, so it can run on a microcontroller.
Day 0 of each consumer has no history; it is excluded from training/evaluation (VALID_FROM).
"""
import numpy as np

EPS = 1e-3
HIST = 7
VALID_FROM = 1


def history(X, hist=HIST):
    """Per-hour mean of the previous `hist` days (fewer at the start of the year)."""
    c = np.cumsum(X, axis=1, dtype=np.float64)
    c = np.concatenate([np.zeros_like(c[:, :1]), c], 1)  # c[:, d] = sum of days < d
    d = np.arange(X.shape[1])
    lo = np.maximum(d - hist, 0)
    cnt = np.maximum(d - lo, 1)[None, :, None, None]
    return ((c[:, d] - c[:, lo]) / cnt).astype(np.float32)


def _corr(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + EPS)


def _longest_zero_run(z):
    run = np.zeros(z.shape[:-1], np.int32)
    best = np.zeros_like(run)
    for t in range(z.shape[-1]):
        run = np.where(z[..., t], run + 1, 0)
        best = np.maximum(best, run)
    return best


def day_features(X, channels):
    """X: (n_cons, n_days, 24, 10) -> F: (n_cons, n_days, n_feat), names."""
    X = X[..., channels]
    P = np.concatenate([X[:, :1], X[:, :-1]], 1)   # yesterday (day 0: itself; excluded anyway)
    H = history(X)
    # hour axis last: (n, d, C, 24)
    x, p, h = (np.moveaxis(a, 2, -1) for a in (X, P, H))
    mx, mp, mh = x.mean(-1), p.mean(-1), h.mean(-1)
    lr = np.log((x + EPS) / (h + EPS))              # per-hour log ratio vs. history
    zero = x <= EPS
    rough = lambda a, m: np.abs(np.diff(a, 2, -1)).mean(-1) / (m + EPS)

    f = {
        "log_mean": np.log1p(mx),
        "ratio_prev": np.log((mx + EPS) / (mp + EPS)),
        "ratio_hist": np.log((mx + EPS) / (mh + EPS)),
        "cv": x.std(-1) / (mx + EPS),
        "cv_hist": h.std(-1) / (mh + EPS),
        "min_rel": x.min(-1) / (mx + EPS),
        "max_rel": x.max(-1) / (mx + EPS),
        "n_zero": zero.sum(-1),
        "zero_run": _longest_zero_run(zero),
        "rough": rough(x, mx),
        "rough_ratio": np.log((rough(x, mx) + EPS) / (rough(h, mh) + EPS)),
        "diff_prev": np.abs(x - p).max(-1) / (mx + EPS),
        "corr_prev": _corr(x, p),
        "corr_hist": _corr(x, h),
        "corr_rev": _corr(x[..., ::-1], h),
        "lr_mean": lr.mean(-1),
        "lr_std": lr.std(-1),
        "step_max": np.abs(np.diff(x, 1, -1)).max(-1) / (mx + EPS),
    }
    names, cols = [], []
    for k, v in f.items():
        for ci, ch in enumerate(channels):
            names.append(f"{k}_c{ch}")
            cols.append(v[..., ci])
    F = np.stack(cols, -1).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), names
