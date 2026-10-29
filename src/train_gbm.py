"""LightGBM on day-level features, consumer-grouped CV. Optionally adds yesterday's features
(--lag) so the model knows whether the previous day was itself tampered with."""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
import data
import features

ap = argparse.ArgumentParser()
ap.add_argument("--channels", default="all", choices=list(data.FEATURE_SETS))
ap.add_argument("--lag", action="store_true")
ap.add_argument("--trees", type=int, default=400)
ap.add_argument("--leaves", type=int, default=31)
ap.add_argument("--loto", action="store_true", help="also run leave-building-type-out")
a = ap.parse_args()

d = data.load()
F, names = features.day_features(d["X"], data.FEATURE_SETS[a.channels])
if a.lag:
    F = np.concatenate([F, np.concatenate([F[:, :1], F[:, :-1]], 1)], -1)
    names = names + [n + "_lag" for n in names]
yd = d["yd"]
V = features.VALID_FROM + int(a.lag)
F, yd = F[:, V:], yd[:, V:]


def run(train_mask, test_mask):
    clf = lgb.LGBMClassifier(n_estimators=a.trees, num_leaves=a.leaves, learning_rate=0.05,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             class_weight="balanced", verbose=-1, random_state=0)
    clf.fit(F[train_mask].reshape(-1, F.shape[-1]), yd[train_mask].ravel())
    p = clf.predict(F[test_mask].reshape(-1, F.shape[-1]))
    y = yd[test_mask].ravel()
    return accuracy_score(y, p), f1_score(y, p, average="macro"), clf, y, p


folds = data.group_folds(d["btype"])
res, ys, ps = [], [], []
for k in range(folds.max() + 1):
    acc, f1, clf, y, p = run(folds != k, folds == k)
    res.append((acc, f1)); ys.append(y); ps.append(p)
    print(f"fold {k}: acc {acc:.4f} macroF1 {f1:.4f}", flush=True)
r = np.array(res)
tag = f"gbm_{a.channels}{'_lag' if a.lag else ''}"
print(f"{tag}: acc {r[:,0].mean():.4f}±{r[:,0].std():.4f} macroF1 {r[:,1].mean():.4f}  n_feat={F.shape[-1]}")
from sklearn.metrics import confusion_matrix
print(confusion_matrix(np.concatenate(ys), np.concatenate(ps)))
out = {"model": tag, "acc": r[:, 0].tolist(), "f1": r[:, 1].tolist()}

if a.loto:
    lr = []
    for t in np.unique(d["btype"]):
        acc, f1, *_ = run(d["btype"] != t, d["btype"] == t)
        lr.append((acc, f1))
    lr = np.array(lr)
    print(f"leave-type-out: acc {lr[:,0].mean():.4f} macroF1 {lr[:,1].mean():.4f}")
    out["loto_acc"] = lr[:, 0].tolist()

imp = sorted(zip(clf.feature_importances_, names), reverse=True)[:15]
print("top features:", [n for _, n in imp])
Path(__file__).parents[1].joinpath("reports").mkdir(exist_ok=True)
json.dump(out, open(Path(__file__).parents[1] / "reports" / f"{tag}.json", "w"), indent=1)
