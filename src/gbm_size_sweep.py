"""Accuracy vs. model-size frontier for LightGBM on day features (consumer-grouped CV).
Size estimate: 12 bytes per internal node + 4 bytes per leaf (float32 threshold, feature idx,
child idx), i.e. what a flat C array export (emlearn-style) needs."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
import data
import features

d = data.load()
F, names = features.day_features(d["X"], data.FEATURE_SETS["all"])
V = features.VALID_FROM
F, yd = F[:, V:], d["yd"][:, V:]
folds = data.group_folds(d["btype"])


def size_bytes(booster):
    nodes = leaves = 0
    for t in booster.dump_model()["tree_info"]:
        leaves += t["num_leaves"]
        nodes += t["num_leaves"] - 1
    return nodes * 12 + leaves * 4


rows = []
for trees, leaves in [(30, 7), (60, 7), (100, 7), (60, 15), (100, 15), (200, 15), (100, 31), (400, 31)]:
    acc, f1, sz = [], [], []
    for k in range(folds.max() + 1):
        tr, te = folds != k, folds == k
        clf = lgb.LGBMClassifier(n_estimators=trees, num_leaves=leaves, learning_rate=0.1 if trees <= 100 else 0.05,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 class_weight="balanced", verbose=-1, random_state=0)
        clf.fit(F[tr].reshape(-1, F.shape[-1]), yd[tr].ravel())
        p = clf.predict(F[te].reshape(-1, F.shape[-1]))
        acc.append(accuracy_score(yd[te].ravel(), p)); f1.append(f1_score(yd[te].ravel(), p, average="macro"))
        sz.append(size_bytes(clf.booster_))
    row = dict(trees=trees, leaves=leaves, acc=float(np.mean(acc)), acc_std=float(np.std(acc)),
               f1=float(np.mean(f1)), kb=float(np.mean(sz)) / 1024)
    rows.append(row)
    print(f"{trees:4d} rounds x7  {leaves:2d} leaves: acc {row['acc']:.4f}±{row['acc_std']:.4f} "
          f"F1 {row['f1']:.4f}  ~{row['kb']:.1f} KB", flush=True)
json.dump(rows, open(Path(__file__).parents[1] / "reports" / "gbm_size_sweep.json", "w"), indent=1)
