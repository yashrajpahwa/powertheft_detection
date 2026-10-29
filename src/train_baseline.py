"""Paper-style baseline: RandomForest(100) on single hourly rows (10 channels + building type),
evaluated with our consumer-grouped folds (no consumer appears in both train and test)."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
import data

d = data.load()
X, yh, bt = d["X"], d["yh"], d["btype"]
folds = data.group_folds(bt)
n = X.shape[0]
rows = X.reshape(n, -1, 10)
typ = np.repeat(bt[:, None], rows.shape[1], 1)[..., None].astype(np.float32)
feats = np.concatenate([rows, typ], -1)
labels = yh.reshape(n, -1)

res = []
for k in range(folds.max() + 1):
    tr, te = folds != k, folds == k
    clf = RandomForestClassifier(100, n_jobs=-1, random_state=0)
    clf.fit(feats[tr].reshape(-1, 11), labels[tr].ravel())
    p = clf.predict(feats[te].reshape(-1, 11))
    y = labels[te].ravel()
    res.append((accuracy_score(y, p), f1_score(y, p, average="macro")))
    print(f"fold {k}: acc {res[-1][0]:.4f} macroF1 {res[-1][1]:.4f}", flush=True)
r = np.array(res)
print(f"RF hourly baseline: acc {r[:,0].mean():.4f}±{r[:,0].std():.4f}  macroF1 {r[:,1].mean():.4f}")
out = Path(__file__).parents[1] / "reports"
out.mkdir(exist_ok=True)
json.dump({"model": "RF hourly (paper-style)", "acc": r[:, 0].tolist(), "f1": r[:, 1].tolist()},
          open(out / "baseline_rf.json", "w"), indent=1)
