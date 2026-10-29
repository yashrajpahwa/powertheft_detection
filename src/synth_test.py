"""Test the int8 TFLite model on synthetic theft data.

Base: held-out consumers (never seen in training). Their theft days are replaced by nearby
normal days to get a clean year; then fresh thefts are injected on random days with our own
generator, sequentially (so history can contain earlier thefts, like the real data).

Scenarios
  realistic  generator matching the dataset's actual theft types (reverse-engineered)
  new_cons   realistic + each consumer rescaled x0.2-5 and 5% hourly noise ("new" consumers)
  paper_spec types as literally described in the paper: T5 = flat day mean, T6 = reversed day
             (out-of-distribution: the model never saw these exact variants)
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
import data
import features
from train_cnn import make_inputs, nn_features

ROOT = Path(__file__).resolve().parents[1]


def clean_base(X, yd, yh):
    """Replace every non-normal day with the nearest fully-normal day of the same consumer."""
    X = X.copy()
    normal = (yh == 0).all(-1)
    for c in range(X.shape[0]):
        ok = np.where(normal[c])[0]
        for dd in np.where(~normal[c])[0]:
            same_wd = ok[(ok - dd) % 7 == 0]              # prefer same weekday
            pool = same_wd if len(same_wd) else ok
            X[c, dd] = X[c, pool[np.abs(pool - dd).argmin()]]
    return X


def inject(X, rng, p_theft, paper_spec=False):
    """Inject thefts day by day. Returns new X and day labels."""
    X = X.copy()
    n, nd = X.shape[:2]
    y = np.zeros((n, nd), np.int64)
    for c in range(n):
        orig = X[c].copy()                                # untampered stream, for T5 rolling mean
        for dd in range(1, nd):
            if rng.random() > p_theft:
                continue
            k = int(rng.integers(1, 7))
            x = orig[dd]
            if k == 1:
                x = x * rng.uniform(0.1, 0.8)
            elif k == 2:
                s = int(rng.integers(0, 23)); e = int(rng.integers(s + 1, 25))
                x = x.copy(); x[s:e] = 0
            elif k == 3:
                x = x * rng.uniform(0.1, 0.8, (24, 1))
            elif k == 4:
                x = x.mean(0, keepdims=True) * rng.uniform(0.1, 0.8) * rng.uniform(0, 1, (24, 1))
            elif k == 5:
                if paper_spec:
                    x = np.repeat(x.mean(0, keepdims=True), 24, 0)
                else:                                      # trailing 24 h rolling mean
                    s = np.concatenate([orig[dd - 1], orig[dd]], 0)
                    x = np.stack([s[t + 1:t + 25].mean(0) for t in range(24)])
            elif k == 6:
                x = x[::-1].copy() if paper_spec else X[c, dd - 1].copy()
            X[c, dd] = x
            y[c, dd] = k
    return X.astype(np.float32), y


class Int8Model:
    def __init__(self, path):
        self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.ins = self.it.get_input_details()
        self.out = self.it.get_output_details()[0]
        assert all(det["dtype"] == np.int8 for det in self.ins), "not an int8 model"

    def predict(self, arrays):
        preds = []
        for i in range(len(arrays[0])):
            for det in self.ins:
                v = next(x for x in arrays if x.shape[1:] == tuple(det["shape"][1:]))[i:i + 1]
                s, z = det["quantization"]
                self.it.set_tensor(det["index"], np.clip(np.round(v / s + z), -128, 127).astype(np.int8))
            self.it.invoke()
            preds.append(self.it.get_tensor(self.out["index"])[0].argmax())
        return np.array(preds)


def run(model, X, y, name):
    ch = data.FEATURE_SETS["all"]
    seq, lvl = make_inputs(X, ch)
    F = nn_features(X, ch)
    V = features.VALID_FROM
    arrays = [a[:, V:].reshape(-1, *a.shape[2:]) for a in (seq, lvl, F)]
    yt = y[:, V:].ravel()
    p = model.predict(arrays)
    acc, f1 = accuracy_score(yt, p), f1_score(yt, p, average="macro")
    recall = confusion_matrix(yt, p, labels=range(7)).diagonal() / np.bincount(yt, minlength=7)
    false_alarm = ((p > 0) & (yt == 0)).sum() / (yt == 0).sum()
    detect = ((p > 0) & (yt > 0)).sum() / (yt > 0).sum()
    print(f"{name:11s} acc {acc:.4f}  macroF1 {f1:.4f}  theft-detect {detect:.4f}  "
          f"false-alarm {false_alarm:.4f}  per-class recall {np.round(recall, 3).tolist()}", flush=True)
    return yt, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "theft_int8.tflite"))
    ap.add_argument("--test-fold", type=int, default=3)
    ap.add_argument("--p-theft", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    d = data.load()
    te = data.group_folds(d["btype"]) == a.test_fold
    base = clean_base(d["X"][te], d["yd"][te], d["yh"][te])
    model = Int8Model(a.model)
    print(f"int8 model {Path(a.model).name}, {te.sum()} held-out consumers, "
          f"theft prob/day {a.p_theft}, classes order {data.CLASSES}")
    pooled = {k: ([], []) for k in ("realistic", "new_cons", "paper_spec")}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        print(f"--- seed {seed}")
        Xr, yr = inject(base, rng, a.p_theft)
        scale = rng.uniform(np.log(0.2), np.log(5), (len(base), 1, 1, 1))
        Xn = base * np.exp(scale) * rng.normal(1, 0.05, base.shape).clip(0.8, 1.2)
        Xn, yn = inject(Xn.astype(np.float32), rng, a.p_theft)
        Xp, yp = inject(base, rng, a.p_theft, paper_spec=True)
        for name, Xs, ys in [("realistic", Xr, yr), ("new_cons", Xn, yn), ("paper_spec", Xp, yp)]:
            yt, p = run(model, Xs, ys, name)
            pooled[name][0].append(yt); pooled[name][1].append(p)

    short = ["Norm", "T1", "T2", "T3", "T4", "T5", "T6"]
    for name, (ys, ps) in pooled.items():
        yt, p = np.concatenate(ys), np.concatenate(ps)
        cm = confusion_matrix(yt, p, labels=range(7))
        print(f"\n=== {name}: all {a.seeds} seeds, {len(yt)} days | acc {accuracy_score(yt, p):.4f} "
              f"| macroF1 {f1_score(yt, p, average='macro'):.4f} "
              f"| binary acc {accuracy_score(yt > 0, p > 0):.4f}")
        print("true\\pred " + "".join(f"{s:>7s}" for s in short) + "  recall")
        for i, row in enumerate(cm):
            print(f"{short[i]:>9s} " + "".join(f"{v:7d}" for v in row) + f"  {row[i] / row.sum():.4f}")
