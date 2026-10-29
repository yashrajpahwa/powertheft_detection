"""Tiny 1D-CNN on (today, yesterday, 7-day history, |today-yesterday|) hourly profiles.

Input per day: (24, 4*C) scaled per channel by the history mean, plus (C,) log history level.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
import numpy as np
from keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
import data
import features

ROOT = Path(__file__).resolve().parents[1]
CLIP = 8.0


def make_inputs(X, channels):
    """X: (n, days, 24, 10) -> seq (n, days, 24, 4C), lvl (n, days, C)."""
    X = X[..., channels]
    P = np.concatenate([X[:, :1], X[:, :-1]], 1)
    H = features.history(X)
    s = H.mean(2, keepdims=True) + features.EPS          # per-day, per-channel scale
    seq = np.concatenate([X / s, P / s, H / s, np.abs(X - P) / s], -1)
    seq = np.clip(seq, 0, CLIP).astype(np.float32)
    lvl = np.log1p(s[:, :, 0]).astype(np.float32)
    return seq, lvl


def nn_features(X, channels):
    """Engineered day features, signed-log scaled for the NN (keeps exact-0 replay distinct)."""
    F = features.day_features(X, channels)[0]
    return (np.sign(F) * np.log1p(np.abs(F) * 100)).astype(np.float32)


def build(n_in, n_lvl, width=32, dense=32, n_cls=7, feat_norm=None):
    """width=0 -> no conv branch (MLP on features only). feat_norm: adapted Normalization
    layer for the engineered-feature input (None -> no feature input)."""
    seq = keras.Input((24, n_in), name="seq")
    lvl = keras.Input((n_lvl,), name="lvl")
    inputs, parts = [seq, lvl], [lvl]
    if width:
        x = layers.Conv1D(width, 3, padding="same", use_bias=False)(seq)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        for dil in (2, 4, 8):
            y = layers.SeparableConv1D(width, 3, padding="same", dilation_rate=dil, use_bias=False)(x)
            y = layers.BatchNormalization()(y)
            x = layers.ReLU()(layers.Add()([x, y]))
        parts += [layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)]
    if feat_norm is not None:
        feat = keras.Input((feat_norm.mean.shape[-1],), name="feat")
        inputs.append(feat)
        parts.append(feat_norm(feat))
    g = layers.Concatenate()(parts)
    g = layers.Dense(dense, activation="relu")(g)
    g = layers.Dropout(0.2)(g)
    g = layers.Dense(dense, activation="relu")(g)
    out = layers.Dense(n_cls, activation="softmax")(g)
    return keras.Model(inputs, out)


def fit(model, xs, y, epochs, seed=0):
    cw = len(y) / (7 * np.bincount(y, minlength=7))
    steps = epochs * int(np.ceil(len(y) / 128))
    sched = keras.optimizers.schedules.CosineDecay(3e-3, steps)
    model.compile(optimizer=keras.optimizers.Adam(sched), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    keras.utils.set_random_seed(seed)
    model.fit(xs, y, sample_weight=cw[y].astype(np.float32), batch_size=128,
              epochs=epochs, verbose=0)
    return model


def feat_layer(F_train):
    norm = layers.Normalization()
    norm.adapt(F_train)
    return norm


def flat(a, mask):
    """Select consumers, drop history-less days, flatten (consumer, day) into samples."""
    return a[mask][:, features.VALID_FROM:].reshape(-1, *a.shape[2:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="all", choices=list(data.FEATURE_SETS))
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dense", type=int, default=32)
    ap.add_argument("--feat", action="store_true", help="add engineered day features")
    ap.add_argument("--folds", type=int, default=4, help="how many CV folds to run")
    a = ap.parse_args()

    d = data.load()
    ch = data.FEATURE_SETS[a.channels]
    seq, lvl = make_inputs(d["X"], ch)
    F = nn_features(d["X"], ch)
    yd = d["yd"].astype(np.int64)
    folds = data.group_folds(d["btype"])
    res, ys, ps = [], [], []
    for k in range(a.folds):
        tr, te = folds != k, folds == k
        norm = feat_layer(flat(F, tr)) if a.feat else None
        m = build(seq.shape[-1], lvl.shape[-1], a.width, a.dense, feat_norm=norm)
        xs = lambda mask: [flat(seq, mask), flat(lvl, mask)] + ([flat(F, mask)] if a.feat else [])
        fit(m, xs(tr), flat(yd, tr), a.epochs, seed=k)
        p = m.predict(xs(te), verbose=0).argmax(-1)
        y = flat(yd, te)
        res.append((accuracy_score(y, p), f1_score(y, p, average="macro"))); ys.append(y); ps.append(p)
        print(f"fold {k}: acc {res[-1][0]:.4f} macroF1 {res[-1][1]:.4f}", flush=True)
    r = np.array(res)
    tag = f"cnn_{a.channels}_w{a.width}_d{a.dense}{'_feat' if a.feat else ''}"
    print(f"{tag}: acc {r[:,0].mean():.4f}±{r[:,0].std():.4f} macroF1 {r[:,1].mean():.4f} params={m.count_params()}")
    print(confusion_matrix(np.concatenate(ys), np.concatenate(ps)))
    json.dump({"model": tag, "params": m.count_params(), "acc": r[:, 0].tolist(), "f1": r[:, 1].tolist()},
              open(ROOT / "reports" / f"{tag}.json", "w"), indent=1)
