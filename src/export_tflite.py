"""Train the hybrid CNN, convert to TFLite (float32 and full-int8), and evaluate both with the
TFLite interpreter on held-out consumers (fold --test-fold, never seen in training).

Outputs: models/theft_float.tflite, models/theft_int8.tflite, reports/tflite_eval.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
import data
from train_cnn import build, feat_layer, fit, flat, make_inputs, nn_features

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--dense", type=int, default=32)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--test-fold", type=int, default=3)
a = ap.parse_args()

d = data.load()
ch = data.FEATURE_SETS["all"]
seq, lvl = make_inputs(d["X"], ch)
F = nn_features(d["X"], ch)
yd = d["yd"].astype(np.int64)
folds = data.group_folds(d["btype"])
tr, te = folds != a.test_fold, folds == a.test_fold
xs = lambda m: [flat(seq, m), flat(lvl, m), flat(F, m)]
x_tr, x_te, y_te = xs(tr), xs(te), flat(yd, te)

model = build(seq.shape[-1], lvl.shape[-1], a.width, a.dense, feat_norm=feat_layer(x_tr[2]))
fit(model, x_tr, flat(yd, tr), a.epochs, seed=0)
p_keras = model.predict(x_te, verbose=0).argmax(-1)
print(f"keras float: acc {accuracy_score(y_te, p_keras):.4f}  params {model.count_params()}")


def input_order(blob):
    """TFLite may reorder inputs; map each interpreter input to our array index by shape."""
    dets = tf.lite.Interpreter(model_content=blob).get_input_details()
    return [next(j for j, x in enumerate(x_te) if x.shape[1:] == tuple(det["shape"][1:])) for det in dets]


def convert(int8, order=None):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x_tr[0]), 1000, replace=False)

        def rep():
            for i in idx:
                yield [x_tr[j][i:i + 1] for j in order]

        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def evaluate(blob):
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    ins, out = it.get_input_details(), it.get_output_details()[0]
    order = input_order(blob)
    preds, t0 = [], time.perf_counter()
    for i in range(len(y_te)):
        for det, j in zip(ins, order):
            v = x_te[j][i:i + 1]
            if det["dtype"] == np.int8:
                s, z = det["quantization"]
                v = np.clip(np.round(v / s + z), -128, 127).astype(np.int8)
            it.set_tensor(det["index"], v)
        it.invoke()
        preds.append(it.get_tensor(out["index"])[0].argmax())
    us = (time.perf_counter() - t0) / len(y_te) * 1e6
    return np.array(preds), us


(ROOT / "models").mkdir(exist_ok=True)
report = {"params": model.count_params(), "test_consumers": int(te.sum()), "test_days": len(y_te)}
order = None
for name, int8 in [("float", False), ("int8", True)]:
    blob = convert(int8, order)
    order = input_order(blob)
    (ROOT / "models" / f"theft_{name}.tflite").write_bytes(blob)
    p, us = evaluate(blob)
    acc, f1 = accuracy_score(y_te, p), f1_score(y_te, p, average="macro")
    report[name] = {"acc": acc, "macro_f1": f1, "kb": len(blob) / 1024, "us_per_day": us,
                    "agree_with_keras": float((p == p_keras).mean())}
    print(f"tflite {name:5s}: acc {acc:.4f} macroF1 {f1:.4f} size {len(blob)/1024:.1f} KB "
          f"{us:.0f} us/inference  agree-with-keras {(p == p_keras).mean():.4f}", flush=True)
    if int8:
        print(confusion_matrix(y_te, p))
json.dump(report, open(ROOT / "reports" / "tflite_eval.json", "w"), indent=1)
