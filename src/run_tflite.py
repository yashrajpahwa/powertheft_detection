"""Run a saved TFLite theft model on held-out consumers (no training).

  python src/run_tflite.py                         # int8 model, accuracy on held-out consumers
  python src/run_tflite.py --consumer 0 --days 20  # print per-day predictions for one consumer
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))
import data
import features
from train_cnn import make_inputs, nn_features

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models" / "theft_int8.tflite"))
ap.add_argument("--test-fold", type=int, default=3, help="must match the fold held out at export")
ap.add_argument("--consumer", type=int, help="index within the held-out consumers")
ap.add_argument("--days", type=int, default=30)
a = ap.parse_args()

d = data.load()
ch = data.FEATURE_SETS["all"]
seq, lvl = make_inputs(d["X"], ch)
F = nn_features(d["X"], ch)
te = np.where(data.group_folds(d["btype"]) == a.test_fold)[0]

it = tf.lite.Interpreter(model_path=a.model)
it.allocate_tensors()
ins, out = it.get_input_details(), it.get_output_details()[0]


def predict(c, day):
    """Classify one day of one consumer; returns class probabilities."""
    arrays = [seq[c, day][None], lvl[c, day][None], F[c, day][None]]
    for det in ins:
        v = next(x for x in arrays if x.shape[1:] == tuple(det["shape"][1:]))
        if det["dtype"] == np.int8:
            s, z = det["quantization"]
            v = np.clip(np.round(v / s + z), -128, 127).astype(np.int8)
        it.set_tensor(det["index"], v)
    it.invoke()
    o = it.get_tensor(out["index"])[0].astype(np.float32)
    if out["dtype"] == np.int8:
        s, z = out["quantization"]
        o = (o - z) * s
    return o


days = range(features.VALID_FROM, d["X"].shape[1])
if a.consumer is not None:
    c = te[a.consumer]
    print(f"consumer #{c} ({d['type_names'][d['btype'][c]]})")
    for day in list(days)[:a.days]:
        p = predict(c, day)
        k = int(p.argmax())
        flag = "  <-- ALERT" if k else ""
        print(f"day {day:3d}  true {data.CLASSES[d['yd'][c, day]]:7s} pred {data.CLASSES[k]:7s} "
              f"conf {p[k]:.2f}{flag}")
else:
    y, p = [], []
    for c in te:
        for day in days:
            y.append(d["yd"][c, day]); p.append(predict(c, day).argmax())
    print(f"{Path(a.model).name}: {len(te)} held-out consumers, {len(y)} days, acc {accuracy_score(y, p):.4f}")
    print(classification_report(y, p, target_names=data.CLASSES, digits=4))
    bin_acc = accuracy_score(np.array(y) > 0, np.array(p) > 0)
    print(f"binary theft-vs-normal accuracy: {bin_acc:.4f}")
