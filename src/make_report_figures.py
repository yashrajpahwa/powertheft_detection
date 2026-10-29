"""Regenerate Figs. 1-9 of Report_UEC523 as PNG files from the saved results.

Numbers come from reports/*.json, reports/synth_test.txt and the hold-out
tables in METHODOLOGY_AND_RESULTS.txt. Output: <repo parent>/report_figures/.
"""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT = ROOT.parent / "report_figures"
OUT.mkdir(exist_ok=True)

BLUE = "#2a78d6"
BLUE_FILL = "#d7e2ef"
BLUE_LIGHT = "#e3edf8"
GREY = "#c4c3b6"
GREY_LIGHT = "#dddcd6"
GREY_BG = "#eeeeec"
INK = "#1a1a1a"
MUTED = "#5a5a5a"
GRID = "#e4e4e4"
CLASSES = ["Normal", "Theft1", "Theft2", "Theft3", "Theft4", "Theft5", "Theft6"]
SHORT = ["Normal", "T1", "T2", "T3", "T4", "T5", "T6"]
DPI = 200

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def load(name):
    return json.loads((REP / name).read_text())


def mean_sd(name):
    acc = np.array(load(name)["acc"]) * 100
    return acc.mean(), acc.std()


def header(fig, title, subtitle=None):
    fig.text(0.03, 0.95, title, fontsize=15, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.03, 0.885, subtitle, fontsize=11, color=MUTED, va="top")


def clean_axes(ax, xgrid=True):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


_CM = None


def read_holdout_cm():
    """Confusion matrix of the int8 model on fold 3, from a fresh TFLite run."""
    global _CM
    if _CM is None:
        y, p = run_int8_holdout()
        _CM = np.zeros((7, 7), int)
        np.add.at(_CM, (y, p), 1)
    return _CM


def run_int8_holdout(test_fold=3):
    """Run models/theft_int8.tflite on every held-out day (same steps as run_tflite.py)."""
    import os
    import sys
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    sys.path.insert(0, str(Path(__file__).parent))
    import data
    import features
    from train_cnn import make_inputs, nn_features

    d = data.load()
    ch = data.FEATURE_SETS["all"]
    seq, lvl = make_inputs(d["X"], ch)
    F = nn_features(d["X"], ch)
    te = np.where(data.group_folds(d["btype"]) == test_fold)[0]
    it = tf.lite.Interpreter(model_path=str(ROOT / "models" / "theft_int8.tflite"))
    it.allocate_tensors()
    ins, out = it.get_input_details(), it.get_output_details()[0]
    y, p = [], []
    for c in te:
        for day in range(features.VALID_FROM, d["X"].shape[1]):
            arrays = [seq[c, day][None], lvl[c, day][None], F[c, day][None]]
            for det in ins:
                v = next(x for x in arrays if x.shape[1:] == tuple(det["shape"][1:]))
                if det["dtype"] == np.int8:
                    s, z = det["quantization"]
                    v = np.clip(np.round(v / s + z), -128, 127).astype(np.int8)
                it.set_tensor(det["index"], v)
            it.invoke()
            y.append(d["yd"][c, day])
            p.append(int(it.get_tensor(out["index"])[0].argmax()))
    y, p = np.array(y), np.array(p)
    np.savez(REP / "holdout_int8_predictions.npz", y_true=y, y_pred=p)
    print(f"int8 hold-out run: {len(y)} days, acc {(y == p).mean():.4f}")
    return y, p


def read_synth_recall():
    """Pooled per-class recall for the three stress-test scenarios."""
    text = (REP / "synth_test.txt").read_text()
    out = {}
    for part in text.split("=== ")[1:]:
        name = part.split(":")[0]
        rec = re.findall(r"^\s+(?:Norm|T[1-6])\s+(?:\d+\s+){7}([\d.]+)\s*$", part, re.M)
        out[name] = np.array([float(r) for r in rec[:7]])
    return out


# ---------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, title, body, style="solid"):
    if style == "built":
        fc, ec, lw, ls = BLUE_FILL, BLUE, 2.2, "-"
    elif style == "highlight":
        fc, ec, lw, ls = BLUE_LIGHT, BLUE, 2.2, "-"
    elif style == "dashed":
        fc, ec, lw, ls = "white", GREY, 1.4, (0, (4, 3))
    else:
        fc, ec, lw, ls = "white", GREY, 1.4, "-"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + 2, y + h - 2.4, title, fontsize=12.5, fontweight="bold", va="top")
    ax.text(x + 2, y + h - 6.2, body, fontsize=11, color=MUTED, va="top", linespacing=1.45)


def arrow(ax, pts):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=GREY, lw=1.4)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4, mutation_scale=12))


def fig1():
    fig = plt.figure(figsize=(13.4, 6.9))
    header(fig, "Fig. 1. The decision is made on the meter; only an alert leaves it")
    ax = fig.add_axes([0, 0, 1, 0.9])
    ax.set_xlim(0, 134); ax.set_ylim(0, 62); ax.axis("off")
    for y0, h, label in [(40, 21.5, "On the meter (ESP32-S3 class)"), (12, 21, "Over IoT and in the cloud")]:
        ax.add_patch(FancyBboxPatch((4, y0), 126, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=GREY_BG, ec=GREY_LIGHT, lw=1))
        ax.text(7, y0 + h - 2, label, fontsize=13, fontweight="bold", va="top")
    w, h, y = 27, 12, 43.5
    xs = [8, 38.5, 69, 99.5]
    box(ax, xs[0], y, w, h, "1 Meter log", "Hourly kW from\n10 sub-meters, 8 days", "dashed")
    box(ax, xs[1], y, w, h, "2 Preprocess", "24 × 40 curves,\n180 stats, 10 levels", "built")
    box(ax, xs[2], y, w, h, "3 int8 hybrid CNN", "37,216 parameters\n64.0 KB", "built")
    box(ax, xs[3], y, w, h, "4 Decide", "Most probable of\nNormal, Theft1–6", "built")
    for a, b in zip(xs, xs[1:]):
        arrow(ax, [(a + w + 0.3, y + h / 2), (b - 0.3, y + h / 2)])
    box(ax, 99.5, 14.5, w, 12, "5 Send alert", "Consumer ID, theft\ntype, probability", "dashed")
    box(ax, 69, 14.5, w, 12, "6 Dashboard", "Ranked list for\nfield inspection", "dashed")
    arrow(ax, [(113, y), (113, 26.6)])
    ax.text(111, 36.5, "theft decisions only", fontsize=11, color=MUTED, ha="right", va="center")
    arrow(ax, [(99.2, 20.5), (96.3, 20.5)])
    ax.add_patch(FancyBboxPatch((4, 2.5), 4, 2.6, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=BLUE_FILL, ec=BLUE, lw=2))
    ax.text(9.5, 3.8, "Built and tested on a PC (int8 TFLite interpreter)", fontsize=11, color=MUTED, va="center")
    ax.add_patch(FancyBboxPatch((69, 2.5), 4, 2.6, boxstyle="round,pad=0,rounding_size=0.5",
                                fc="white", ec=GREY, lw=1.4, ls=(0, (3, 2))))
    ax.text(74.5, 3.8, "Deployment design, not yet built", fontsize=11, color=MUTED, va="center")
    save(fig, "fig1_system_overview.png")


def fig2():
    params = load("tflite_eval.json")["params"]
    fig = plt.figure(figsize=(13.4, 8.2))
    header(fig, "Fig. 2. Learned shape features and engineered statistics meet before the classifier")
    ax = fig.add_axes([0, 0, 1, 0.91])
    ax.set_xlim(0, 134); ax.set_ylim(0, 82); ax.axis("off")
    ax.text(4, 79, "Convolutional branch", fontsize=11.5, fontweight="bold", color=MUTED, va="top")
    w, h, y = 20, 12.5, 61.5
    xs = [4, 27, 49.5, 72]
    box(ax, xs[0], y, w, h, "Input", "24 h × 40\nscaled curves")
    box(ax, xs[1], y, w, h, "Conv1D", "48 filters,\nkernel 3, ReLU")
    box(ax, xs[2], y, w, h, "Residual × 3", "dilation\n2, 4, 8")
    box(ax, xs[3], y, w, h, "Pooling", "global avg + max\n48 + 48 values")
    for a, b in zip(xs, xs[1:]):
        arrow(ax, [(a + w + 0.3, y + 6.2), (b - 0.3, y + 6.2)])
    box(ax, 99, 45, 31, 12.5, "Concatenate", "96 + 190 = 286\nfeatures", "highlight")
    arrow(ax, [(92.3, y + 6.2), (95.5, y + 6.2), (95.5, 52), (98.7, 52)])
    ax.text(93, y + 8.5, "96", fontsize=11, color=MUTED)
    ax.text(99, 59.5, "Fusion and output", fontsize=11.5, fontweight="bold", color=MUTED)
    ax.text(4, 55.5, "Statistics branch", fontsize=11.5, fontweight="bold", color=MUTED, va="top")
    box(ax, 4, 38, 42.5, 12.5, "Statistics and levels", "180 signed-log statistics φ(f)\n+ 10 load levels ℓ")
    arrow(ax, [(46.8, 44), (95.5, 44), (95.5, 48), (98.7, 48)])
    ax.text(88, 45.2, "190", fontsize=11, color=MUTED, ha="right")
    box(ax, 99, 24, 31, 12.5, "Dense 64 × 2", "ReLU,\ndropout 0.2")
    box(ax, 99, 3, 31, 12.5, "Softmax, 7 classes", "Normal or\nTheft1 to Theft6")
    arrow(ax, [(114.5, 44.8), (114.5, 36.8)])
    arrow(ax, [(114.5, 23.8), (114.5, 15.8)])
    ax.text(4, 28, f"{params:,} trainable parameters in total;\nboth branches trained jointly with one loss.",
            fontsize=11, color=MUTED, va="top", linespacing=1.5)
    save(fig, "fig2_hybrid_cnn_architecture.png")


# ------------------------------------------------------------------ charts
def fig3():
    rows = [("LightGBM, 400 trees", "gbm_all.json", False),
            ("Hybrid CNN 48/64 (final)", "cnn_all_w48_d64_feat.json", True),
            ("Hybrid CNN 32/32", "cnn_all_w32_d32_feat.json", False),
            ("Random forest, single hour", "baseline_rf.json", False)]
    stats = [mean_sd(f) for _, f, _ in rows]
    gap = stats[0][0] - stats[1][0]
    gbm_kb = load("gbm_size_sweep.json")[-1]["kb"]
    fig = plt.figure(figsize=(13.4, 5.4))
    header(fig, f"Fig. 3. Hybrid CNN reaches {stats[1][0]:.2f}%, just below the LightGBM ceiling",
           f"Mean 4-fold CV accuracy (%) by consumer, whiskers ±1 SD. Gap to LightGBM: {gap:.2f} points")
    ax = fig.add_axes([0.40, 0.17, 0.56, 0.56])
    for i, ((label, _, final), (m, sd)) in enumerate(zip(rows, stats)):
        c = BLUE if final else GREY
        ax.plot([m - sd, m + sd], [i, i], color=c, lw=2, solid_capstyle="butt")
        ax.plot(m, i, "o", color=c, ms=11, zorder=3)
        if m < 90:
            ax.text(m + 2.6, i, f"{m:.2f}%", va="center", fontsize=12, color=MUTED)
        else:
            ax.text(m - 0.6, i, f"{m:.2f}%", va="center", ha="right", fontsize=12,
                    color=INK if final else MUTED, fontweight="bold" if final else "normal")
    ax.axvline(stats[0][0], color=MUTED, ls=(0, (3, 3)), lw=1.2)
    ax.text(stats[0][0], -0.75, f"LightGBM ceiling ({gbm_kb:.0f} KB, not TFLite)",
            ha="right", va="bottom", fontsize=11.5, color=MUTED)
    ax.set_yticks(range(4))
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[2]:
            t.set_fontweight("bold"); t.set_color(INK)
    ax.tick_params(axis="y", pad=95)
    ax.set_ylim(3.5, -0.6)
    ax.set_xlim(77, 100.5)
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_xticklabels([f"{t}%" for t in [80, 85, 90, 95, 100]], fontsize=12)
    clean_axes(ax)
    save(fig, "fig3_model_comparison.png")


def fig4():
    rows = [("Total kW only (1 channel)", "gbm_total.json"),
            ("Electricity only (6 channels)", "gbm_elec.json"),
            ("All sub-meters (10 channels)", "gbm_all.json")]
    acc = [mean_sd(f)[0] for _, f in rows]
    err = [100 - a for a in acc]
    cut = (1 - err[2] / err[0]) * 100
    fig = plt.figure(figsize=(13.4, 4.6))
    header(fig, "Fig. 4. Sub-meter channels cut the error rate by more than half",
           f"LightGBM, 4-fold CV error rate (100 − accuracy, %). All 10 channels vs total kW: {cut:.0f}% fewer errors")
    ax = fig.add_axes([0.33, 0.18, 0.62, 0.55])
    for i, (e, a) in enumerate(zip(err, acc)):
        final = i == 2
        ax.barh(i, e, height=0.52, color=BLUE if final else GREY)
        ax.text(e + 0.06, i, f"{e:.2f}% error ({a:.2f}% accuracy)", va="center", fontsize=12.5,
                color=INK if final else MUTED, fontweight="bold" if final else "normal")
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
    ax.get_yticklabels()[2].set_fontweight("bold"); ax.get_yticklabels()[2].set_color(INK)
    ax.set_ylim(2.5, -0.5); ax.set_xlim(0, 6.4)
    ax.set_xticks([])
    clean_axes(ax, xgrid=False)
    ax.axvline(0, color="#bbbbbb", lw=1)
    save(fig, "fig4_submeter_channels.png")


def fig5():
    # Single-run fold-0 ablations (METHODOLOGY_AND_RESULTS.txt, table B1 rows 8-12).
    rows = [("MLP on statistics", "+ signed-log transform", 92.94,
             load("cnn_all_w0_d64_feat.json")["acc"][0] * 100),
            ("CNN on hourly curves", "+ statistics branch",
             load("cnn_all_w32.json")["acc"][0] * 100, 98.32),
            ("Hybrid CNN", "+ signed-log transform", 97.97, 98.32)]
    fig = plt.figure(figsize=(13.4, 5.4))
    header(fig, "Fig. 5. The signed-log transform and the statistics branch both lift accuracy",
           "Fold-0 accuracy (%), grey dot before the change, blue dot after; label = gain in points")
    ax = fig.add_axes([0.38, 0.17, 0.56, 0.58])
    for i, (_, _, b, a) in enumerate(rows):
        ax.plot([b, a], [i, i], color="#9fc2ea", lw=4, solid_capstyle="butt", zorder=2)
        ax.plot(b, i, "o", color=GREY, ms=12, zorder=3)
        ax.plot(a, i, "o", color=BLUE, ms=12, zorder=3)
        ax.text((a + b) / 2 if a - b > 1 else a - 0.2, i - 0.22, f"+{a - b:.2f}",
                ha="center", va="bottom", fontsize=12.5, fontweight="bold")
    ax.set_yticks(range(3)); ax.set_yticklabels([""] * 3)
    for i, (t, s, _, _) in enumerate(rows):
        ax.text(-0.025, i - 0.08, t, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=12.5, fontweight="bold")
        ax.text(-0.025, i + 0.02, s, transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=12.5, color=MUTED)
    ax.set_ylim(2.5, -0.7); ax.set_xlim(92, 99.2)
    ax.set_xticks(range(92, 100))
    ax.set_xticklabels([f"{t}%" for t in range(92, 100)], fontsize=12)
    clean_axes(ax)
    save(fig, "fig5_feature_ablations.png")


def fig6():
    ev = load("tflite_eval.json")
    rows = [("LightGBM (CV ceiling)", load("gbm_size_sweep.json")[-1]["kb"], False),
            ("Hybrid CNN, float32", ev["float"]["kb"], False),
            ("Hybrid CNN, int8 (final)", ev["int8"]["kb"], True)]
    ratio = ev["float"]["kb"] / ev["int8"]["kb"]
    cost = (ev["float"]["acc"] - ev["int8"]["acc"]) * 100
    fig = plt.figure(figsize=(13.4, 4.9))
    header(fig, "Fig. 6. Only the int8 model fits the 100 KB budget of the meter",
           f"Model file size (KB). int8 is {ratio:.1f} times smaller than float32 for a {cost:.2f}-point accuracy cost")
    ax = fig.add_axes([0.30, 0.17, 0.65, 0.6])
    for i, (_, kb, final) in enumerate(rows):
        ax.barh(i, kb, height=0.52, color=BLUE if final else GREY, zorder=2)
        if kb > 500:
            ax.text(kb - 15, i, f"{kb:.1f} KB", va="center", ha="right", fontsize=12.5, zorder=3)
        else:
            ax.text(max(kb, 100) + 15 + (12 if final else 0), i, f"{kb:.1f} KB", va="center", fontsize=12.5,
                    color=INK if final else MUTED, fontweight="bold" if final else "normal")
    ax.axvline(100, color=MUTED, ls=(0, (3, 3)), lw=1.3, zorder=4)
    ax.text(110, -0.55, "100 KB budget", fontsize=11.5, color=MUTED, va="bottom")
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
    ax.get_yticklabels()[2].set_fontweight("bold"); ax.get_yticklabels()[2].set_color(INK)
    ax.set_ylim(2.5, -0.55); ax.set_xlim(0, 1050)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels([f"{t} KB" for t in [0, 250, 500, 750, 1000]], fontsize=12)
    clean_axes(ax)
    save(fig, "fig6_model_size.png")


def fig7():
    cm = read_holdout_cm()
    recall = np.diag(cm) / cm.sum(1)
    prec = np.diag(cm) / cm.sum(0)
    fig = plt.figure(figsize=(13.4, 7.1))
    header(fig, "Fig. 7. Theft1 and Theft6 are the classes most often missed",
           f"int8 model, hold-out fold 3 ({cm.sum():,} days). Recall below 0.94 is labelled")
    fig.text(0.04, 0.80, "●", color=BLUE, fontsize=15, va="center")
    fig.text(0.058, 0.80, "Recall", fontsize=12.5, va="center")
    fig.text(0.14, 0.80, "○", color=GREY, fontsize=17, va="center", fontweight="bold")
    fig.text(0.162, 0.80, "Precision", fontsize=12.5, va="center")
    ax = fig.add_axes([0.20, 0.14, 0.73, 0.62])
    low = recall < 0.94
    for i in range(7):
        ax.plot([recall[i], prec[i]], [i, i], color=GREY, lw=2, zorder=2)
        ax.plot(prec[i], i, "o", mfc="white", mec=GREY, mew=2.2, ms=12, zorder=3)
        ax.plot(recall[i], i, "o", color=BLUE, ms=12, zorder=4)
        if low[i]:
            ax.text(recall[i], i - 0.28, f"recall {recall[i]:.3f}", ha="center", va="bottom",
                    fontsize=12.5, fontweight="bold")
    ax.set_yticks(range(7)); ax.set_yticklabels(CLASSES, fontsize=12.5)
    for t, l in zip(ax.get_yticklabels(), low):
        if l:
            t.set_fontweight("bold"); t.set_color(INK)
    ax.tick_params(axis="y", pad=22)
    ax.set_ylim(6.5, -0.6); ax.set_xlim(0.92, 1.0)
    ax.set_xticks([0.92, 0.94, 0.96, 0.98, 1.0])
    ax.set_xticklabels(["0.92", "0.94", "0.96", "0.98", "1.00"], fontsize=12)
    clean_axes(ax)
    save(fig, "fig7_per_class_precision_recall.png")


def fig8():
    cm = read_holdout_cm()
    n = cm.sum()
    wrong = n - np.trace(cm)
    n_t1 = cm[0, 1] + cm[1, 0]
    off = cm.copy(); np.fill_diagonal(off, 0)
    vmax = off.max()
    fig = plt.figure(figsize=(13.4, 7.5))
    header(fig, "Fig. 8. Most hold-out errors fall between Normal and Theft1",
           f"{wrong} of {n:,} days misclassified; {n_t1} of them ({n_t1 / wrong:.0%}) between Normal "
           "and Theft1. Blue shade = error count")
    ax = fig.add_axes([0.24, 0.05, 0.68, 0.70])
    ax.set_xlim(0, 7); ax.set_ylim(7, -1.25); ax.axis("off")
    lo, hi = np.array(matplotlib.colors.to_rgb(BLUE_LIGHT)), np.array(matplotlib.colors.to_rgb("#5b9be3"))
    for r in range(7):
        for c in range(7):
            v = cm[r, c]
            if r == c:
                fc = GREY_BG
            elif v:
                t = np.sqrt(v / vmax)
                fc = lo + (hi - lo) * t
            else:
                fc = "white"
            ax.add_patch(FancyBboxPatch((c + 0.03, r + 0.04), 0.94, 0.92,
                                        boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=fc, ec="#e2e2e2", lw=1.2))
            if v:
                ax.text(c + 0.5, r + 0.5, f"{v:,}", ha="center", va="center", fontsize=12.5,
                        fontweight="normal" if r == c else "bold")
        ax.text(-0.2, r + 0.5, CLASSES[r], ha="right", va="center", fontsize=12.5, color=MUTED)
    for c in range(7):
        ax.text(c + 0.5, -0.4, SHORT[c], ha="center", va="center", fontsize=12.5, color=MUTED)
    ax.text(3.5, -1.05, "Predicted class", ha="center", va="center", fontsize=12.5, fontweight="bold")
    ax.text(-0.2, -0.4, "True class", ha="right", va="center", fontsize=12.5, fontweight="bold")
    save(fig, "fig8_confusion_matrix.png")


def fig9():
    rec = read_synth_recall()
    scen = [("Realistic", "realistic", GREY), ("Rescaled", "new_cons", GREY_LIGHT),
            ("Unseen variants", "paper_spec", BLUE)]
    days = 17472
    worst = rec["paper_spec"][6] * 100
    fig = plt.figure(figsize=(13.4, 6.9))
    header(fig, "Fig. 9. Only the time-reversed T6 defeats the model",
           f"Per-class recall (%), {days:,} synthetic days per scenario. "
           f"Lowest: T6 under unseen variants, {worst:.1f}%")
    for k, (label, _, col) in enumerate(scen):
        x = 0.035 + k * 0.17
        fig.text(x, 0.80, "■", color=col, fontsize=17, va="center")
        fig.text(x + 0.022, 0.80, label, fontsize=12.5, va="center")
    ax = fig.add_axes([0.08, 0.15, 0.89, 0.58])
    x = np.arange(7)
    bw = 0.26
    for k, (_, key, col) in enumerate(scen):
        ax.bar(x + (k - 1) * (bw + 0.02), rec[key] * 100, width=bw, color=col, zorder=2)
    ax.text(6 + bw + 0.02, worst + 2, f"{worst:.1f}%", ha="center", va="bottom", fontsize=12.5,
            fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=12.5)
    ax.get_xticklabels()[6].set_fontweight("bold"); ax.get_xticklabels()[6].set_color(INK)
    ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]], fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    ax.axhline(0, color="#999999", lw=1)
    fig.text(0.08, 0.04, "For Normal, recall is the share of normal days left unflagged "
             "(1 − false-alarm rate).", fontsize=11.5, color=MUTED)
    save(fig, "fig9_stress_test_recall.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        f()
