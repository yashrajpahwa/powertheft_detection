# TinyML Electricity-Theft Detection (TDD2022)

A tiny int8 TensorFlow Lite model that classifies each consumer-day as Normal or one of 6 theft
types, using the TDD2022 dataset (`df.csv`, Zidi et al., JKSU-CIS 2023).

## Results (leak-free: test consumers are never seen in training)

| Model | Input | 7-class acc | Macro-F1 | Size |
|---|---|---|---|---|
| Paper-style RF (single hourly row), our split | 1 hour | 81.1% | 0.640 | — |
| Paper's reported RF (their protocol) | 1 hour | 85.0% | 0.841 (F-measure) | — |
| LightGBM, day features (accuracy ceiling) | day + history | 98.74% | 0.980 | ~965 KB |
| Hybrid CNN, float, 4-fold CV | day + history | 98.15% | 0.973 | 157 KB |
| **Hybrid CNN, int8 TFLite** (held-out consumers) | day + history | **97.89%** | **0.970** | **64 KB** |

int8 quantization costs 0.05 points. Binary theft-vs-normal accuracy is 98.9%.
Inference takes 37 µs on a laptop CPU.

## Why the paper is capped at about 85%
It classifies each hour in isolation. In the actual data, theft is injected per **day**, and
**Theft6 is an exact replay of the previous day** (not a "reversal", as the paper says).
Theft5 is a smoothed near-mean curve, and Theft3 and Theft4 are per-hour random scalings.
None of these can be seen in a single hour. Our model sees today's 24 h profile next to
yesterday and a 7-day history mean.

## Model
Inputs per day (all 10 meter channels):
- `seq` (24 × 40): today, yesterday, 7-day history mean, |today − yesterday|, each scaled by the history level
- `lvl` (10): log history level per channel
- `feat` (180): 18 engineered stats per channel (ratio vs. history, zero runs, roughness,
  replay distance, correlations, …), signed-log scaled

Architecture: a Conv1D stem, 3 residual dilated separable-conv blocks, average and max pooling,
concatenation with `lvl` + `feat`, then two Dense(64) layers and a 7-way softmax. 37k parameters.

## Usage
```
pip install -r requirements.txt
python src/data.py --check            # build cache, sanity checks
python src/train_baseline.py          # paper-style RF baseline
python src/train_gbm.py               # LightGBM ceiling (--channels all|elec|total)
python src/train_cnn.py --feat --width 48 --dense 64   # 4-fold CV of the TinyML model
python src/export_tflite.py --width 48 --dense 64      # train, quantize, save models/*.tflite
python src/run_tflite.py              # evaluate saved int8 model on held-out consumers
python src/run_tflite.py --consumer 2 --days 20        # per-day predictions / alerts
```

## Protocol notes
- 64 consumers (16 building types × 4) × 365 days. 15 all-zero junk rows are dropped.
- Folds group by consumer, with one consumer of each building type per fold.
- Day label = the dominant theft label of the day (Theft2 labels only its zeroed hours).
  Day 0 has no history and is excluded.
- Channel ablation (LightGBM): all 10 channels 98.74%, electricity-only 98.45%, total kW only 96.77%.
  The theft generator modifies the gas channels too, so they carry no "clean reference" leak.
