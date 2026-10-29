"""Load TDD2022 (df.csv) into a (consumer, day, hour, channel) tensor with day labels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "df.csv"
CACHE = ROOT / "cache" / "days.npz"

CLASSES = ["Normal", "Theft1", "Theft2", "Theft3", "Theft4", "Theft5", "Theft6"]
HOURS_PER_YEAR = 8760
ELEC = list(range(6))      # Facility, Fans, Cooling, Heating, InteriorLights, InteriorEquipment
GAS = list(range(6, 10))   # Gas:Facility, Heating:Gas, InteriorEquipment:Gas, WaterHeater:Gas
FEATURE_SETS = {"total": [0], "elec": ELEC, "all": ELEC + GAS}


def load(use_cache=True):
    """Returns dict with X (64,365,24,10) float32, yh (64,365,24) int8 hourly labels,
    yd (64,365) int8 day labels, btype (64,) building-type index, type_names."""
    if use_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    df = pd.read_csv(CSV)
    df = df[df["Class"] != "0"].reset_index(drop=True)  # 15 all-zero junk rows
    n_cons = len(df) // HOURS_PER_YEAR
    assert n_cons * HOURS_PER_YEAR == len(df)

    X = df.iloc[:, 1:11].to_numpy(np.float32).reshape(n_cons, 365, 24, 10)
    yh = df["theft"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(np.int8)
    yh = yh.reshape(n_cons, 365, 24)
    types = df["Class"].to_numpy().reshape(n_cons, HOURS_PER_YEAR)
    assert (types == types[:, :1]).all(), "consumer blocks are not aligned"
    type_names = np.array(sorted(set(types[:, 0])))
    btype = np.searchsorted(type_names, types[:, 0])

    # Day label: most frequent theft label present that day, else Normal.
    yd = np.zeros(yh.shape[:2], np.int8)
    for k in range(1, len(CLASSES)):
        cnt = (yh == k).sum(-1)
        best = (yh == yd[..., None]).sum(-1) * (yd > 0)
        yd = np.where(cnt > best, k, yd).astype(np.int8)

    out = dict(X=X, yh=yh, yd=yd, btype=btype, type_names=type_names)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def group_folds(btype, n_folds=4, seed=0):
    """Fold id per consumer. Each building type has 4 consumers; put one of each
    type in every fold so all folds see every type in training (unseen *consumer*)."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(btype), int)
    for t in np.unique(btype):
        idx = np.where(btype == t)[0]
        fold[idx] = rng.permutation(len(idx)) % n_folds
    return fold


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = load(use_cache=not a.check)
    X, yh, yd, bt = d["X"], d["yh"], d["yd"], d["btype"]
    print("X", X.shape, "types", len(d["type_names"]), "consumers/type", np.bincount(bt))
    print("hour labels", np.bincount(yh.ravel()))
    print("day labels ", np.bincount(yd.ravel()))
    mixed = ((yh != yh[..., :1]).any(-1)).sum()
    print("days with mixed hourly labels:", mixed)
    f = group_folds(bt)
    for k in range(f.max() + 1):
        assert len(set(np.where(f == k)[0]) & set(np.where(f != k)[0])) == 0
    print("folds", np.bincount(f), "OK")
