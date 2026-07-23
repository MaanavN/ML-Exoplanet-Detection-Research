"""
Shared train/val/test star split for the exoplanet ML project.

Problem
-------
The modeling notebooks (baseline, xgboost, cnn, cnn_stripped, cnn_hybrid,
transformer) each constructed their own train/val/test split with different
star-orderings:
  - baseline & xgboost : observations.groupby('star_name')  -> sorted alphabetically
  - cnn / cnn_stripped  : list(set(...))                     -> PYTHONHASHSEED-dependent (non-reproducible)
  - cnn_hybrid          : observations['star_name'].unique() -> insertion order (deterministic, but != sorted)
  - transformer         : list(set(...)) split per-class     -> PYTHONHASHSEED-dependent (non-reproducible)
So the headline cross-notebook comparison numbers were computed on *different*
test stars, and three of the notebooks were not even reproducible run-to-run.

Contract
--------
This module reproduces baseline.ipynb's canonical split *exactly* (sorted star
names, 60/20/20 via two train_test_split calls, stratified, random_state=42)
once, and pins it to split.json as three star-name lists. Every modeling
notebook loads those lists and partitions its per-star features/sequences by
membership -- so every model is evaluated on the same test stars, and the split
no longer depends on hash ordering.

Usage in a notebook
------------------
    from split import load_split
    train_stars, val_stars, test_stars = load_split()   # reads split.json
    # then assign each star to a partition by name membership

split.json is committed to the repo, so notebooks work without the raw pickle.
To (re)generate split.json the raw observations.pkl is required -- run
``python split.py`` from the repo root after running data_prep.ipynb on Kaggle
(observations.pkl lives at /kaggle/input/datasets/maanav0114/harps-n-dataset/).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

# Same path every modeling notebook reads from.
OBSERVATIONS_PKL = "/kaggle/input/datasets/maanav0114/harps-n-dataset/observations.pkl"
SPLIT_JSON = Path(__file__).resolve().parent / "split.json"

# Canonical split hyperparameters -- must match baseline.ipynb exactly.
SEED = 42
TEST_SIZE_STAGE_1 = 0.4   # 60 / 40
TEST_SIZE_STAGE_2 = 0.5   # 40 -> 20 / 20
LABEL_COLUMN = "has_exoplanets"


def build_star_labels(observations):
    """Return (sorted_star_names, y) where y[i] is the label for sorted_star_names[i].

    Mirrors baseline.ipynb: observations.groupby('star_name') with default sort=True
    yields stars in alphabetical order; the label is the (per-star constant)
    value of has_exoplanets for any of that star's observations.
    """
    grouped = observations.groupby("star_name", sort=True)[LABEL_COLUMN]
    star_names = list(grouped.groups.keys())          # sorted alphabetically
    # Take the first label value per star (it is constant per star by construction).
    y = np.array([int(grouped.get_group(s).iloc[0]) for s in star_names], dtype=int)
    return star_names, y


def compute_split(observations):
    """Compute the canonical 60/20/20 stratified split and return three star-name lists.

    Reproduces baseline.ipynb cells 2-3: groupby-sort star order, two-stage
    train_test_split (test_size=0.4 then 0.5 on the temp), stratify on the
    label vector of the stage input, random_state=42. Returns (train, val, test)
    star-name lists in the *split order* produced by train_test_split.
    """
    from sklearn.model_selection import train_test_split

    star_names, y = build_star_labels(observations)
    idx = np.arange(len(star_names))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=TEST_SIZE_STAGE_1, random_state=SEED, stratify=y,
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=TEST_SIZE_STAGE_2, random_state=SEED, stratify=y_temp,
    )

    def names(idxs):
        return [star_names[i] for i in idxs]

    return names(idx_train), names(idx_val), names(idx_test)


def save_split(observations, path=SPLIT_JSON):
    """Compute the canonical split and write three star-name lists to split.json."""
    train, val, test = compute_split(observations)
    payload = {
        "contract": "60/20/20 stratified split by star; sorted star order; "
                    "random_state=42. See split.py. Do not edit by hand.",
        "seed": SEED,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train": train,
        "val": val,
        "test": test,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved split to {path}: train={len(train)} val={len(val)} test={len(test)}")
    return payload


def load_split(path=SPLIT_JSON):
    """Load the shared train/val/test star-name lists from split.json.

    Returns (train_stars, val_stars, test_stars) as sorted-independent lists.
    Every modeling notebook calls this instead of constructing its own split.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python split.py` on a machine with access to "
            f"{OBSERVATIONS_PKL} (i.e. after running data_prep.ipynb on Kaggle) "
            "to (re)generate split.json, then commit it to the repo."
        )
    with open(path) as f:
        payload = json.load(f)
    return list(payload["train"]), list(payload["val"]), list(payload["test"])


def ensure_split(observations, path=SPLIT_JSON):
    """Return (train, val, test) star-name lists, building split.json if missing.

    Modeling notebooks call this with their loaded `observations` DataFrame so
    the split is materialized once (from observations.pkl) even if split.json
    has not been pre-committed. Idempotent: if split.json already exists it is
    loaded as-is.
    """
    if os.path.exists(path):
        return load_split(path)
    return compute_split(observations)


def bootstrap_roc_auc(y_true, y_score, n_resamples=200, seed=42):
    """Bootstrap 95% CI for ROC-AUC via the percentile method.

    Resamples (y_true, y_score) pairs with replacement n_resamples times,
    computes ROC-AUC on each resample, and returns (point_estimate, lo, hi)
    where lo/hi are the 2.5th and 97.5th percentiles of the bootstrap
    distribution. The point estimate is the AUC on the full (un-resampled) data.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(y_true) != len(y_score):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_score={len(y_score)}")
    if len(y_true) < 2:
        raise ValueError("need at least 2 samples to bootstrap ROC-AUC")

    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        yt = y_true[sample_idx]
        ys = y_score[sample_idx]
        # ROC-AUC is undefined if a resample contains only one class; resample once more.
        attempts = 0
        while len(np.unique(yt)) < 2 and attempts < 10:
            sample_idx = rng.integers(0, n, size=n)
            yt = y_true[sample_idx]
            ys = y_score[sample_idx]
            attempts += 1
        if len(np.unique(yt)) < 2:
            aucs[i] = point  # fall back to point estimate if degenerate
            continue
        aucs[i] = roc_auc_score(yt, ys)
    lo = float(np.percentile(aucs, 2.5))
    hi = float(np.percentile(aucs, 97.5))
    return point, lo, hi


def bootstrap_pr_auc(y_true, y_score, n_resamples=200, seed=42):
    """Bootstrap 95% CI for PR-AUC (average precision) via the percentile method.

    Resamples (y_true, y_score) pairs with replacement n_resamples times,
    computes average_precision_score on each resample, and returns
    (point_estimate, lo, hi) where lo/hi are the 2.5th and 97.5th percentiles.
    The point estimate is the AP on the full (un-resampled) data.
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(y_true) != len(y_score):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_score={len(y_score)}")
    if len(y_true) < 2:
        raise ValueError("need at least 2 samples to bootstrap PR-AUC")

    point = float(average_precision_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aps = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        yt = y_true[sample_idx]
        ys = y_score[sample_idx]
        # PR-AUC is undefined if a resample contains only one class; resample once more.
        attempts = 0
        while len(np.unique(yt)) < 2 and attempts < 10:
            sample_idx = rng.integers(0, n, size=n)
            yt = y_true[sample_idx]
            ys = y_score[sample_idx]
            attempts += 1
        if len(np.unique(yt)) < 2:
            aps[i] = point  # fall back to point estimate if degenerate
            continue
        aps[i] = average_precision_score(yt, ys)
    lo = float(np.percentile(aps, 2.5))
    hi = float(np.percentile(aps, 97.5))
    return point, lo, hi


def main():
    """Recompute split.json from observations.pkl (requires the Kaggle dataset path)."""
    import pandas as pd
    if not os.path.exists(OBSERVATIONS_PKL):
        raise FileNotFoundError(
            f"{OBSERVATIONS_PKL} not found. Run data_prep.ipynb on Kaggle to produce "
            "observations.pkl and make it available at the path above, then re-run."
        )
    observations = pd.read_pickle(OBSERVATIONS_PKL)
    save_split(observations)


if __name__ == "__main__":
    main()
