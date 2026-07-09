"""Dataset utilities for simulation-based pretraining.

Stage 1: Shuffled timestamps (ViPer-RV methodology)
Stage 2: Ordered timestamps (realistic cadence)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sim_utils import simulate_dataset, bjd_positional_encoding
import pandas as pd


# Must match the Transformer's timestamp encoding exactly
NUM_FREQS = 8
MIN_PERIOD = 1.0
MAX_PERIOD = 7300.0
PERIODS = np.logspace(np.log10(MIN_PERIOD), np.log10(MAX_PERIOD), NUM_FREQS)
FREQS = 2.0 * np.pi / PERIODS


def star_to_features(star_data):
    """Convert a simulated star dict into the (seq_len, 21) feature array.

    Must match the format used in transformer.ipynb Cell 1:
    [rv_centered, rv_err, exposure_time, rhkp, halpha] + 16 timestamp encoding dims
    """
    t = star_data['bjd']
    ref_bjd = t[0]
    n = len(t)

    features = []
    for i in range(n):
        pos_enc = []
        dt = t[i] - ref_bjd
        for f in FREQS:
            pos_enc.append(np.sin(f * dt))
            pos_enc.append(np.cos(f * dt))

        row = [
            star_data['rv_centered'][i],
            star_data['rv_err'][i],
            star_data['exposure_time'][i],
            star_data['rhkp'][i],
            star_data['halpha'][i],
        ] + pos_enc
        features.append(row)

    return np.array(features, dtype=np.float32)


def star_to_features_shuffled(star_data):
    """Stage 1: Shuffled timestamps.

    Shuffle the observation order and re-assign synthetic timestamps
    spanning a uniform range. This forces the model to learn Doppler
    SHAPES, not time-correlated activity patterns.
    """
    n = len(star_data['bjd'])

    # Shuffle observation indices
    perm = np.random.permutation(n)

    # Synthetic uniform timestamps (1-2 year span, per ViPer-RV §4)
    new_bjd = np.linspace(0, 500, n) + np.random.normal(0, 5, n)

    # Create shuffled star dict
    shuffled_star = {
        'bjd': new_bjd,
        'rv_centered': star_data['rv_centered'][perm],
        'rv_err': star_data['rv_err'][perm],
        'exposure_time': star_data['exposure_time'][perm],
        'rhkp': star_data['rhkp'][perm],
        'halpha': star_data['halpha'][perm],
        'has_exoplanets': star_data['has_exoplanets'],
    }

    return star_to_features(shuffled_star)


class SimDataset(Dataset):
    """Dataset of simulated stars for pretraining.

    Mode 'shuffled': Stage 1 (ViPer-RV approach)
    Mode 'ordered': Stage 2 (realistic cadence)
    """
    def __init__(self, sim_stars, mode='shuffled', seed=42):
        self.mode = mode
        self.labels = [s['has_exoplanets'] for _, s in sim_stars]

        if mode == 'shuffled':
            self.data = [star_to_features_shuffled(s) for _, s in sim_stars]
        else:
            self.data = [star_to_features(s) for _, s in sim_stars]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)


def collate_stars(batch):
    """Same collate function as transformer.ipynb — pad variable-length sequences.

    Returns: (padded, mask, labels) where padded is (B, max_seq_len, 21).
    """
    stars, labels = zip(*batch)
    max_len = max(s.shape[0] for s in stars)

    padded, mask = [], []
    for star in stars:
        seq_len = star.shape[0]
        pad_len = max_len - seq_len
        if pad_len > 0:
            padding = torch.zeros(pad_len, star.shape[1])
            padded_star = torch.cat([star, padding], dim=0)
        else:
            padded_star = star
        star_mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        padded.append(padded_star)
        mask.append(star_mask)

    return torch.stack(padded), torch.stack(mask), torch.stack(labels)


def get_sim_loaders(observations, n_sim=50000, batch_size=32,
                    shuffle=True, mode='shuffled', seed=42,
                    device='cpu'):
    """Generate simulated data and return train/val DataLoaders."""

    # Generate simulated stars
    sim_stars = simulate_dataset(observations, n_sim_stars=n_sim,
                                  positive_fraction=0.5, seed=seed)

    # Standardize using simulated data statistics (computed on train subset)
    # Split sim data 80/20 train/val
    n_val = len(sim_stars) // 5
    np.random.seed(seed)
    indices = np.random.permutation(len(sim_stars))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_stars = [sim_stars[i] for i in train_idx]
    val_stars = [sim_stars[i] for i in val_idx]

    train_ds = SimDataset(train_stars, mode=mode, seed=seed)
    val_ds = SimDataset(val_stars, mode=mode, seed=seed)

    # Compute normalization stats from training set
    all_train = np.concatenate([d.numpy() for d, _ in [train_ds[i] for i in range(min(1000, len(train_ds)))]], axis=0)
    feat_mean = all_train.mean(axis=0)
    feat_std = np.clip(all_train.std(axis=0), 1e-8, None)

    # Apply normalization
    for i in range(len(train_ds)):
        train_ds.data[i] = (train_ds.data[i] - feat_mean) / feat_std
    for i in range(len(val_ds)):
        val_ds.data[i] = (val_ds.data[i] - feat_mean) / feat_std

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              collate_fn=collate_stars, pin_memory=(device=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_stars, pin_memory=(device=='cuda'))

    return train_loader, val_loader, feat_mean, feat_std
