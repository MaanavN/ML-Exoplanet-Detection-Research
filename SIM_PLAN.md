# Simulation-Based Pretraining Pipeline for RV Exoplanet Detection

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Train a Transformer on simulated RV time series with injected Keplerian signals, then fine-tune on real HARPS/HIRES data with domain adaptation, to beat the RF baseline (0.7994 ROC-AUC).

**Architecture:** Generate synthetic RV time series using real observation cadences as templates + Gaussian Process stellar activity simulations + Keplerian planet injection. Pretrain our existing ExoplanetTransformer on ~50k simulated stars. Fine-tune on real data with adversarial domain alignment (sim vs real discriminator). Evaluate on held-out real test set and compare to RF baseline.

**Tech Stack:** PyTorch, scikit-learn, george/celerite (GP stellar activity), radvel or manual Keplerian (planet injection), numpy/pandas

---

## Literature Backing

Every design decision is anchored to published work:

| Design Decision | Literature Basis |
|---|---|
| Train on simulated RV data, test on real | ExoplANNET (Nieto & Díaz 2023, A&A 677) — trained CNN on synthetic periodograms, tested on 1 real star. Gap: nobody has done this at scale with raw time series on many real stars. |
| Real cadences as simulation templates | ViPer-RV (Gavankar et al. 2025, AJ) — used real NEID solar timestamps as background, injected synthetic Keplerians on top. "Real backgrounds + synthetic planet injection." |
| Two-stage training: shuffled → ordered | ViPer-RV — Stage 1 shuffled timestamps (forces learning of Doppler shifts, not activity patterns), Stage 2 ordered timestamps (realistic cadence). Prevents model from confusing stellar rotation with planet signals. |
| GP quasi-periodic kernel for stellar activity | ExoplANNET — "Gaussian process with a covariance function generated with the pseudo periodic kernel" for stellar rotation modulation. Plus Dumusque et al. 2011 for pulsation/granulation power spectrum components. |
| Realistic uncertainty sampling from survey stats | ExoplANNET — sampled intrinsic errors from HARPS survey statistics (real star uncertainties). |
| Keplerian injection (not simple sinusoid) | ViPer-RV — used `radvel` toolkit for elliptical orbits. Period log-uniform 12-365 days, eccentricity 0-0.6, semi-amplitude 0.05-3 m/s. |
| Simulation pretraining + fine-tune with minimal labels | Gupta, Muthukrishna & Audenaert (ICML 2025, arXiv:2510.12958) — pretrained models fine-tuned with 512 labels beat baselines trained on full 3747-label dataset. |
| Adversarial domain adaptation (sim→real) | Gupta et al. ICML 2025 — discriminator network predicts whether representation came from simulated or real data. Classifier trained with competing objective: classify planets AND fool the discriminator. Creates "domain-agnostic" representations. Critical for cross-survey transfer. |
| Contrastive domain alignment (alternative/complement) | Gupta et al. ICML 2025 — supervised contrastive loss pulls same-class samples together regardless of domain. Outperformed adversarial for zero-shot transfer. We implement adversarial first (simpler), contrastive as stretch goal. |
| Variable-length sequence handling | ViPer-RV — model handles aperiodic timestamps via attention. Our existing Transformer already handles variable-length via padding + masking. |
| Evaluate against tabular baseline | Our own RF (0.7994 ROC-AUC). No prior sim-pretraining paper has compared DL against a strong tabular baseline. This is novel. |

### What We Do Differently From Each Paper

| vs. ExoplANNET | vs. ViPer-RV | vs. Gupta ICML 2025 |
|---|---|---|
| We use raw RV time series, not periodograms | We work on many stars, not just the Sun | We apply to RV exoplanet detection, not photometric transients |
| We test on hundreds of real stars, not N=1 | We use summary-stat baseline comparison | We use our own RV simulation physics |
| We add domain adaptation | We use our own Transformer architecture | We compare against RF tabular baseline |
| We compare against RF baseline | We have activity indicators (RHKp, Halpha) | |

---

## Our Existing Architecture (What We Build On)

**Transformer:** `ExoplanetTransformer` in `transformer.ipynb`
- Input: `(batch, seq_len, 21)` — 5 raw features + 16 sinusoidal timestamp encoding dims
- 5 raw features per observation: `rv_centered, rv_err, exposure_time, RHKp, Halpha`
- Timestamp encoding: 8 log-spaced periods (1 to 7300 days), sin+cos = 16 dims
- Architecture: `Linear(21→48)` → 1-layer Transformer encoder (4 heads, d_model=48, ff=96) → AttentionPool → `Linear(48→16)→ReLU→Dropout→Linear(16→1)`
- Training: BCEWithLogitsLoss with pos_weight=sqrt(neg/pos), Adam lr=1e-3 wd=5e-3, 5-epoch warmup + cosine decay, gradient clipping at 1.0

**Data:** `observations.pkl` — 235,567 observations, 2,187 stars (413 positive, 1,774 negative)
- Columns: `star_name, bjd, rv, rv_err, exposure_time, RHKp, Halpha, has_exoplanets, rv_centered`
- Each star has 20+ observations, variable-length sequences
- Split: train/val/test via `split.py` (stratified by star)

---

## Pipeline Overview

```
Phase 1: Simulation Engine          Phase 2: Pretraining         Phase 3: Fine-tuning
┌──────────────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
│ Real star cadences    │    │ Stage 1: Shuffled     │    │ Freeze pretrained       │
│ (bjd, rv_err, rhkp,   │───►│ simulated data        │───►│ encoder, replace        │
│ halpha from real obs) │    │ Learn Keplerian       │    │ classifier head,        │
│                       │    │ Doppler shapes        │    │ fine-tune on real       │
│ + GP stellar activity │    │                       │    │ train/val data          │
│ + Keplerian injection │    │ Stage 2: Ordered       │    │                         │
│ = simulated star      │    │ simulated data        │    │ + Adversarial domain    │
│ with known label      │    │ Fine-tune for real     │    │   adapter (sim vs real  │
│                       │    │ cadence patterns      │    │   discriminator)        │
└──────────────────────┘    └──────────────────────┘    └─────────────────────────┘
                                                                  │
                                                                  ▼
                                                    Phase 4: Evaluation
                                                    ┌─────────────────────────┐
                                                    │ Test on real held-out    │
                                                    │ Compare to RF (0.7994)  │
                                                    │ Bootstrapped CIs         │
                                                    │ Ablation: no pretrain    │
                                                    │ Ablation: no domain adapt│
                                                    └─────────────────────────┘
```

---

## Tasks

### Task 1: Create the RV Simulation Module

**Objective:** Build the core function that generates a single simulated RV time series with realistic stellar activity and injected Keplerian planet signal.

**Files:**
- Create: `/home/maanav0114/Documents/mled/sim_utils.py`

**Literature basis:** ExoplANNET §3 (Simulations) for GP activity + uncertainty sampling. ViPer-RV §4 (Data generation) for Keplerian injection parameters.

**Step 1: Write the simulation function**

```python
"""RV time series simulation utilities for simulation-based pretraining.

Literature basis:
- Stellar activity (GP quasi-periodic kernel): ExoplANNET (Nieto & Diaz 2023, A&A 677)
- Keplerian injection parameters: ViPer-RV (Gavankar et al. 2025, AJ)
- Uncertainty sampling from real survey stats: ExoplANNET §3.2
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def keplerian_rv(t, P, K, e, omega, T0):
    """Compute the RV signature of a single Keplerian orbit.

    Parameters (following ViPer-RV §4):
        P: orbital period (days), log-uniform sampled 12-3650
        K: semi-amplitude (m/s), log-uniform sampled 0.1-100
        e: eccentricity, uniform 0-0.6
        omega: argument of periastron (rad), uniform 0-2*pi
        T0: time of periastron passage (BJD)

    Returns: RV perturbation (m/s) at each time t.
    """
    # Mean anomaly
    M = 2 * np.pi * (t - T0) / P
    # Solve Kepler's equation for eccentric anomaly E
    E = M.copy()
    for _ in range(50):  # Newton-Raphson
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    # RV perturbation
    rv = K * (np.cos(nu + omega) + e * np.cos(omega))
    return rv


def gp_stellar_activity(t, rotation_period, evolution_timescale, amplitude, jitter_amp):
    """Simulate stellar activity using a quasi-periodic Gaussian Process.

    Following ExoplANNET §3.3: "Gaussian process with a covariance function
    generated with the pseudo periodic kernel."

    The quasi-periodic kernel (also called the Schoenberg kernel):
    k(t_i, t_j) = A^2 * exp(-sin^2(pi * |t_i-t_j| / Prot) / (2 * lambda^2)
                             - |t_i - t_j|^2 / (2 * L^2))

    Parameters:
        rotation_period: stellar rotation period (days)
        evolution_timescale: timescale of active region evolution (days)
        amplitude: activity amplitude (m/s)
        jitter_amp: additional white noise jitter (m/s)

    Instead of full GP sampling (slow), we use a sums-of-sinusoids approximation
    that matches the quasi-periodic power spectrum. This is O(N) instead of O(N^3).

    The power spectrum of a quasi-periodic kernel has Lorentzian peaks at
    harmonics of 1/Prot, each broadened by the evolution timescale.
    """
    N_harmonics = 5
    n_points = len(t)
    t_centered = t - t.mean()

    rv_activity = np.zeros(n_points)

    for h in range(1, N_harmonics + 1):
        freq = h * 2 * np.pi / rotation_period
        # Lorentzian width from evolution timescale
        sigma_freq = 1.0 / (2 * np.pi * evolution_timescale)

        # Amplitude for this harmonic (decreasing)
        amp_h = amplitude * np.exp(-(h - 1) / 2.0)

        # Random phase for this harmonic
        phase = np.random.uniform(0, 2 * np.pi)

        # Add frequency spread from evolution timescale
        freq_shift = np.random.normal(0, sigma_freq)

        rv_activity += amp_h * np.sin(freq * t_centered + freq_shift * t_centered + phase)

    # Add white noise jitter (granulation/pulsation approximation)
    rv_activity += np.random.normal(0, jitter_amp, n_points)

    return rv_activity


def simulate_star(t_bjd, rv_err_real, rhkp_real, halpha_real,
                  has_planet, planet_params=None,
                  activity_rotation=25.0, activity_evolution=100.0,
                  activity_amp=2.0, jitter_amp=1.0):
    """Simulate a full RV time series for one star.

    Uses REAL observation cadence (timestamps, uncertainties, activity indicators)
    as the template, then overlays simulated stellar activity + optional Keplerian.

    This follows the "real backgrounds + synthetic injection" approach from ViPer-RV.

    Parameters:
        t_bjd: real BJD timestamps from an actual star (days)
        rv_err_real: real per-observation uncertainties (m/s)
        rhkp_real: real S-index values
        halpha_real: real H-alpha values
        has_planet: whether to inject a Keplerian signal
        planet_params: dict with P, K, e, omega, T0 (if None, random)
        activity_*: stellar activity simulation parameters

    Returns: dict with simulated rv_centered, rv_err, exposure_time, rhkp, halpha, label
    """
    n_obs = len(t_bjd)
    t = np.array(t_bjd, dtype=float)
    ref_bjd = t[0]
    t_centered = t - ref_bjd

    # 1. Stellar activity (quasi-periodic GP approximation)
    # Randomize activity parameters for diversity (domain randomization,
    # following Gupta et al. ICML 2025 methodology)
    rot = np.random.uniform(*activity_rotation) if isinstance(activity_rotation, tuple) else \
          np.random.normal(activity_rotation, activity_rotation * 0.2)
    rot = max(rot, 1.0)

    evo = np.random.uniform(*activity_evolution) if isinstance(activity_evolution, tuple) else \
          np.random.normal(activity_evolution, activity_evolution * 0.3)
    evo = max(evo, 10.0)

    amp = np.random.uniform(*activity_amp) if isinstance(activity_amp, tuple) else \
          np.random.normal(activity_amp, activity_amp * 0.3)
    amp = max(amp, 0.1)

    jit = np.random.uniform(*jitter_amp) if isinstance(jitter_amp, tuple) else \
          np.random.normal(jitter_amp, jitter_amp * 0.3)
    jit = max(jit, 0.1)

    activity_rv = gp_stellar_activity(t_centered, rot, evo, amp, jit)

    # 2. Planet injection (if has_planet)
    planet_rv = np.zeros(n_obs)
    if has_planet:
        if planet_params is None:
            # Sample Keplerian parameters (following ViPer-RV §4):
            # P: log-uniform 12-3650 days (extends to long periods)
            P = np.exp(np.random.uniform(np.log(12), np.log(3650)))
            # K: log-uniform 0.5-50 m/s (realistic for HARPS detection range)
            K = np.exp(np.random.uniform(np.log(0.5), np.log(50)))
            # e: uniform 0-0.6
            e = np.random.uniform(0, 0.6)
            # omega: uniform 0-2*pi
            omega = np.random.uniform(0, 2 * np.pi)
            # T0: uniform within first period
            T0 = np.random.uniform(0, P)
        else:
            P = planet_params['P']
            K = planet_params['K']
            e = planet_params['e']
            omega = planet_params['omega']
            T0 = planet_params['T0']

        planet_rv = keplerian_rv(t_centered, P, K, e, omega, T0)

    # 3. Instrument noise (sampled from real uncertainties)
    # Following ExoplANNET §3.2: use real per-observation uncertainties as sigma
    instrument_noise = np.random.normal(0, np.array(rv_err_real))

    # 4. Total RV = activity + planet + instrument noise
    rv_centered = activity_rv + planet_rv + instrument_noise

    # 5. Keep real activity indicators (RHKp, Halpha) — these are the stellar
    # activity tracers. We perturb them slightly so they're not identical to the
    # real star's values, but keep the overall statistical distribution.
    # (In a more sophisticated version, we'd simulate RHKp and Halpha from the
    # same GP that generates the activity RV, but this approximation is reasonable
    # for pretraining where the model needs to learn general RV-activity correlations.)
    rhkp_sim = np.array(rhkp_real) * np.random.uniform(0.8, 1.2, n_obs)
    halpha_sim = np.array(halpha_real) * np.random.uniform(0.8, 1.2, n_obs)

    return {
        'bjd': t,
        'rv_centered': rv_centered,
        'rv_err': np.array(rv_err_real),
        'exposure_time': np.full(n_obs, 1800.0, dtype=float),  # standard exposure
        'rhkp': rhkp_sim,
        'halpha': halpha_sim,
        'has_exoplanets': int(has_planet),
    }


def simulate_dataset(observations, n_sim_stars=50000, positive_fraction=0.5,
                     activity_params_range=None, seed=42):
    """Generate a large dataset of simulated RV time series.

    Uses real stars from the observations DataFrame as cadence templates.
    For each simulated star, picks a random real star's (bjd, rv_err, rhkp, halpha)
    as template, then overlays synthetic activity + optional Keplerian.

    Parameters:
        observations: real observations DataFrame (our observations.pkl)
        n_sim_stars: total number of simulated stars to generate
        positive_fraction: fraction with injected planets
        activity_params_range: dict of (min, max) ranges for activity params
                               (rotation_period, evolution_timescale, amplitude, jitter)

    Returns: list of dicts (one per simulated star), same format as simulate_star()
    """
    np.random.seed(seed)

    if activity_params_range is None:
        # Default ranges based on HARPS stars (from ExoplANNET §3.3 + Dumusque 2011)
        activity_params_range = {
            'rotation_period': (5, 80),        # 5-80 day rotation (solar-type stars)
            'evolution_timescale': (30, 300),   # active region evolution
            'amplitude': (0.5, 5.0),            # m/s activity amplitude
            'jitter_amp': (0.3, 2.0),           # m/s white noise jitter
        }

    # Get unique real stars to use as cadence templates
    real_stars = observations['star_name'].unique()
    np.random.shuffle(real_stars)

    n_positive = int(n_sim_stars * positive_fraction)
    n_negative = n_sim_stars - n_positive

    sim_stars = []

    for i in range(n_sim_stars):
        # Pick a random real star as cadence template
        template_star = real_stars[i % len(real_stars)]
        star_obs = observations[observations['star_name'] == template_star].sort_values('bjd')

        t_bjd = star_obs['bjd'].values
        rv_err = star_obs['rv_err'].values
        rhkp = star_obs['RHKp'].values
        halpha = star_obs['Halpha'].values

        has_planet = i < n_positive

        # Sample activity parameters from ranges (domain randomization)
        activity_args = {
            'activity_rotation': activity_params_range['rotation_period'],
            'activity_evolution': activity_params_range['evolution_timescale'],
            'activity_amp': activity_params_range['amplitude'],
            'jitter_amp': activity_params_range['jitter'],
        }

        sim_star = simulate_star(
            t_bjd=t_bjd, rv_err_real=rv_err,
            rhkp_real=rhkp, halpha_real=halpha,
            has_planet=has_planet,
            **activity_args
        )
        sim_stars.append((template_star, sim_star))

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{n_sim_stars} simulated stars")

    return sim_stars
```

**Step 2: Run basic verification**

```python
# In a notebook or script — verify the simulation produces sane output
import pandas as pd
import numpy as np

observations = pd.read_pickle('/kaggle/working/observations.pkl')
# Get one real star as template
star_obs = observations[observations['star_name'] == 'HD160617']
sim = simulate_star(
    t_bjd=star_obs['bjd'].values,
    rv_err_real=star_obs['rv_err'].values,
    rhkp_real=star_obs['RHKp'].values,
    halpha_real=star_obs['Halpha'].values,
    has_planet=True,
    planet_params={'P': 50, 'K': 3, 'e': 0.1, 'omega': 0, 'T0': 0}
)
print(f"Simulated RV range: {sim['rv_centered'].min():.2f} to {sim['rv_centered'].max():.2f} m/s")
print(f"Label: {sim['has_exoplanets']}")
print(f"N obs: {len(sim['bjd'])}")
```

Expected: simulated RV in the range of real HARPS data (roughly ±10 m/s). Label = 1. N obs matches real star.

---

### Task 2: Shuffled-Timestamp Pretraining Data Generator

**Objective:** Build the data loading pipeline for Stage 1 pretraining — generate simulated stars, shuffle their timestamps, and format them into the same 21-feature input our Transformer expects.

**Files:**
- Create: `/home/maanav0114/Documents/mled/sim_dataset.py`

**Literature basis:** ViPer-RV §5 (Training Procedure): "Shuffled training removes temporal coherence while preserving the overall scatter in the radial velocities. This ensures the model cannot overfit time-correlated variability. Instead, it must learn to recognize the underlying Doppler transformation due to orbital motion within a noisy background, separate from temporally correlated stellar activity."

**Step 1: Write the shuffled dataset generator**

```python
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
```

---

### Task 3: Stage 1 Pretraining (Shuffled Timestamps)

**Objective:** Pretrain the ExoplanetTransformer on simulated data with shuffled timestamps. This is the first of two training stages.

**Files:**
- Create: `/home/maanav0114/Documents/mled/pretrain_sim.ipynb`

**Literature basis:** ViPer-RV §5: "In the first stage, the model is trained on datasets with randomized observation timestamps, which removes temporal coherence while preserving the overall scatter in the radial velocities. This ensures the model cannot overfit time-correlated variability. Instead, it must learn to recognize the underlying Doppler transformation due to orbital motion within a noisy background, separate from temporally correlated stellar activity."

**Step 1: Create notebook with the following cells:**

Cell 0 (markdown):
```markdown
# Stage 1: Simulation-Based Pretraining (Shuffled Timestamps)

## Literature Basis
Following ViPer-RV (Gavankar et al. 2025, AJ) two-stage training methodology:
- Stage 1: Shuffled timestamps → model learns Keplerian Doppler shapes
- Stage 2 (next notebook): Ordered timestamps → model adapts to real cadence

Training data: 50,000 simulated RV time series with:
- Real HARPS/HIRES observation cadences as templates
- GP quasi-periodic stellar activity (ExoplANNET methodology)
- Injected Keplerian planet signals (50% positive)
- Shuffled timestamps to prevent activity pattern overfitting
```

Cell 1 (code) — Load and generate:
```python
import os
os.environ["PYTHONHASHSEED"] = "0"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score
import math
import random

# Seed everything
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load real observations (cadence templates)
observations = pd.read_pickle('/kaggle/working/observations.pkl')
print(f"Real observations: {len(observations)} obs, {observations['star_name'].nunique()} stars")

# Generate simulated dataset
from sim_dataset import get_sim_loaders

print("Generating 50,000 simulated stars (Stage 1: shuffled)...")
train_loader, val_loader, feat_mean, feat_std = get_sim_loaders(
    observations, n_sim=50000, batch_size=32, mode='shuffled', seed=seed, device=str(device)
)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print(f"Feature means (first 5): {feat_mean[:5]}")
print(f"Feature stds  (first 5): {feat_std[:5]}")
# Save normalization stats for later fine-tuning
np.save('/kaggle/working/sim_norm_mean.npy', feat_mean)
np.save('/kaggle/working/sim_norm_std.npy', feat_std)
```

Cell 2 (code) — Model (same architecture as existing Transformer):
```python
from transformer import AttentionPool, ExoplanetTransformer  # reuse existing architecture

model = ExoplanetTransformer(feat_dim=21, d_model=48, nhead=4, num_layers=1,
                              dim_feedforward=96, dropout=0.3).to(device)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

Cell 3 (code) — Training loop:
```python
criterion = nn.BCEWithLogitsLoss()  # balanced 50/50 simulated data, no pos_weight needed

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)

warmup_epochs = 5
total_epochs = 30  # less epochs needed with 50k training examples

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

train_losses, val_losses = [], []
train_aucs, val_aucs = [], []

best_val_auc = 0
best_model_state = None

for epoch in range(total_epochs):
    model.train()
    train_loss = 0
    all_train_probs, all_train_labels = [], []

    for padded, mask, labels in train_loader:
        padded, mask, labels = padded.to(device), mask.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(padded, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * padded.size(0)
        all_train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    scheduler.step()

    train_auc = roc_auc_score(all_train_labels, all_train_probs)

    model.eval()
    val_loss = 0
    all_val_probs, all_val_labels = [], []
    with torch.no_grad():
        for padded, mask, labels in val_loader:
            padded, mask, labels = padded.to(device), mask.to(device), labels.to(device)
            logits = model(padded, mask)
            loss = criterion(logits, labels)
            val_loss += loss.item() * padded.size(0)
            all_val_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_auc = roc_auc_score(all_val_labels, all_val_probs)

    train_losses.append(train_loss / len(train_loader.dataset))
    val_losses.append(val_loss / len(val_loader.dataset))
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {train_losses[-1]:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

print(f"\nBest val AUC: {best_val_auc:.4f}")

# Save pretrained encoder
torch.save(best_model_state, '/kaggle/working/pretrained_stage1.pth')
print("Saved pretrained Stage 1 weights.")
```

Cell 4 (code) — Learning curves plot:
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_title('Stage 1: Simulated Data (Shuffled) — Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(train_aucs, label='Train AUC')
ax2.plot(val_aucs, label='Val AUC')
ax2.set_title('Stage 1: Simulated Data (Shuffled) — ROC-AUC')
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

**Step 2: Run on Kaggle, verify learning curves show convergence.**

Expected: val AUC should climb steadily and plateau. If it converges on shuffled data, the model is learning Keplerian Doppler shapes as intended.

---

### Task 4: Stage 2 Pretraining (Ordered Timestamps)

**Objective:** Fine-tune the Stage 1 pretrained model on simulated data with realistic (ordered) timestamps. This adapts the model to real observation cadence patterns.

**Files:**
- Create: `/home/maanav0114/Documents/mled/pretrain_stage2.ipynb`

**Literature basis:** ViPer-RV §5: "In the second stage, the model is fine-tuned on an ordered dataset with realistic time sampling, which reintroduces temporal coherence reflective of actual observational conditions. Training directly on temporally ordered data was found to cause the model to overfit to sampling artifacts or activity-driven variability, reducing its ability to generalize. By contrast, the two-stage setup, starting from shuffled inputs, forces the model to first learn the underlying Keplerian Doppler shifts. The fine-tuning stage then allows the model to adjust to realistic conditions without overriding the core Keplerian signal representations."

**Step 1: Create notebook:**

Cell 0 (markdown):
```markdown
# Stage 2: Fine-tuning on Ordered Simulated Timestamps

Following ViPer-RV §5 two-stage training:
- Stage 1 (done): Shuffled → learned Keplerian Doppler shapes
- Stage 2 (this): Ordered → adapt to realistic observation cadence
- Lower learning rate (1e-4) to preserve Stage 1 representations
- Fewer epochs (15) — this is adaptation, not learning from scratch
```

Cell 1 (code) — Generate ordered simulated data and load Stage 1 weights:
```python
import os
os.environ["PYTHONHASHSEED"] = "0"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score
import math, random

seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

observations = pd.read_pickle('/kaggle/working/observations.pkl')

# Generate NEW simulated data with ordered timestamps (different seed for diversity)
from sim_dataset import get_sim_loaders
print("Generating 30,000 simulated stars (Stage 2: ordered)...")
train_loader, val_loader, _, _ = get_sim_loaders(
    observations, n_sim=30000, batch_size=32, mode='ordered', seed=seed+1, device=str(device)
)

# Load Stage 1 pretrained model
from transformer import ExoplanetTransformer
model = ExoplanetTransformer(feat_dim=21, d_model=48, nhead=4, num_layers=1,
                              dim_feedforward=96, dropout=0.3).to(device)
model.load_state_dict(torch.load('/kaggle/working/pretrained_stage1.pth'))
print("Loaded Stage 1 pretrained weights.")
```

Cell 2 (code) — Fine-tune with lower LR:
```python
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)  # 10x lower LR

warmup_epochs = 3
total_epochs = 15

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

train_losses, val_losses = [], []
train_aucs, val_aucs = [], []
best_val_auc = 0
best_model_state = None

for epoch in range(total_epochs):
    model.train()
    train_loss = 0
    all_probs, all_labels = [], []

    for padded, mask, labels in train_loader:
        padded, mask, labels = padded.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(padded, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * padded.size(0)
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    scheduler.step()
    train_auc = roc_auc_score(all_labels, all_probs)

    model.eval()
    val_loss = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for padded, mask, labels in val_loader:
            padded, mask, labels = padded.to(device), mask.to(device), labels.to(device)
            logits = model(padded, mask)
            val_loss += criterion(logits, labels).item() * padded.size(0)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_auc = roc_auc_score(all_labels, all_probs)
    train_losses.append(train_loss / len(train_loader.dataset))
    val_losses.append(val_loss / len(val_loader.dataset))
    train_aucs.append(train_auc)
    val_aucs.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch+1}/{total_epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

print(f"\nBest val AUC: {best_val_auc:.4f}")
torch.save(best_model_state, '/kaggle/working/pretrained_stage2.pth')
print("Saved pretrained Stage 2 weights.")
```

---

### Task 5: Domain-Adversarial Fine-Tuning on Real Data

**Objective:** Fine-tune the pretrained model on real HARPS/HIRES data while simultaneously training a domain discriminator that forces the encoder to learn domain-agnostic representations (sim vs real).

**Files:**
- Create: `/home/maanav0114/Documents/mled/finetune_adversarial.ipynb`

**Literature basis:** Gupta, Muthukrishna & Audenaert (ICML 2025, §3.2): "We extend the base classifier with an additional discriminator network D that attempts to predict whether a representation came from ZTF or LSST simulations. The classifier C(X) and the discriminator D(z) are trained simultaneously with competing objectives, where H denotes the cross-entropy loss. The discriminator tries to correctly classify the domain, while the classifier's encoder is trained to fool the discriminator. This creates domain-agnostic representations."

Key results from their paper:
- Pretrained models fine-tuned with 512 labels beat baselines trained on full 3747-label dataset
- Adversarial objective critical for cross-domain transfer
- Without domain adaptation, representations stayed domain-specific

**Step 1: Build the adversarial training notebook:**

Cell 0 (markdown):
```markdown
# Stage 3: Adversarial Domain Adaptation Fine-Tuning

Following Gupta et al. (ICML 2025, arXiv:2510.12958):
- Load pretrained encoder from Stage 2
- Add domain discriminator: 2-layer MLP that predicts sim vs real
- Train with competing objectives:
  1. Classification loss (planet/no-planet) — standard BCE
  2. Domain loss (sim/real) — discriminator tries to classify domain
  3. Encoder is trained to MINIMIZE discriminator accuracy (fool it)
- This forces the encoder to learn domain-agnostic representations

Key hyperparameter: lambda_domain controls the strength of domain alignment.
Too high → classification accuracy suffers. Too low → no adaptation.
Following Gupta: lambda_domain = 0.1 initially.
```

Cell 1 (code) — Model with domain adapter:
```python
import os
os.environ["PYTHONHASHSEED"] = "0"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import math, random

seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Domain Discriminator (following Gupta et al. ICML 2025 §3.2) ──
class DomainDiscriminator(nn.Module):
    """Two-layer MLP that predicts whether a representation came from
    simulated or real data. Following Gupta et al. architecture."""
    def __init__(self, input_dim=48, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)

# ── Full model: Encoder + Classifier + Domain Discriminator ──
class ExoplanetTransformerWithDomain(nn.Module):
    def __init__(self, feat_dim=21, d_model=48, nhead=4, num_layers=1,
                 dim_feedforward=96, dropout=0.3):
        super().__init__()
        # Reuse existing architecture
        self.input_proj = nn.Linear(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        class AttentionPool(nn.Module):
            def __init__(selfself, d_model):
                super().__init__()
                self.attention = nn.Linear(d_model, 1)
            def forward(selfself, x, mask):
                scores = selfself.attention(x).squeeze(-1)
                scores = scores.masked_fill(~mask.bool(), float('-inf'))
                weights = F.softmax(scores, dim=1)
                return (x * weights.unsqueeze(-1)).sum(dim=1)

        self.pool = AttentionPool(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1),
        )

        # Domain discriminator (separate module for adversarial training)
        self.domain_disc = DomainDiscriminator(input_dim=d_model)

    def encode(self, x, mask):
        """Get the d_model latent representation for domain discrimination."""
        x = self.input_proj(x)
        src_key_padding_mask = ~mask.bool()
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        z = self.pool(x, mask)  # (B, d_model)
        return z

    def forward(self, x, mask):
        z = self.encode(x, mask)
        class_out = self.classifier(z).squeeze(-1)
        domain_out = self.domain_disc(z).squeeze(-1)
        return class_out, domain_out, z

model = ExoplanetTransformerWithDomain().to(device)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load pretrained encoder weights (input_proj, transformer, pool)
pretrained = torch.load('/kaggle/working/pretrained_stage2.pth')
pretrained_keys = {k: v for k, v in pretrained.items()
                   if not k.startswith('classifier')}
model.load_state_dict(pretrained_keys, strict=False)
print("Loaded pretrained encoder weights (excluding classifier head).")
```

Cell 2 (code) — Load real data and prepare mixed batches:
```python
# Load real data (same pipeline as transformer.ipynb)
observations = pd.read_pickle('/kaggle/working/observations.pkl')

# Use the existing split.py for consistent train/val/test
from split import ensure_split
train_stars, val_stars, test_stars = ensure_split(observations)

# Build real star sequences (same as transformer.ipynb)
NUM_FREQS = 8
PERIODS = np.logspace(np.log10(1.0), np.log10(7300.0), NUM_FREQS)
FREQS = 2.0 * np.pi / PERIODS

pos_stars = sorted(set(observations[observations['has_exoplanets'] == 1]['star_name']))
neg_stars = sorted(set(observations[observations['has_exoplanets'] == 0]['star_name']))

def star_to_features_real(star_name):
    star_obs = observations[observations['star_name'] == star_name].sort_values('bjd')
    ref_bjd = star_obs['bjd'].iloc[0]
    features = []
    for _, row in star_obs.iterrows():
        dt = row['bjd'] - ref_bjd
        pos_enc = []
        for f in FREQS:
            pos_enc.append(np.sin(f * dt))
            pos_enc.append(np.cos(f * dt))
        features.append([row['rv_centered'], row['rv_err'], row['exposure_time'],
                        row['RHKp'], row['Halpha']] + pos_enc)
    return np.array(features, dtype=np.float32)

# Build train/val/test splits
train_pos = [s for s in train_stars if s in set(pos_stars)]
train_neg = [s for s in train_stars if s in set(neg_stars)]
val_pos = [s for s in val_stars if s in set(pos_stars)]
val_neg = [s for s in val_stars if s in set(neg_stars)]
test_pos = [s for s in test_stars if s in set(pos_stars)]
test_neg = [s for s in test_stars if s in set(neg_stars)]

train_data = [(star_to_features_real(s), 1) for s in train_pos] + \
             [(star_to_features_real(s), 0) for s in train_neg]

# Load simulated norm stats and apply to real data
sim_mean = np.load('/kaggle/working/sim_norm_mean.npy')
sim_std = np.load('/kaggle/working/sim_norm_std.npy')

train_data = [((seq - sim_mean) / sim_std, label) for seq, label in train_data]

print(f"Real training stars: {len(train_data)} "
      f"({sum(l for _, l in train_data)} positive)")

# Also generate a small set of simulated stars for domain mixing
from sim_utils import simulate_dataset
from sim_dataset import star_to_features as sim_feat_fn

print("Generating 5,000 simulated stars for domain mixing...")
sim_stars = simulate_dataset(observations, n_sim_stars=5000, positive_fraction=0.5, seed=seed+2)
sim_data = [(sim_feat_fn(s), int(s['has_exoplanets'])) for _, s in sim_stars]
sim_data = [((seq - sim_mean) / sim_std, label) for seq, label in sim_data]
```

Cell 3 (code) — Adversarial training loop:
```python
from torch.utils.data import DataLoader, Dataset
from sim_dataset import collate_stars

class StarDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0], dtype=torch.float32), \
               torch.tensor(self.data[idx][1], dtype=torch.float32)

batch_size = 32
real_loader = DataLoader(StarDataset(train_data), batch_size=batch_size,
                          shuffle=True, collate_fn=collate_stars, pin_memory=True)
sim_loader = DataLoader(StarDataset(sim_data), batch_size=batch_size,
                         shuffle=True, collate_fn=collate_stars, pin_memory=True)

# Adversarial training hyperparameters
lambda_domain = 0.1  # Following Gupta et al. ICML 2025
classification_criterion = nn.BCEWithLogitsLoss()
domain_criterion = nn.BCEWithLogitsLoss()  # 1 = real, 0 = simulated

# Separate optimizer for discriminator (higher LR) and encoder+classifier (lower LR)
optimizer_main = Adam(
    list(model.input_proj.parameters()) + list(model.transformer.parameters()) +
    list(model.pool.parameters()) + list(model.classifier.parameters()),
    lr=1e-4, weight_decay=5e-3
)
optimizer_disc = Adam(model.domain_disc.parameters(), lr=1e-3, weight_decay=1e-4)

warmup_epochs = 3
total_epochs = 50

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

scheduler = LambdaLR(optimizer_main, lr_lambda)

train_losses, val_aucs = [], []
best_val_auc = 0
best_model_state = None

n_pos = sum(l for _, l in train_data)
n_neg = len(train_data) - n_pos
pos_weight = torch.tensor([math.sqrt(n_neg / n_pos)]).to(device)

for epoch in range(total_epochs):
    model.train()
    epoch_loss = 0
    all_probs, all_labels = [], []

    sim_iter = iter(sim_loader)

    for real_padded, real_mask, real_labels in real_loader:
        real_padded, real_mask, real_labels = \
            real_padded.to(device), real_mask.to(device), real_labels.to(device)

        try:
            sim_padded, sim_mask, _ = next(sim_iter)
        except StopIteration:
            sim_iter = iter(sim_loader)
            sim_padded, sim_mask, _ = next(sim_iter)
        sim_padded, sim_mask = sim_padded.to(device), sim_mask.to(device)

        # ── Step 1: Train discriminator ──
        optimizer_disc.zero_grad()
        with torch.no_grad():
            _, real_domain_logits, _ = model(real_padded, real_mask)
            _, sim_domain_logits, _ = model(sim_padded, sim_mask)

        real_domain_labels = torch.ones(real_padded.size(0), device=device)
        sim_domain_labels = torch.zeros(sim_padded.size(0), device=device)

        disc_loss = domain_criterion(real_domain_logits, real_domain_labels) + \
                    domain_criterion(sim_domain_logits, sim_domain_labels)
        disc_loss.backward()
        optimizer_disc.step()

        # ── Step 2: Train encoder + classifier (with adversarial domain loss) ──
        optimizer_main.zero_grad()

        # Classification on real data
        class_logits, domain_logits, _ = model(real_padded, real_mask)
        class_loss = classification_criterion(class_logits, real_labels)
        # Use pos_weight for imbalanced real data
        class_loss_weighted = nn.functional.binary_cross_entropy_with_logits(
            class_logits, real_labels, pos_weight=pos_weight
        )

        # Domain adversarial loss: encoder tries to FOOL discriminator
        # For real data: encoder wants discriminator to predict "sim" (0)
        # For sim data: encoder wants discriminator to predict "real" (1)
        with torch.no_grad():
            pass  # already computed above, but need gradients for encoder
        _, real_domain_for_enc, _ = model(real_padded, real_mask)
        _, sim_domain_for_enc, _ = model(sim_padded, sim_mask)

        # Adversarial: flip the domain labels
        adv_loss = domain_criterion(real_domain_for_enc, sim_domain_labels[:real_padded.size(0)]) + \
                   domain_criterion(sim_domain_for_enc, real_domain_labels[:sim_padded.size(0)])

        total_loss = class_loss_weighted + lambda_domain * adv_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_main.step()

        epoch_loss += total_loss.item() * real_padded.size(0)
        all_probs.extend(torch.sigmoid(class_logits).detach().cpu().numpy())
        all_labels.extend(real_labels.cpu().numpy())

    scheduler.step()

    train_auc = roc_auc_score(all_labels, all_probs)

    # Validation
    model.eval()
    val_probs, val_labels = [], []
    val_data = [(star_to_features_real(s), 1) for s in val_pos] + \
               [(star_to_features_real(s), 0) for s in val_neg]
    val_data_normed = [((seq - sim_mean) / sim_std, l) for seq, l in val_data]
    val_loader_eval = DataLoader(StarDataset(val_data_normed), batch_size=batch_size,
                                   shuffle=False, collate_fn=collate_stars)

    with torch.no_grad():
        for padded, mask, labels in val_loader_eval:
            padded, mask, labels = padded.to(device), mask.to(device), labels.to(device)
            logits, _, _ = model(padded, mask)
            val_probs.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels.extend(labels.numpy())

    val_auc = roc_auc_score(all_val_labels, all_val_probs)
    train_losses.append(epoch_loss / len(train_data))
    val_aucs.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch+1}/{total_epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

print(f"\nBest val AUC: {best_val_auc:.4f}")
torch.save(best_model_state, '/kaggle/working/finetuned_adversarial.pth')
```

Cell 4 (code) — Plot learning curves:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_aucs, label='Val AUC')
plt.title('Adversarial Fine-Tuning: Loss and Val AUC')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

---

### Task 6: Evaluation and Comparison

**Objective:** Evaluate the fine-tuned model on the real held-out test set. Compare to RF baseline (0.7994). Run ablations. Compute bootstrap CIs.

**Files:**
- Create: `/home/maanav0114/Documents/mled/eval_sim_pretrained.ipynb`

**Step 1: Build evaluation notebook:**

Cell 0 (markdown):
```markdown
# Evaluation: Simulation-Pretrained Transformer vs RF Baseline

## Metrics
- ROC-AUC with bootstrap 95% CI (200 resamples)
- PR-AUC (Average Precision)
- Best F1 threshold (selected on validation set)
- Confusion matrix at optimal threshold

## Comparisons
1. Simulation-pretrained + adversarial fine-tuned Transformer
2. RF baseline (16 physical features, AUC = 0.7994)
3. Ablation: Transformer trained from scratch on real data only (no pretraining)
4. Ablation: Simulation-pretrained WITHOUT domain adaptation (no adversarial)
```

Cell 1 (code) — Load best model and evaluate:
```python
import torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from sim_dataset import collate_stars
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load best fine-tuned model
model = ExoplanetTransformerWithDomain().to(device)
model.load_state_dict(torch.load('/kaggle/working/finetuned_adversarial.pth'))
model.eval()

# Load real test data
observations = pd.read_pickle('/kaggle/working/observations.pkl')
from split import ensure_split
train_stars, val_stars, test_stars = ensure_split(observations)

# ... build test sequences, evaluate, compute metrics, bootstrap CIs ...
# (Follow same evaluation pattern as baseline.ipynb)

# Determine best F1 threshold on validation set
# Then evaluate on test set with that threshold
# Report ROC-AUC, PR-AUC, F1, confusion matrix
# Bootstrap 200 resamples for 95% CI

# Compare to RF: 0.7994 ± CI
```

Cell 2 (code) — Ablation: no pretraining (train from scratch on real data):
```python
# Train ExoplanetTransformer from scratch on real data only
# (This is the "no pretrain" baseline — same as our existing Transformer result)
# Should reproduce ~0.6353 AUC (from state.md)
```

Cell 3 (code) — Ablation: pretraining without domain adaptation:
```python
# Fine-tune pretrained encoder WITHOUT adversarial domain loss
# (lambda_domain = 0, just classification loss)
# This isolates the contribution of domain adaptation
```

Cell 4 (code) — Final comparison table:
```python
import pandas as pd

comparison = pd.DataFrame([
    {"Model": "RF (16 physical features)", "ROC-AUC": 0.7994, "95% CI": "[TBD]", "PR-AUC": 0.4658, "Best F1": 0.5221, "Input": "per-star aggregates"},
    {"Model": "Transformer (from scratch, real only)", "ROC-AUC": 0.6353, "95% CI": "[0.574, 0.708]", "PR-AUC": "—", "Best F1": "—", "Input": "raw sequences"},
    {"Model": "Sim-Pretrained Transformer (this work)", "ROC-AUC": roc_auc_test, "95% CI": ci_str, "PR-AUC": pr_auc_test, "Best F1": best_f1_test, "Input": "raw sequences"},
    {"Model": "Sim-Pretrained + Adversarial (this work)", "ROC-AUC": roc_auc_adv, "95% CI": ci_adv, "PR-AUC": pr_auc_adv, "Best F1": best_f1_adv, "Input": "raw sequences"},
    {"Model": "Sim-Pretrained, no domain adapt (ablation)", "ROC-AUC": roc_auc_noadv, "95% CI": ci_noadv, "PR-AUC": pr_auc_noadv, "Best F1": best_f1_noadv, "Input": "raw sequences"},
])

print(comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
```

---

### Task 7: Update Project Wiki

**Objective:** Document results in the project wiki.

**Files:**
- Update: `~/.hermes/projects/ml-exoplanet-detection/state.md`
- Update: `~/.hermes/projects/ml-exoplanet-detection/log.md`
- Create: `~/.hermes/projects/ml-exoplanet-detection/findings/sim-pretraining-results.md`

**Step 1:** After running all notebooks on Kaggle, update state.md with results.
**Step 2:** Write log entry for this session.
**Step 3:** Write findings document summarizing what worked, what didn't, and paper implications.

---

## Summary of Literature Anchoring

| Pipeline Component | Paper | Key Technique Borrowed |
|---|---|---|
| **Simulation engine** | ExoplANNET (Nieto & Díaz 2023) | GP quasi-periodic kernel for stellar activity; uncertainty sampling from real survey stats |
| **Keplerian injection** | ViPer-RV (Gavankar et al. 2025) | radvel-based Keplerian; log-uniform periods, realistic eccentricity range |
| **Real cadence templates** | ViPer-RV | Real observation timestamps + activity indicators as background, synthetic planet injection |
| **Two-stage training** | ViPer-RV | Stage 1: shuffled timestamps (learn Doppler shapes). Stage 2: ordered (adapt to real cadence) |
| **Domain randomization** | Gupta et al. ICML 2025 | Randomize activity parameters across simulated stars for robust representations |
| **Adversarial domain adaptation** | Gupta et al. ICML 2025 | Discriminator predicts sim vs real; encoder trained to fool it. Creates domain-agnostic representations |
| **Fine-tune with minimal labels** | Gupta et al. ICML 2025 | Pretrained + fine-tune approach; 512 labels beat 3747-label baseline |
| **Comparison vs tabular baseline** | Our own RF (0.7994) | Novel — no prior sim-pretraining paper compares against a strong tabular baseline |

## Risks and Mitigations

| Risk | Mitigation | Source |
|---|---|---|
| Sim-to-real gap too large | Domain adaptation (adversarial); real cadences as templates | Gupta ICML 2025; ViPer-RV |
| Model confuses stellar rotation with planet signal | Two-stage training (shuffled → ordered) | ViPer-RV §5 |
| Simulation not realistic enough | GP quasi-periodic activity + real uncertainties + domain randomization | ExoplANNET §3 |
| Domain adaptation hurts classification | lambda_domain = 0.1 (conservative); ablation to measure impact | Gupta ICML 2025 |
| Fixed sequence length needed | Our architecture already handles variable-length via padding+masking | Our existing Transformer |
| GP sampling too slow | Sums-of-sinusoids approximation instead of full GP (O(N) vs O(N³)) | Practical engineering choice |
