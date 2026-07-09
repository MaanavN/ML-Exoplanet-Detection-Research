"""RV time series simulation utilities for simulation-based pretraining.

Literature basis:
- Stellar activity (GP quasi-periodic kernel): ExoplANNET (Nieto & Diaz 2023, A&A 677)
- Keplerian injection parameters: ViPer-RV (Gavankar et al. 2025, AJ)
- Uncertainty sampling from real survey stats: ExoplANNET §3.2
"""

import numpy as np
import pandas as pd


# ── Timestamp positional encoding (must match transformer.ipynb / sim_dataset.py) ──
NUM_FREQS = 8
MIN_PERIOD = 1.0
MAX_PERIOD = 7300.0
PERIODS = np.logspace(np.log10(MIN_PERIOD), np.log10(MAX_PERIOD), NUM_FREQS)
FREQS = 2.0 * np.pi / PERIODS


def bjd_positional_encoding(bjd, ref_bjd):
    """Sinusoidal positional encoding for BJD timestamps.

    Produces 16 dims (8 frequencies x [sin, cos]). Matches the encoding used
    in transformer.ipynb Cell 1 and sim_dataset.star_to_features().
    """
    dt = bjd - ref_bjd
    encoding = []
    for f in FREQS:
        encoding.append(np.sin(f * dt))
        encoding.append(np.cos(f * dt))
    return encoding


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
                  activity_amp=2.0, jitter_amp=1.0,
                  exposure_time_real=None):
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
        exposure_time_real: real per-observation exposure times (seconds).
            If None, falls back to 1800.0 constant (NOT recommended — causes
            zero-variance feature after normalization).

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

    # 5. Couple activity indicators to the GP activity realization.
    # rhkp and Halpha are chromospheric activity tracers — they should correlate
    # with the activity-induced RV, not just be the real template values scaled ±20%.
    # We generate them from the same activity_rv signal so the model can learn
    # the RV↔indicator correlation (which is what a real RV model should exploit).
    #
    # Physical basis: more active stars → higher S-index and H-alpha → larger RV jitter.
    # We model: rhkp_sim = rhkp_baseline + activity_rv * conversion_factor
    # where conversion_factor maps RV activity amplitude to indicator variation.
    rhkp_baseline = np.mean(rhkp_real)
    halpha_baseline = np.mean(halpha_real)
    # Conversion factors: how much the activity RV (m/s) shifts the indicators.
    # These are approximate but give the right order of magnitude of correlation.
    rhkp_conv = 0.001 * np.max(np.abs(activity_rv)) / max(np.std(rhkp_real), 1e-6)
    halpha_conv = 0.001 * np.max(np.abs(activity_rv)) / max(np.std(halpha_real), 1e-6)

    rhkp_sim = rhkp_baseline + activity_rv * rhkp_conv + np.random.normal(0, np.std(rhkp_real) * 0.1, n_obs)
    halpha_sim = halpha_baseline + activity_rv * halpha_conv + np.random.normal(0, np.std(halpha_real) * 0.1, n_obs)

    return {
        'bjd': t,
        'rv_centered': rv_centered,
        'rv_err': np.array(rv_err_real),
        'exposure_time': np.array(exposure_time_real, dtype=float) if exposure_time_real is not None
                          else np.full(n_obs, 1800.0, dtype=float),
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
        exposure_time = star_obs['exposure_time'].values if 'exposure_time' in star_obs.columns else None

        has_planet = i < n_positive

        # Sample activity parameters from ranges (domain randomization)
        activity_args = {
            'activity_rotation': activity_params_range['rotation_period'],
            'activity_evolution': activity_params_range['evolution_timescale'],
            'activity_amp': activity_params_range['amplitude'],
            'jitter_amp': activity_params_range['jitter_amp'],
        }

        sim_star = simulate_star(
            t_bjd=t_bjd, rv_err_real=rv_err,
            rhkp_real=rhkp, halpha_real=halpha,
            has_planet=has_planet,
            exposure_time_real=exposure_time,
            **activity_args
        )
        sim_stars.append((template_star, sim_star))

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{n_sim_stars} simulated stars")

    return sim_stars
