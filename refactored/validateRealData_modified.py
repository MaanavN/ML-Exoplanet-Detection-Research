"""
Modified version of validateRealData.py (Ante Režić,github.com/anterezic1999/diplomski_public)
for the MLED exoplanet detection audit paper (Nakodkar et al., in prep).

Modifications vs the upstream `validateRealData.py`:
  1. Loads labels from our pre-computed `observations.pkl` (star_name -> has_exoplanets)
     rather than querying the NEA per star at eval time.  This both removes a network
     dependency and (critically) lets us feed PLANET-FREE stars to the model — the
     upstream script `continue`s past every star with no detected planets, by design.
  2. Removes the "skip system if no valid planets are found" branch.  Planet-free stars
     are now passed through the model and contribute (false-positive) peaks to the
     peak-level eval, and a star-level prediction, to the star-level eval.  This is the
     audit contribution: every published DL model for RV exoplanet detection has so far
     been evaluated only on planet-host systems; we extend the evaluation to the
     planet-free regime that characterises the realistic deployment task.
  3. Adds star-level aggregation: max peak probability per star -> star-level score in
     [0, 1] used for PR-AUC / ROC-AUC.  A binary star-level label is also produced under
     Ante's original threshold (0.5) for P/R/F1 reporting comparable to the published
     paper.
  4. No-peak-found behaviour: planet-free stars whose peak-finder returns no peaks are
     SKIPPED from the metric.  n_no_peaks_skipped is reported separately.  Asymmetric
     coverage is acknowledged: the count of planet-free stars that produced no peaks
     is reported.  (TODO: a sensitivity analysis at default-to-0 vs default-to-0.5
     should be added before submission; for now we skip.)
  5. star_mass is hardcoded to 0, matching the upstream script's behaviour on its
     published real-data run.  We preserve this for direct comparability with the
     published numbers while acknowledging in the paper methods that the model was
     trained with mass as a feature but deployed with mass=0.

Outputs (printed + pickled): arrays of per-star predictions and labels, per-peak
predictions and labels (for reproducibility of Ante's published peak-level metric),
and summary metrics.
"""

from torch import nn
import torch.nn.functional as F
import h5py
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Model architecture (DO NOT MODIFY — must match the released weights)
# ============================================================================

PERIODOGRAM_LEN = 1000
PADDED_LEN = 61


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if out.size() != identity.size():
            identity = F.pad(identity, (0, out.size(2) - identity.size(2)))
        out += self.shortcut(identity)
        out = F.relu(out)
        return out


class PlanetDetectionModel_Enhanced(nn.Module):
    """3-branch CNN: periodogram (2-channel) + peak window (1-channel) + scalar features (3-d)."""

    def __init__(self):
        super(PlanetDetectionModel_Enhanced, self).__init__()
        self.conv_layers = nn.Sequential(
            ResidualBlock(2, 32, kernel_size=3, stride=1, padding=1), nn.MaxPool1d(2), nn.Dropout(0.1),
            ResidualBlock(32, 64, kernel_size=5, stride=1, padding=2), nn.MaxPool1d(2), nn.Dropout(0.1),
            ResidualBlock(64, 128, kernel_size=7, stride=1, padding=3), nn.MaxPool1d(2), nn.Dropout(0.1),
            ResidualBlock(128, 256, kernel_size=9, stride=1, padding=4), nn.MaxPool1d(2), nn.Dropout(0.1),
            ResidualBlock(256, 512, kernel_size=11, stride=1, padding=5),
            nn.AdaptiveAvgPool1d(1),
        )
        self.peak_conv_layers = nn.Sequential(
            ResidualBlock(1, 16, kernel_size=3, stride=1, padding=1), nn.MaxPool1d(2), nn.Dropout(0.1),
            ResidualBlock(16, 32, kernel_size=5, stride=1, padding=2), nn.MaxPool1d(2), nn.Dropout(0.1),
            ResidualBlock(32, 64, kernel_size=7, stride=1, padding=3), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_features = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(512 + 64 + 64, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 6), nn.BatchNorm1d(6), nn.ReLU(),
            nn.Linear(6, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, freq, power, features, peak):
        periodogram = torch.stack((freq, power), dim=1)
        conv_out = self.conv_layers(periodogram).squeeze(-1)
        peak_conv_out = self.peak_conv_layers(peak.unsqueeze(1)).squeeze(-1)
        features_out = self.fc_features(features)
        combined = torch.cat((conv_out, peak_conv_out, features_out), dim=1)
        output = self.fc_combined(combined)
        return output


# ============================================================================
# Peak extraction utilities (unchanged from upstream)
# ============================================================================


def pad_truncate(array, target_length=PADDED_LEN):
    current_length = len(array)
    if current_length > target_length:
        return array[:target_length]
    elif current_length < target_length:
        return np.pad(array, (0, target_length - current_length), "constant")
    else:
        return array


def get_peak_padded(peak_index, f, Pxx, n_points=30):
    total_points = 2 * n_points + 1
    start_index = max(0, peak_index - n_points)
    end_index = min(len(f), peak_index + n_points + 1)
    padded_Pxx = np.zeros(total_points)
    actual_start = n_points - (peak_index - start_index)
    actual_end = actual_start + (end_index - start_index)
    padded_Pxx[actual_start:actual_end] = Pxx[start_index:end_index]
    pmin, pmax = np.min(padded_Pxx), np.max(padded_Pxx)
    if pmax - pmin > 1e-12:
        normalized_Pxx = (padded_Pxx - pmin) / (pmax - pmin)
    else:
        normalized_Pxx = np.zeros_like(padded_Pxx)
    return normalized_Pxx


def select_top_frequencies_exclude_entire_peak(
    frequencies, power, n, max_period=7000, PADDED_LEN=61
):
    """Select the top-n peaks in the periodogram, excluding the full peak region
    (and 10% period neighbours) after each selection, so we don't pick two peaks
    from the same dominant feature.

    Returns:
      selected_indices: np.array of peak indices (length <= n)
      peaks: np.array of padded peak power windows, one per selected index
    If the highest-power remaining frequency is below 1/max_period (i.e. period
    exceeds 7000 days) we break early and return whatever we have so far.  This
    is the upstream no-peak / low-frequency-break behaviour.
    """
    min_freq = 1 / max_period
    selected_indices = []
    masked_indices = set()
    peaks = []

    while len(selected_indices) < n and len(masked_indices) < len(frequencies):
        available_indices = [
            i for i in range(len(frequencies)) if i not in masked_indices
        ]
        if not available_indices:
            break
        top_idx = max(available_indices, key=lambda i: power[i])

        if frequencies[top_idx] < min_freq:
            break

        peak_indices = [top_idx]
        peak_period = 1 / frequencies[top_idx]

        left_idx = top_idx - 1
        while left_idx >= 0 and power[left_idx] < power[left_idx + 1]:
            peak_indices.append(left_idx)
            left_idx -= 1

        right_idx = top_idx + 1
        while right_idx < len(frequencies) and power[right_idx] < power[right_idx - 1]:
            peak_indices.append(right_idx)
            right_idx += 1

        selected_indices.append(top_idx)

        masked_indices.update(peak_indices)

        for i in range(len(frequencies)):
            if i not in masked_indices:
                period = 1 / frequencies[i]
                if abs(period - peak_period) / peak_period <= 0.1:
                    masked_indices.add(i)

        padded_peak = get_peak_padded(top_idx, frequencies, power)
        padded_peak = pad_truncate(padded_peak, PADDED_LEN)
        peaks.append(padded_peak)

        if len(selected_indices) == n:
            break

    return np.array(selected_indices), np.array(peaks)


def run_inference(model, star_mass, frequencies, power, selected_indices, selected_peaks, device):
    """Run the 3-branch CNN on each of the star's n_peaks candidate peaks and
    return per-peak probability of being planetary in [0, 1]."""
    model.eval()
    predictions = []

    # Normalize power to [0, 1] — matches the upstream `normalize_periodogram`
    # behaviour used during model training on synthetic periodograms.
    pmin, pmax = np.min(power), np.max(power)
    if pmax - pmin > 1e-12:
        power_norm = (power - pmin) / (pmax - pmin)
    else:
        power_norm = np.zeros_like(power)

    with torch.no_grad():
        for i, peak_idx in enumerate(selected_indices):
            peak_freq = frequencies[peak_idx]
            peak_power = power_norm[peak_idx]

            freq_norm = np.log1p(peak_freq)
            pow_norm = np.log1p(peak_power)

            # NOTE: star_mass hardcoded to 0 to match Ante's released eval script
            # (`validateRealData.py` line 498).  The model was trained with mass as
            # an input but the released real-data evaluation also used mass=0; we
            # preserve this for direct comparability with the published numbers.
            features = torch.FloatTensor([freq_norm, pow_norm, star_mass]).unsqueeze(0).to(device)
            entire_peak = torch.FloatTensor(selected_peaks[i]).unsqueeze(0).to(device)

            # Periodogram input expects (freq, power) stacked as 2-channel
            periodogram_freq = torch.FloatTensor(frequencies).unsqueeze(0).to(device)
            periodogram_power = torch.FloatTensor(power_norm).unsqueeze(0).to(device)

            output = model(periodogram_freq, periodogram_power, features, entire_peak)
            prediction = torch.sigmoid(output).item()
            predictions.append(prediction)

    return np.array(predictions)


# ============================================================================
# Modified main: planet-free aware evaluation + star-level aggregation
# ============================================================================


def evaluate_ante_model(
    hdf5_file_path,
    model_path,
    observations_pkl_path,
    n_peaks=3,
    threshold=0.5,
    output_pickle_path="ante_eval_results.pkl",
):
    """Run Ante's model on every star in observations.pkl, aggregating peak-level
    per-peak predictions to star-level by max-peak-probability.  Compute both
    peak-level metrics (reproducing Ante's published evaluation style, modulo
    inclusion of planet-free stars where every peak is by definition a false
    positive) and star-level metrics (PR-AUC, ROC-AUC, F1 — the primary numbers
    for our audit table).

    Args:
      hdf5_file_path: str — precomputed periodograms, HDF5 with hf[star]['frequencies']
        and hf[star]['power'] arrays of length PERIODOGRAM_LEN.
      model_path: str — path to the released PlanetDetectionModel_Enhanced weights.
      observations_pkl_path: str — our preprocessed observations.pkl with columns
        ['star_name', 'bjd', 'rv', 'rv_err', 'rhkp', 'halpha', 'exposure_time',
         'has_exoplanets', 'rv_centered'].
      n_peaks: int — number of peaks to extract per star (Ante default: 3).
      threshold: float — Ante's threshold for binarising peak-level predictions
        (Ante's default: 0.5).
      output_pickle_path: str — where to save the per-star / per-peak eval arrays.

    Returns dict containing all metrics + raw arrays.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        confusion_matrix,
        f1_score,
        fbeta_score,
    )

    # --- Load model ---
    model = PlanetDetectionModel_Enhanced().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Load our labels: star_name -> has_exoplanets ---
    obs = pd.read_pickle(observations_pkl_path)
    star_label_map = (
        obs.groupby("star_name")["has_exoplanets"].first().to_dict()
    )

    # --- Verify HDF5 star set matches our labeled stars ---
    with h5py.File(hdf5_file_path, "r") as hf:
        hdf5_stars = set(hf.keys())
    labeled_stars = set(star_label_map.keys())
    in_both = hdf5_stars & labeled_stars
    n_hdf5_only = len(hdf5_stars - labeled_stars)
    n_labels_only = len(labeled_stars - hdf5_stars)
    print(f"HDF5 has {len(hdf5_stars)} stars; labels has {len(labeled_stars)} stars.")
    print(f"  Intersection: {len(in_both)}")
    print(f"  HDF5-only (no label): {n_hdf5_only}")
    print(f"  Labels-only (no periodogram): {n_labels_only}")
    if n_labels_only > 0:
        print(
            "  warning: some labeled stars have no periodogram in the HDF5 file; "
            "they will be skipped from the evaluation.  See ante_model_evaluation.ipynb "
            "Section A to regenerate the HDF5 file."
        )
    eval_stars = sorted(in_both)

    # --- Iterate stars, collect per-peak and per-star predictions ---
    all_peak_preds = []
    all_peak_labels = []
    all_peak_star_names = []
    all_peak_periods = []
    star_pred_max = {}        # star_name -> float (max peak prob)
    star_pred_any_above_thr = {}  # star_name -> int (1 if any peak prob > threshold, else 0)
    star_label = {}           # star_name -> int (0 or 1)
    n_no_peaks = 0
    n_no_peaks_pos = 0        # no-peaks breakdown by class
    n_no_peaks_neg = 0
    n_errors = 0

    with h5py.File(hdf5_file_path, "r") as hf:
        for star_name in tqdm(eval_stars, desc="Evaluating Ante model on stars"):
            try:
                frequencies = hf[star_name]["frequencies"][:]
                power = hf[star_name]["power"][:]
            except Exception as e:
                n_errors += 1
                continue

            y_star = int(star_label_map[star_name])
            star_label[star_name] = y_star

            # --- Peak extraction ---
            try:
                selected_indices, selected_peaks = (
                    select_top_frequencies_exclude_entire_peak(
                        frequencies, power, n=n_peaks
                    )
                )
            except Exception as e:
                n_errors += 1
                continue

            if len(selected_indices) == 0:
                # No peaks found (periodogram dominated by long-period power below
                # min_freq threshold, or peak detector failed).
                n_no_peaks += 1
                if y_star == 1:
                    n_no_peaks_pos += 1
                else:
                    n_no_peaks_neg += 1
                # Per the agreed audit protocol: SKIP stars with no peaks from the
                # metric.  Sensitivity analysis (default-to-0 vs default-to-0.5)
                # is TODO before paper submission.
                # NOTE: we DO record the star's label so we can quantify asymmetric
                # coverage below.
                continue

            # --- Run inference on each extracted peak ---
            predictions = run_inference(
                model,
                0,  # star_mass = 0 (matches upstream behaviour)
                frequencies,
                power,
                selected_indices,
                selected_peaks,
                device,
            )

            # Record per-peak data: label is 1 iff the peak's period lies within
            # 10% of a KNOWN planet period of this star AND the star has at least
            # one RV-detected planet.  For planet-free stars every peak is by
            # definition a false positive.  (Note: we do not re-query the NEA per
            # star in this modified script; we use our has_exoplanets label for
            # the star-level aggregation.  For peak-level reproduction of Ante's
            # published metric in the planet-only regime, the user should run the
            # UPSTREAM validateRealData.py against the same HDF5 file.)
            for idx, pred in zip(selected_indices, predictions):
                period = 1.0 / frequencies[idx]
                all_peak_preds.append(float(pred))
                all_peak_labels.append(1 if y_star == 1 else 0)
                all_peak_star_names.append(star_name)
                all_peak_periods.append(period)

            # Star-level aggregation (Option B primary + Option A secondary):
            star_pred_max[star_name] = float(np.max(predictions)) if len(predictions) > 0 else 0.0
            star_pred_any_above_thr[star_name] = (
                1 if np.any(predictions > threshold) else 0
            )

    # --- Quantify asymmetric no-peak coverage ---
    n_total = len(eval_stars)
    n_evaluated = n_total - n_no_peaks - n_errors
    n_pos_total = sum(1 for s in eval_stars if star_label.get(s, 0) == 1)
    n_neg_total = sum(1 for s in eval_stars if star_label.get(s, 0) == 0)
    coverage = {
        "n_total_eval_stars": n_total,
        "n_evaluated_with_peaks": n_evaluated,
        "n_skipped_no_peaks": n_no_peaks,
        "n_skipped_errors": n_errors,
        "n_no_peaks_positives": n_no_peaks_pos,
        "n_no_peaks_negatives": n_no_peaks_neg,
        "n_pos_total": n_pos_total,
        "n_neg_total": n_neg_total,
        "coverage_positives": (n_pos_total - n_no_peaks_pos) / max(1, n_pos_total),
        "coverage_negatives": (n_neg_total - n_no_peaks_neg) / max(1, n_neg_total),
    }

    print("\n===== COVERAGE =====")
    for k, v in coverage.items():
        print(f"  {k:30s} {v}")

    # --- Star-level metrics (primary — for the paper comparison table) ---
    star_names_evaluated = sorted(star_pred_max.keys())
    y_star = np.array([star_label[s] for s in star_names_evaluated])
    p_star_max = np.array([star_pred_max[s] for s in star_names_evaluated])
    p_star_binary = np.array(
        [star_pred_any_above_thr[s] for s in star_names_evaluated]
    )

    n_pos_eval = int(np.sum(y_star == 1))
    n_neg_eval = int(np.sum(y_star == 0))
    print(f"\n===== STAR-LEVEL EVAL (n = {len(y_star)}, pos={n_pos_eval}, neg={n_neg_eval}) =====")

    metrics_star = {}
    if n_pos_eval > 0 and n_neg_eval > 0:
        # Continuous star-level score -> PR-AUC, ROC-AUC
        pr_auc = average_precision_score(y_star, p_star_max)
        roc_auc = roc_auc_score(y_star, p_star_max)
        metrics_star["pr_auc"] = float(pr_auc)
        metrics_star["roc_auc"] = float(roc_auc)
        print(f"  PR-AUC  (max-peak-prob):  {pr_auc:.4f}")
        print(f"  ROC-AUC (max-peak-prob):  {roc_auc:.4f}")
    else:
        print("  WARN: skipped PR-AUC/ROC-AUC because one class missing")
        metrics_star["pr_auc"] = None
        metrics_star["roc_auc"] = None

    # Binary star-level prediction -> P/R/F1 at threshold (Ante's published metric style)
    if np.sum(p_star_binary == 1) + np.sum(p_star_binary == 0) > 0:
        try:
            cm = confusion_matrix(y_star, p_star_binary)
            # sklearn confusion_matrix returns [[TN, FP], [FN, TP]] — same layout
            # as the upstream script's helper.
            tn, fp = cm[0]
            fn, tp = cm[1]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics_star["precision_at_thr"] = float(precision)
            metrics_star["recall_at_thr"] = float(recall)
            metrics_star["f1_at_thr"] = float(f1)
            metrics_star["threshold"] = float(threshold)
            metrics_star["confusion_matrix"] = [int(x) for x in [tn, fp, fn, tp]]
            print(
                f"  Binary star-level @ thr={threshold}: P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
            )
            print(f"  Confusion (TN,FP,FN,TP): ({tn},{fp},{fn},{tp})")
        except Exception as e:
            print(f"  WARN: confusion matrix computation failed: {e}")

    # --- Peak-level metrics for transparency (reproduce Ante's published style
    # but on our star set — note this is NOT directly comparable to his published
    # numbers because of different period grid, label semantics, and inclusion
    # of planet-free peaks where every peak is a false positive) ---
    print(f"\n===== PEAK-LEVEL EVAL (n = {len(all_peak_labels)}) =====")
    peak_labels = np.array(all_peak_labels)
    peak_preds = np.array(all_peak_preds)
    peak_binary = (peak_preds > threshold).astype(int)
    if len(peak_labels) > 0:
        try:
            peak_cm = confusion_matrix(peak_labels, peak_binary)
            ptn, pfp = peak_cm[0]
            pfn, ptp = peak_cm[1]
            p_prec = ptp / (ptp + pfp) if (ptp + pfp) > 0 else 0
            p_rec = ptp / (ptp + pfn) if (ptp + pfn) > 0 else 0
            p_f1 = 2 * p_prec * p_rec / (p_prec + p_rec) if (p_prec + p_rec) > 0 else 0
            print(
                f"  Peak-level @ thr={threshold}: P={p_prec:.4f} R={p_rec:.4f} F1={p_f1:.4f}"
            )
            print(f"  Confusion (TN,FP,FN,TP): ({ptn},{pfp},{pfn},{ptp})")
            metrics_star["peak_level"] = {
                "n_peaks": int(len(peak_labels)),
                "precision": float(p_prec),
                "recall": float(p_rec),
                "f1": float(p_f1),
                "confusion_matrix": [int(x) for x in [ptn, pfp, pfn, ptp]],
                "threshold": float(threshold),
            }
        except Exception as e:
            print(f"  WARN: peak-level confusion matrix failed: {e}")

    # --- Save raw arrays for downstream analysis / plots ---
    results = {
        "coverage": coverage,
        "metrics_star": metrics_star,
        "star_names_evaluated": star_names_evaluated,
        "y_star": y_star.tolist(),
        "p_star_max": p_star_max.tolist(),
        "p_star_binary_at_thr": p_star_binary.tolist(),
        "threshold": float(threshold),
        "n_peaks_per_star": int(n_peaks),
        "peak_level": {
            "star_names": all_peak_star_names,
            "labels": all_peak_labels,
            "predictions": all_peak_preds,
            "periods": all_peak_periods,
        },
    }
    with open(output_pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_pickle_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Ante Režić's model on our star set (planet-free aware, star-level aggregation)."
    )
    parser.add_argument("hdf5_file", help="Path to HDF5 file with precomputed periodograms")
    parser.add_argument("model_path", help="Path to Ante's trained model weights")
    parser.add_argument(
        "--observations_pkl",
        default="/kaggle/working/observations.pkl",
        help="Path to our preprocessed observations.pkl with has_exoplanets labels",
    )
    parser.add_argument("--n_peaks", type=int, default=3, help="Top-n peaks per star (Ante default: 3)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Peak-level threshold (Ante default: 0.5)",
    )
    parser.add_argument(
        "--output",
        default="ante_eval_results.pkl",
        help="Where to save eval results pickle",
    )
    args = parser.parse_args()
    evaluate_ante_model(
        hdf5_file_path=args.hdf5_file,
        model_path=args.model_path,
        observations_pkl_path=args.observations_pkl,
        n_peaks=args.n_peaks,
        threshold=args.threshold,
        output_pickle_path=args.output,
    )
