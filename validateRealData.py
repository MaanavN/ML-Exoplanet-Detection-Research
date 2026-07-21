from torch import nn
import argparse
import torch.nn.functional as F
import h5py
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import difflib
import json
import urllib.parse

# Constants
PERIODOGRAM_LEN = 1000
PADDED_LEN = 61
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                nn.BatchNorm1d(out_channels)
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

class AttentionLayer(nn.Module):
    def __init__(self, channels):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class PlanetDetectionModel_Enhanced(nn.Module):
    def __init__(self):
        super(PlanetDetectionModel_Enhanced, self).__init__()

        self.conv_layers = nn.Sequential(
            ResidualBlock(2, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(32),
            ResidualBlock(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(64),
            ResidualBlock(64, 128, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(128),
            ResidualBlock(128, 256, kernel_size=9, stride=1, padding=4),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(256),
            ResidualBlock(256, 512, kernel_size=11, stride=1, padding=5),
            #AttentionLayer(512),
            nn.AdaptiveAvgPool1d(1)
        )

        self.peak_conv_layers = nn.Sequential(
            ResidualBlock(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(16),
            ResidualBlock(16, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(32),
            ResidualBlock(32, 64, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(64),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_features = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(512 + 64 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 6),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

def get_system_aliases(star_name):
    base_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/nph-aliaslookup.py?objname="
    encoded_name = urllib.parse.quote(star_name)
    url = base_url + encoded_name

    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = json.loads(response.text)
            return data
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON response")
            return None
    else:
        print(f"Error: {response.status_code}")
        return None

def pad_truncate(array, target_length=PADDED_LEN):
    current_length = len(array)
    if current_length > target_length:
        return array[:target_length]
    elif current_length < target_length:
        return np.pad(array, (0, target_length - current_length), 'constant')
    else:
        return array

def select_top_frequencies_exclude_entire_peak(frequencies, power, n, max_period=7000, PADDED_LEN=61):
    min_freq = 1 / max_period
    selected_indices = []
    selected_frequencies = []
    masked_indices = set()
    peaks = []

    while len(selected_indices) < n and len(masked_indices) < len(frequencies):
        available_indices = [i for i in range(len(frequencies)) if i not in masked_indices]
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
        selected_frequencies.append(frequencies[top_idx])

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

def get_peak_padded(peak_index, f, Pxx, n_points=30):
    total_points = 2 * n_points + 1  # Total points including the center

    # Calculate start and end indices
    start_index = max(0, peak_index - n_points)
    end_index = min(len(f), peak_index + n_points + 1)

    # Initialize array with zeros
    padded_Pxx = np.zeros(total_points)

    # Fill in the actual values
    actual_start = n_points - (peak_index - start_index)
    actual_end = actual_start + (end_index - start_index)

    padded_Pxx[actual_start:actual_end] = Pxx[start_index:end_index]

    # Normalize padded_Pxx
    normalized_Pxx = (padded_Pxx - np.min(padded_Pxx)) / (np.max(padded_Pxx) - np.min(padded_Pxx))

    return normalized_Pxx

def construct_periodogram(power, frequencies):
    # Ensure power and frequencies are the same length
    assert len(power) == len(frequencies) == PERIODOGRAM_LEN, "Power and frequencies must both have length PERIODOGRAM_LEN"

    # Create the new array
    periodogram = np.zeros(PERIODOGRAM_LEN*2)

    # Fill the first PERIODOGRAM_LEN elements with power values
    periodogram[:PERIODOGRAM_LEN] = frequencies

    # Fill the next PERIODOGRAM_LEN elements with frequency values
    periodogram[PERIODOGRAM_LEN:PERIODOGRAM_LEN*2] = power

    return periodogram

def normalize_periodogram(data):
    # Extract frequency and power
    freq = data[:PERIODOGRAM_LEN]
    power = data[PERIODOGRAM_LEN:PERIODOGRAM_LEN*2]

    # Apply log1p transformation
    freq_log = np.log1p(freq)
    power_log = np.log1p(power)

    # # Normalize using the provided mean and std values
    # freq_mean, freq_std = 3.2810406108236996, 4.036496125329278
    # power_mean, power_std = 3.5355131461859264, 3.3413114910405226
    #
    # freq_normalized = (freq_log - freq_mean) / freq_std
    # power_normalized = (power_log - power_mean) / power_std


    # Combine normalized data
    normalized_data = np.concatenate([
        freq_log,
        power_log
    ])

    return normalized_data

def run_inference(model, star_mass, frequencies, power, selected_indices, selected_peaks, device):
    model.eval()
    predictions = []

    # Prepare the full periodogram
    power = power - np.min(power)
    power = power / np.max(power)
    periodogram = np.column_stack((frequencies, power))
    periodogram_tensor = torch.FloatTensor(periodogram).unsqueeze(0).to(device)  # Shape: [1, 1000, 2]

    with torch.no_grad():
        for i, peak_idx in enumerate(selected_indices):
            peak_freq = frequencies[peak_idx]
            peak_power = power[peak_idx]

            # Normalize frequency and power
            freq_norm = np.log1p(peak_freq)
            pow_norm = np.log1p(peak_power)

            # Prepare the feature tensor for this specific frequency/power pair
            features = torch.FloatTensor([freq_norm, pow_norm, star_mass]).unsqueeze(0).to(device)

            # Get the corresponding peak from selected_peaks and reshape
            entire_peak = torch.FloatTensor(selected_peaks[i]).unsqueeze(0).to(device)  # Shape: [1, 61]

            output = model(periodogram_tensor[:, :, 0], periodogram_tensor[:, :, 1], features, entire_peak)
            prediction = torch.sigmoid(output).item()
            predictions.append(prediction)

    return np.array(predictions)

def plot_periodogram(frequencies, power, star_name, selected_indices, predictions, margin):
    plt.figure(figsize=(12, 6))
    plt.plot(1/frequencies, power)

    # Separate indices based on prediction values
    low_indices = [idx for idx, pred in zip(selected_indices, predictions) if pred < margin]
    high_indices = [idx for idx, pred in zip(selected_indices, predictions) if pred >= margin]

    # Plot low prediction points in red
    plt.scatter(1/frequencies[low_indices], power[low_indices], color='red', s=100, zorder=5)

    # Plot high prediction points in dark green
    plt.scatter(1/frequencies[high_indices], power[high_indices], color='darkgreen', s=100, zorder=5)

    for i, idx in enumerate(selected_indices):
        plt.annotate(f'{predictions[i]:.2f}', (1/frequencies[idx], power[idx]),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.title(f'Periodogram for {star_name}')
    plt.xscale('log')

    min_period = min(1/frequencies)
    max_period = max(1/frequencies)

    plt.gca().invert_xaxis()
    plt.xlim(max_period, min_period)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.tight_layout()
    plt.show()

def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[TN, FP], [FN, TP]])

def classification_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    report = f"""
    Classification Report:
    Accuracy: {accuracy:.4f}
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    F1-score: {f1_score:.4f}
    """
    return report

def get_planet_names(aliases_data):
    a = 0
    if aliases_data and 'system' in aliases_data and 'objects' in aliases_data['system']:
        planet_set = aliases_data['system']['objects'].get('planet_set', {})
        planets = planet_set.get('planets', {})

        planet_names = []

        if planets:
            for planet_name, planet_info in planets.items():
                planet_names.append(planet_name)
            return planet_names
        else:
            #print("No planets found in the system data.")
            a = 1
    else:
        #print("No planet data available for this system.")
        a = 1

    return []

def get_planet_data(planets):
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    planetary_data = {}

    for planet in planets:
        query = f"select pl_name, pl_orbper, discoverymethod, pl_bmasse from pscomppars where pl_name = '{planet}'"

        params = {
            "query": query,
            "format": "json"
        }

        try:
            print(params)
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if data:
                planet_info = data[0]
                if planet_info['pl_orbper'] is not None and planet_info['discoverymethod'] == 'Radial Velocity':
                    planetary_data[planet] = {
                        'period': planet_info['pl_orbper'],
                        "mass": planet_info['pl_bmasse'],
                        'discovery_method': planet_info['discoverymethod']
                    }
            else:
                print(f"No data found for planet: {planet}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching data for planet {planet}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.content}")

    return planetary_data

def main(hdf5_file_path, model_path, n_peaks, margin, output_csv):
    # Load the model
    model = PlanetDetectionModel_Enhanced().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with h5py.File(hdf5_file_path, 'r') as hf:
        star_names = list(hf.keys())

    print(f"Number of star systems: {len(star_names)}")

    all_true_labels = []
    all_predictions = []
    n_skipped = 0

    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Star Name', 'Planet Name', 'Period', 'Mass', 'Prediction', 'Is True Planet'])

        for star_name in tqdm(star_names, desc="Processing star systems"):
            try:
                # Get system aliases and planetary data
                aliases_data = get_system_aliases(star_name)
                planets = get_planet_names(aliases_data)
                planetary_data = get_planet_data(planets)

                # Skip this system if no valid planets are found
                if not planetary_data:
                    n_skipped += 1
                    continue

                with h5py.File(hdf5_file_path, 'r') as hf:
                    frequencies = hf[star_name]['frequencies'][:]
                    power = hf[star_name]['power'][:]

                # Get top n frequencies by power
                selected_indices, selected_peaks = select_top_frequencies_exclude_entire_peak(frequencies, power, n=n_peaks)

                if len(selected_indices) > 0:
                    # Run inference
                    predictions = run_inference(model, 0, frequencies, power, selected_indices, selected_peaks, device)

                    for i, (idx, pred) in enumerate(zip(selected_indices, predictions)):
                        period = 1 / frequencies[idx]

                        # Check if this period matches any known planet
                        is_true_planet = False
                        matched_planet = None
                        for planet, data in planetary_data.items():
                            known_period = float(data['period'])
                            if abs(period - known_period) / known_period < 0.1:  # 10% tolerance
                                is_true_planet = True
                                matched_planet = planet
                                break

                        all_true_labels.append(int(is_true_planet))
                        all_predictions.append(float(pred))

                        # Write to CSV
                        if matched_planet:
                            csvwriter.writerow([
                                star_name,
                                matched_planet,
                                planetary_data[matched_planet]['period'],
                                planetary_data[matched_planet]['mass'],
                                pred,
                                is_true_planet
                            ])
                        else:
                            csvwriter.writerow([
                                star_name,
                                f"Unknown_{i}",
                                period,
                                "Unknown",
                                pred,
                                is_true_planet
                            ])

            except Exception as e:
                print(f"An error occurred processing star {star_name}: {str(e)}")
                continue

    print("Skipped ", n_skipped, " due to missing data.")

    # Convert predictions to binary (0 or 1) based on the margin
    binary_predictions = [1 if pred > margin else 0 for pred in all_predictions]

    # Compute and print confusion matrix
    cm = confusion_matrix(all_true_labels, binary_predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, binary_predictions))

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze periodograms and detect planets.")
    parser.add_argument("hdf5_file", help="Path to the HDF5 file containing periodograms")
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("--n_peaks", type=int, default=3, help="Number of top peaks to test (default: 3)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Model threshold to classify as detection (default: 0.5)")
    args = parser.parse_args()

    main(args.hdf5_file, args.model_path, args.n_peaks, args.threshold, "planet_predictions.csv")
