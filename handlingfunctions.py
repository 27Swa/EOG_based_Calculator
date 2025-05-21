import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt, find_peaks
import os
MODEL_PATH = "Morphological Feature model.joblib"
UPLOAD_DIR = "./Tests"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# === EOG Classifier Configuration ===
SAMPLE_RATE = 176  # Matches test script
LOW_CUTOFF = 0.5
HIGH_CUTOFF = 20
ORDER = 2

label_map = {
    0: 'up',
    1: 'down',
    2: 'right',
    3: 'left',
    4: 'blink'
}


def uploaded_file(upload_file):
    file_path = os.path.join(UPLOAD_DIR, upload_file.name)

    with open(file_path, "wb") as f:
        f.write(upload_file.getvalue())

    return file_path

def delete_file_safely(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def butter_bandpass_filter(Input_Signal, LOW_Cutoff, High_cuttOff, Sampling_Rate, order):
    nyq = 0.5 * Sampling_Rate
    low = LOW_Cutoff / nyq
    high = High_cuttOff / nyq
    Numerator, denominator = butter(order, [low, high], btype="band", output="ba", analog=False, fs=None)
    return filtfilt(Numerator, denominator, Input_Signal)

def validate_signal_file(filepath):
    try:
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read().strip().replace('\n', ',')
                    values = [float(x) for x in content.split(',') if x.strip()]
                    if len(values) < 10:
                        raise ValueError("Signal too short")
                    return np.array(values)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not read file with any supported encoding")
    except Exception as e:
        raise ValueError(f"Invalid signal file: {str(e)}")

def extract_morphological_features(signal_data):
    features = []
    for signal in signal_data:
        peaks, _ = find_peaks(signal)
        valleys, _ = find_peaks(-signal)

        peak_amp, peak_pos = (0, 0) if not peaks.size else (
            signal[peaks[np.argmax(signal[peaks])]],
            peaks[np.argmax(signal[peaks])]
        )

        valley_amp, valley_pos = (0, 0) if not valleys.size else (
            signal[valleys[np.argmin(signal[valleys])]],
            valleys[np.argmin(signal[valleys])]
        )

        features.append([
            np.sum(np.abs(np.diff(signal))),  # wavelength
            peak_amp, valley_amp,
            np.trapz(np.abs(signal)),  # area
            peak_pos, valley_pos
        ])
    return np.array(features)

def features_selection(h_features, v_features):
    columns = [
        'Wavelength (H)', 'Peak Amplitude (H)', 'Valley Amplitude (H)', 'Area Under Curve (H)', 'Peak Position (H)',
        'Valley Position (H)',
        'Wavelength (V)', 'Peak Amplitude (V)', 'Valley Amplitude (V)', 'Area Under Curve (V)', 'Peak Position (V)',
        'Valley Position (V)'
    ]

    combined_features = np.concatenate([h_features, v_features], axis=1)
    features_df = pd.DataFrame(combined_features, columns=columns)

    selected_columns = [
        'Peak Amplitude (H)', 'Peak Position (H)', 'Valley Position (H)',
        'Peak Amplitude (V)', 'Peak Position (V)', 'Valley Position (V)'
    ]
    selected_features = features_df[selected_columns]
    return selected_features

def prediction(df):
    model = joblib.load(MODEL_PATH)

    pred = model.predict(df)[0]
    label = label_map.get(pred, "unknown")
    return label