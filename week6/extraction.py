import json
import numpy as np
from scipy.stats import skew

SENTINEL_RSSI = 127
WINDOW_MS = 3000
MIN_SAMPLES = 3

LOCATION_FILES = {
    "backroom": "backroom.txt",
    "snacks": "snacks.txt",
    "tape": "tape.txt",
    "table": "table.txt",
}


def load_recording(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "rssi" not in rec or "timestamp" not in rec:
                continue
            if rec["rssi"] == SENTINEL_RSSI:
                continue
            records.append((rec["timestamp"], rec["rssi"]))
    records.sort(key=lambda r: r[0])
    return records


def extract_window(rssi_values):
    arr = np.array(rssi_values, dtype=float)
    mean_rssi = arr.mean()
    std_rssi = arr.std()
    if std_rssi == 0 or len(arr) <= 2:
        skew_rssi = 0.0
    else:
        skew_rssi = skew(arr)
    delta = np.diff(arr)
    delta_variance = np.var(delta) if len(delta) > 0 else 0.0
    return {
        "mean": mean_rssi,
        "std": std_rssi,
        "skew": skew_rssi,
        "delta_var": delta_variance,
    }


def window_extract(records, window_ms=WINDOW_MS, label=None):
    features = []
    if not records:
        return features

    start_time = records[0][0]
    end_time = records[-1][0]
    current = start_time

    while current <= end_time:
        window_vals = [rssi for ts, rssi in records if current <= ts < current + window_ms]
        if len(window_vals) >= MIN_SAMPLES:
            feats = extract_window(window_vals)
            feats["label"] = label
            features.append(feats)
        current += window_ms

    return features


def main():
    all_features = []
    print(
        f"Windowing at {WINDOW_MS/1000:.0f}s per window "
        f"(minimum {MIN_SAMPLES} samples/window)\n"
    )

    for location, filepath in LOCATION_FILES.items():
        records = load_recording(filepath)
        span_s = (records[-1][0] - records[0][0]) / 1000 if records else 0
        feats = window_extract(records, label=location)
        all_features.extend(feats)
        print(
            f"{location:12s} | raw readings: {len(records):3d} | "
            f"span: {span_s:5.1f}s | usable windows: {len(feats)}"
        )

    print(f"\nTotal feature samples across all locations: {len(all_features)}")
    return all_features


if __name__ == "__main__":
    features = main()
