import json
import sys
import matplotlib.pyplot as plt

from extraction1 import SENTINEL_RSSI, extract_window
from main_week5 import (
    DynamicProximityStateMachine,
    StdSmoother,
)

TELEMETRY_FILE = "chaos_walk_telemetry.jsonl"
GROUNDTRUTH_FILE = "chaos_walk_groundtruth.jsonl"
WINDOW_MS = 3000
RSSI_FLOOR = -90
MIN_SAMPLES = 3

TRUE_PRESENCE = {"baseline", "chest_hold", "recovered"}
TRUE_ABSENCE = {"walk_away", "backpack_drop"}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def label_at_time(groundtruth, ts):
    current = None
    for rec in groundtruth:
        if rec["timestamp"] <= ts:
            current = rec["action"]
        else:
            break
    return current


def window_telemetry(records, window_ms=WINDOW_MS):
    if not records:
        return []
    windows = []
    start = records[0]["timestamp"]
    end = records[-1]["timestamp"]
    current = start
    while current <= end:
        vals = [
            r["rssi"]
            for r in records
            if current <= r["timestamp"] < current + window_ms and r["rssi"] != SENTINEL_RSSI
        ]
        windows.append((current, vals))
        current += window_ms
    return windows


def run_fsm(windows):
    smoother = StdSmoother(history_length=5)
    fsm = DynamicProximityStateMachine(
        base_grace_seconds=3.0,
        k_factor=1.5,
        max_grace_seconds=15.0,
    )

    timeline = []
    for ts, vals in windows:
        if len(vals) >= MIN_SAMPLES:
            avg_rssi = sum(vals) / len(vals)
            detected = avg_rssi >= RSSI_FLOOR
            std_val = extract_window(vals)["std"]
        else:
            avg_rssi = None
            detected = False
            std_val = None

        if detected and std_val is not None:
            smoothed = smoother.update(std_val)
        else:
            smoothed = smoother.smoothed_value

        state = fsm.update_state(detected, smoothed)
        timeline.append(
            {
                "timestamp": ts,
                "state": state,
                "detected": detected,
                "avg_rssi": avg_rssi,
            }
        )

    return timeline


def count_errors(timeline, groundtruth):
    false_positives = 0
    false_negatives = 0
    prev_state = "PRESENT"
    absent_departure = False
    in_departure = False

    for point in timeline:
        gt_label = label_at_time(groundtruth, point["timestamp"])
        state = point["state"]

        if gt_label in TRUE_ABSENCE:
            if not in_departure:
                in_departure = True
                absent_departure = False
            if state == "ABSENT":
                absent_departure = True
        else:
            if in_departure and not absent_departure:
                false_negatives += 1
            in_departure = False

        if gt_label in TRUE_PRESENCE and state == "ABSENT" and prev_state != "ABSENT":
            false_positives += 1

        prev_state = state

    if in_departure and not absent_departure:
        false_negatives += 1

    return false_positives, false_negatives


def plot_timeline(timeline, groundtruth):
    t0 = timeline[0]["timestamp"]
    times = [(p["timestamp"] - t0) / 1000 for p in timeline]

    state_map = {"PRESENT": 2, "UNSTABLE": 1, "ABSENT": 0}
    state_vals = [state_map[p["state"]] for p in timeline]

    rssi_times = [(p["timestamp"] - t0) / 1000 for p in timeline if p["avg_rssi"] is not None]
    rssi_vals = [p["avg_rssi"] for p in timeline if p["avg_rssi"] is not None]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax1.step(times, state_vals, where="post", color="tab:blue", linewidth=2)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["ABSENT", "UNSTABLE", "PRESENT"])
    ax1.set_title("FSM State Over Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rssi_times, rssi_vals, color="tab:green", linewidth=1.5, marker="o", markersize=2)
    ax2.axhline(
        y=RSSI_FLOOR,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"detection floor ({RSSI_FLOOR} dB)",
    )
    ax2.set_ylabel("RSSI (dB)")
    ax2.set_title("Signal Strength (RSSI) Over Time")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    gt_labels = sorted(set(r["action"] for r in groundtruth))
    gt_map = {label: i for i, label in enumerate(gt_labels)}
    gt_times = [(r["timestamp"] - t0) / 1000 for r in groundtruth]
    gt_vals = [gt_map[r["action"]] for r in groundtruth]

    ax3.step(gt_times, gt_vals, where="post", color="tab:red", linewidth=2)
    ax3.set_yticks(list(gt_map.values()))
    ax3.set_yticklabels(list(gt_map.keys()))
    ax3.set_title("Ground-Truth Action Over Time")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("chaos_walk_result.png", dpi=150)
    print("Saved plot to chaos_walk_result.png")


def main():
    telemetry = load_jsonl(TELEMETRY_FILE)
    groundtruth = load_jsonl(GROUNDTRUTH_FILE)

    print(f"Loaded {len(telemetry)} raw telemetry records from {TELEMETRY_FILE}")
    print(f"Loaded {len(groundtruth)} ground-truth markers from {GROUNDTRUTH_FILE}")

    if len(telemetry) == 0:
        print(f"\nERROR: '{TELEMETRY_FILE}' has no usable records.")
        sys.exit(1)
    if len(groundtruth) == 0:
        print(f"\nERROR: '{GROUNDTRUTH_FILE}' has no markers.")
        sys.exit(1)

    windows = window_telemetry(telemetry)
    print(f"Built {len(windows)} time windows ({WINDOW_MS/1000:.0f}s each)")

    usable_windows = sum(1 for _, vals in windows if len(vals) >= MIN_SAMPLES)
    print(f"{usable_windows} of {len(windows)} windows have >= {MIN_SAMPLES} samples")

    timeline = run_fsm(windows)

    if len(timeline) == 0:
        print("\nERROR: FSM produced an empty timeline despite having windows.")
        sys.exit(1)

    fp, fn = count_errors(timeline, groundtruth)

    print(f"\nTotal windows analyzed: {len(timeline)}")
    print(f"False positives (locked while still present): {fp}")
    print(f"False negatives (missed a true departure):     {fn}")

    plot_timeline(timeline, groundtruth)


if __name__ == "__main__":
    main()
