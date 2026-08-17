import json
import sys
import matplotlib.pyplot as plt

from extraction import extract_window, SENTINEL_RSSI
from main_week5 import DynamicProximityStateMachine, StdSmoother

TELEMETRY_FILE = "chaos_walk_telemetry.jsonl"
GROUNDTRUTH_FILE = "chaos_walk_groundtruth.jsonl"
WINDOW_MS = 3000
RSSI_THRESHOLD_FLOOR = -90
MIN_SAMPLES_PER_WINDOW = 3

ATTENUATION_CONDITIONS = {"deep_pocket", "wall_layers", "deep_backpack"}
INTERFERENCE_CONDITIONS = {"2.4GHz_noise"}
RECOVERY_CONDITIONS = {"recovered"}


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


def run_fsm_over_windows(windows):
    smoother = StdSmoother(history_length=5)
    fsm = DynamicProximityStateMachine(
        base_grace_seconds=3.0, k_factor=1.5, max_grace_seconds=15.0
    )

    timeline = []
    for ts, vals in windows:
        if len(vals) >= MIN_SAMPLES_PER_WINDOW:
            avg_rssi = sum(vals) / len(vals)
            detected = avg_rssi >= RSSI_THRESHOLD_FLOOR
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
            {"timestamp": ts, "state": state, "detected": detected, "avg_rssi": avg_rssi}
        )

    return timeline


def summarize_by_condition(timeline, groundtruth):
    conditions = sorted(set(r["action"] for r in groundtruth))
    summary = {}

    for condition in conditions:
        points = [p for p in timeline if label_at_time(groundtruth, p["timestamp"]) == condition]
        rssi_vals = [p["avg_rssi"] for p in points if p["avg_rssi"] is not None]
        absent_count = sum(1 for p in points if p["state"] == "ABSENT")
        unstable_count = sum(1 for p in points if p["state"] == "UNSTABLE")
        present_count = sum(1 for p in points if p["state"] == "PRESENT")

        summary[condition] = {
            "windows": len(points),
            "avg_rssi": sum(rssi_vals) / len(rssi_vals) if rssi_vals else None,
            "min_rssi": min(rssi_vals) if rssi_vals else None,
            "max_rssi": max(rssi_vals) if rssi_vals else None,
            "present_windows": present_count,
            "unstable_windows": unstable_count,
            "absent_windows": absent_count,
            "flagged_as_false_positive": (
                condition in ATTENUATION_CONDITIONS | INTERFERENCE_CONDITIONS | RECOVERY_CONDITIONS
                and absent_count > 0
            ),
        }

    return summary


def plot_timeline(timeline, groundtruth):
    t0 = timeline[0]["timestamp"]
    times = [(p["timestamp"] - t0) / 1000 for p in timeline]
    state_map = {"PRESENT": 2, "UNSTABLE": 1, "ABSENT": 0}
    state_vals = [state_map[p["state"]] for p in timeline]

    rssi_times = [(p["timestamp"] - t0) / 1000 for p in timeline if p["avg_rssi"] is not None]
    rssi_vals = [p["avg_rssi"] for p in timeline if p["avg_rssi"] is not None]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    ax1.step(times, state_vals, where="post", color="tab:blue", linewidth=2)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["ABSENT", "UNSTABLE", "PRESENT"])
    ax1.set_title("FSM State Over Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rssi_times, rssi_vals, color="tab:green", linewidth=1.5, marker="o", markersize=2)
    ax2.axhline(
        y=RSSI_THRESHOLD_FLOOR,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"detection floor ({RSSI_THRESHOLD_FLOOR} dB)",
    )
    ax2.set_ylabel("RSSI (dB)")
    ax2.set_title("Signal Strength Across All Adversarial Conditions")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    gt_labels = sorted(set(r["action"] for r in groundtruth))
    gt_map = {label: i for i, label in enumerate(gt_labels)}
    gt_times = [(r["timestamp"] - t0) / 1000 for r in groundtruth]
    gt_vals = [gt_map[r["action"]] for r in groundtruth]

    ax3.step(gt_times, gt_vals, where="post", color="tab:red", linewidth=2)
    ax3.set_yticks(list(gt_map.values()))
    ax3.set_yticklabels(list(gt_map.keys()))
    ax3.set_title("Adversarial Condition Over Time")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("week6_result.png", dpi=150)
    print("Saved plot to week6_result.png")


def main():
    telemetry = load_jsonl(TELEMETRY_FILE)
    groundtruth = load_jsonl(GROUNDTRUTH_FILE)

    print(f"Loaded {len(telemetry)} telemetry records, {len(groundtruth)} ground-truth markers")

    if len(telemetry) == 0 or len(groundtruth) == 0:
        print("ERROR: one of the input files is empty.")
        sys.exit(1)

    windows = window_telemetry(telemetry)
    timeline = run_fsm_over_windows(windows)

    if not timeline:
        print("ERROR: no usable windows produced from telemetry.")
        sys.exit(1)

    summary = summarize_by_condition(timeline, groundtruth)

    print(f"\nTotal windows analyzed: {len(timeline)}\n")
    print(
        f"{'Condition':<16} {'Windows':>8} {'AvgRSSI':>9} {'MinRSSI':>9} "
        f"{'PRESENT':>8} {'UNSTABLE':>9} {'ABSENT':>7} {'FalsePos?':>10}"
    )
    print("-" * 82)
    for condition, s in summary.items():
        avg_rssi = f"{s['avg_rssi']:.1f}" if s["avg_rssi"] is not None else "--"
        min_rssi = f"{s['min_rssi']:.1f}" if s["min_rssi"] is not None else "--"
        flag = "YES" if s["flagged_as_false_positive"] else "no"
        print(
            f"{condition:<16} {s['windows']:>8} {avg_rssi:>9} {min_rssi:>9} "
            f"{s['present_windows']:>8} {s['unstable_windows']:>9} "
            f"{s['absent_windows']:>7} {flag:>10}"
        )

    plot_timeline(timeline, groundtruth)


if __name__ == "__main__":
    main()
