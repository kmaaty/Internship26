import json
import numpy as np
import matplotlib.pyplot as plt

RAW_LOG_FILE = "fusion_raw_log.jsonl"
SENTINEL_RSSI = 127
N_POINTS = 200


def load_split(path):
    per_node = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "rssi" not in rec or "timestamp" not in rec:
                continue
            if rec["rssi"] == SENTINEL_RSSI:
                continue
            per_node.setdefault(rec["node_id"], []).append((rec["timestamp"], rec["rssi"]))

    streams = {}
    for node_id, records in per_node.items():
        records.sort(key=lambda r: r[0])
        timestamps = np.array([r[0] for r in records], dtype=float)
        values = np.array([r[1] for r in records], dtype=float)
        elapsed = timestamps - timestamps[0]
        span = elapsed[-1] if elapsed[-1] > 0 else 1.0
        progress = elapsed / span
        streams[node_id] = (progress, values)
        print(
            f"{node_id}: {len(values)} readings, "
            f"active window {records[0][0]} -> {records[-1][0]} "
            f"({span/1000:.1f}s)"
        )
    return streams


def resample_to_grid(progress, values, grid):
    return np.interp(grid, progress, values)


def main():
    streams = load_split(RAW_LOG_FILE)
    if len(streams) < 2:
        print("Need at least 2 distinct node_ids in the log. Found:", list(streams.keys()))
        return

    node_ids = sorted(streams.keys())
    n1, n2 = node_ids[0], node_ids[1]
    grid = np.linspace(0, 1, N_POINTS)

    v1 = resample_to_grid(*streams[n1], grid)
    v2 = resample_to_grid(*streams[n2], grid)

    fused = np.maximum(v1, v2)
    worst_single = np.minimum(v1, v2)
    db_improvement = fused - worst_single

    print(f"\nAverage dB improvement from fusion: {np.mean(db_improvement):.2f} dB")
    print(f"Maximum dB improvement from fusion: {np.max(db_improvement):.2f} dB")
    print(f"{n1} mean/min: {np.mean(v1):.2f} / {np.min(v1):.2f}")
    print(f"{n2} mean/min: {np.mean(v2):.2f} / {np.min(v2):.2f}")
    print(f"Fused mean/min: {np.mean(fused):.2f} / {np.min(fused):.2f}")

    angle_deg = grid * 360
    plt.figure(figsize=(11, 6))
    plt.plot(angle_deg, v1, label=f"{n1} (Position A)", alpha=0.7)
    plt.plot(angle_deg, v2, label=f"{n2} (Position B)", alpha=0.7)
    plt.plot(angle_deg, fused, label="Fused (max of both)", linewidth=2.5, color="black")
    plt.xlabel("Rotation Progress (degrees, approx.)")
    plt.ylabel("RSSI (dBm)")
    plt.title("Single-Receiver vs Fused Multi-Receiver Signal During 360° Rotation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rotation_fusion_comparison.png", dpi=150)
    print("\nSaved rotation_fusion_comparison.png")


if __name__ == "__main__":
    main()
