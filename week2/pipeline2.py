import numpy as np
import matplotlib.pyplot as plt
import csv

from pipeline1 import (
    load_rssi,
    full_signal_pipeline,
    qr_grid_search,
)

CLEAN_FILE = "telemetry_rawC.jsonl"
DIRTY_FILE = "telemetry_rawD.jsonl"


def main():
    clean_rssi = load_rssi(CLEAN_FILE)
    dirty_rssi = load_rssi(DIRTY_FILE)
    ground_truth = np.mean(clean_rssi)

    print(f"Loaded {len(clean_rssi)} clean samples, ground truth = {ground_truth:.2f} dBm")
    print(f"Loaded {len(dirty_rssi)} dirty samples\n")

    grid_results = qr_grid_search(clean_rssi, ground_truth)
    print("Q/R Tuning Grid (best to worst RMSE):")
    for r in grid_results:
        print(f"  Q={r['Q']:<5} R={r['R']:<5} RMSE={r['RMSE']}")

    with open("qr_grid_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Q", "R", "RMSE"])
        writer.writeheader()
        writer.writerows(grid_results)
    print("\nSaved qr_grid_results.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    qs = [r["Q"] for r in grid_results]
    rmses = [r["RMSE"] for r in grid_results]
    rs = [r["R"] for r in grid_results]
    sc = ax.scatter(qs, rmses, c=rs, cmap="viridis", s=150, edgecolor="black")
    for r in grid_results:
        ax.annotate(
            f"Q={r['Q']}, R={r['R']}",
            (r["Q"], r["RMSE"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Process Variance (Q)")
    ax.set_ylabel("RMSE vs Stationary Ground Truth")
    ax.set_title("Kalman Filter Q/R Tuning Grid")
    plt.colorbar(sc, label="Measurement Variance (R)")
    plt.tight_layout()
    plt.savefig("qr_tuning_scatter.png", dpi=150)
    print("Saved qr_tuning_scatter.png")

    best = grid_results[0]
    final_smoothed = full_signal_pipeline(dirty_rssi, Q=best["Q"], R=best["R"])
    kalman_only = [
        x for x in full_signal_pipeline(dirty_rssi, hampel_window=1, Q=best["Q"], R=best["R"])
    ]

    print(f"\nUsing best Q/R (Q={best['Q']}, R={best['R']}) on dirty dataset:")
    print(f"  Min value, Hampel+Kalman: {min(final_smoothed):.2f}")
    print(f"  Min value, Kalman-only:   {min(kalman_only):.2f}")


if __name__ == "__main__":
    main()
