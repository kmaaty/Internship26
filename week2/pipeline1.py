import asyncio
import json
import time
import numpy as np
from bleak import BleakScanner

OUTPUT_FILE = "telemetry_raw.jsonl"
TARGET_ADDRESS = "C36EC788-519D-E0E9-7C69-901AC19394B7"
BATCH_INTERVAL_MS = 1000
SENTINEL_RSSI = 127

packet_queue = asyncio.Queue()


def ble_packet(device, advertisement_data):
    if device.address != TARGET_ADDRESS:
        return
    if advertisement_data.rssi == SENTINEL_RSSI:
        return
    payload = {
        "timestamp": time.time_ns() // 1_000_000,
        "node_id": "receiver_node_01",
        "rssi": advertisement_data.rssi,
    }
    packet_queue.put_nowait(payload)


async def batch_writer(interval_ms=BATCH_INTERVAL_MS):
    while True:
        await asyncio.sleep(interval_ms / 1000)
        batch = []
        while not packet_queue.empty():
            batch.append(packet_queue.get_nowait())
        if batch:
            with open(OUTPUT_FILE, "a") as f:
                for p in batch:
                    f.write(json.dumps(p) + "\n")


async def run_ingestion():
    scanner = BleakScanner(detection_callback=ble_packet)
    await scanner.start()
    print("Listening for BLE advertisements (queued/batched mode)... Ctrl+C to stop.")
    await batch_writer()


def load_rssi(path):
    values = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if record["rssi"] != SENTINEL_RSSI:
                values.append(record["rssi"])
    return values


class ReferenceFilterPipeline:

    def __init__(self, process_variance=0.1, measurement_variance=4.0):
        self.Q = process_variance
        self.R = measurement_variance
        self.x = -60.0
        self.P = 1.0

    def compute_kalman(self, raw_rssi: float) -> float:
        self.P = self.P + self.Q
        kalman_gain = self.P / (self.P + self.R)
        self.x = self.x + kalman_gain * (raw_rssi - self.x)
        self.P = (1 - kalman_gain) * self.P
        return self.x


def hampel_filter_stream(data, window_size=5, threshold=3.0):
    output = []
    buf = []
    for x in data:
        buf.append(x)
        if len(buf) > window_size:
            buf.pop(0)
        if len(buf) < window_size:
            output.append(x)
            continue
        median = np.median(buf)
        mad = np.median(np.abs(np.array(buf) - median)) * 1.4826
        if mad == 0 or abs(x - median) <= threshold * mad:
            output.append(x)
        else:
            output.append(median)
    return output


def full_signal_pipeline(raw_rssi_stream, hampel_window=5, hampel_threshold=2.0, Q=0.01, R=1.0):
    scrubbed = hampel_filter_stream(
        raw_rssi_stream, window_size=hampel_window, threshold=hampel_threshold
    )
    kf = ReferenceFilterPipeline(process_variance=Q, measurement_variance=R)
    return [kf.compute_kalman(x) for x in scrubbed]


def qr_grid_search(
    raw_rssi_stream, ground_truth, q_values=(0.01, 0.1, 1.0), r_values=(1.0, 4.0, 16.0)
):
    results = []
    for q in q_values:
        for r in r_values:
            kf = ReferenceFilterPipeline(process_variance=q, measurement_variance=r)
            smoothed = [kf.compute_kalman(x) for x in raw_rssi_stream]
            rmse = np.sqrt(np.mean((np.array(smoothed) - ground_truth) ** 2))
            results.append({"Q": q, "R": r, "RMSE": round(rmse, 4)})
    return sorted(results, key=lambda r: r["RMSE"])


if __name__ == "__main__":
    try:
        asyncio.run(run_ingestion())
    except KeyboardInterrupt:
        print("\nStopped. Telemetry saved to", OUTPUT_FILE)
