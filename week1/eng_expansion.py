import asyncio
from bleak import BleakScanner
import json
import time

OUTPUT_FILE = "telemetry_raw.jsonl"
TARGET_ADDRESS = "C36EC788-519D-E0E9-7C69-901AC19394B7"

packet_queue = asyncio.Queue()


def ble_packet(device, advertisement_data):
    if device.address == TARGET_ADDRESS:
        payload = {
            "timestamp": time.time_ns() // 1_000_000,
            "node_id": "receiver_node_01",
            "rssi": advertisement_data.rssi,
        }
        packet_queue.put_nowait(payload)


async def batch_writer(interval_ms=1000):
    while True:
        await asyncio.sleep(interval_ms / 1000)
        batch = []
        while not packet_queue.empty():
            batch.append(packet_queue.get_nowait())
        if batch:
            with open(OUTPUT_FILE, "a") as f:
                for p in batch:
                    f.write(json.dumps(p) + "\n")


async def main():
    scanner = BleakScanner(detection_callback=ble_packet)
    await scanner.start()
    print("Listening (queued/batched mode)... Press Ctrl+C to stop.")
    await batch_writer()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped. Telemetry saved to", OUTPUT_FILE)
