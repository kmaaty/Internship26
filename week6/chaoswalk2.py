import asyncio
import json
import time
import threading

from bleak import BleakScanner

TARGET_ADDRESS = "C36EC788-519D-E0E9-7C69-901AC19394B7"

TELEMETRY_FILE = "chaos_walk_telemetry.jsonl"
GROUNDTRUTH_FILE = "chaos_walk_groundtruth.jsonl"

SCRIPTED_ACTIONS = ["deep_pocket", "deep_backpack", "wall_layers", "2.4GHz_noise", "recovered"]

stop_flag = threading.Event()
packet_count = 0


def handle_ble(device, advertising_data):
    global packet_count
    if device.address.upper() != TARGET_ADDRESS.upper():
        return

    record = {
        "timestamp": time.time_ns() // 1_000_000,
        "rssi": advertising_data.rssi,
    }
    with open(TELEMETRY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    packet_count += 1


def progress_printer():
    last_count = -1
    while not stop_flag.is_set():
        time.sleep(2.0)
        if packet_count != last_count:
            print(f"  [scanner] {packet_count} phone packets captured so far...")
            last_count = packet_count


def groundtruth_input_loop():
    with open(GROUNDTRUTH_FILE, "w") as gt:
        idx = 0
        print("\n=== CHAOS WALK GROUND TRUTH LOGGER ===")
        print("Scripted order:", " -> ".join(SCRIPTED_ACTIONS))
        print("Press ENTER to advance to the next scripted action,")
        print("or type a custom label and press ENTER for an ad-hoc marker.")
        print("Type 'stop' to end the recording.\n")

        while idx < len(SCRIPTED_ACTIONS):
            typed = input(f"[next: {SCRIPTED_ACTIONS[idx]}] > ").strip()
            ts = time.time_ns() // 1_000_000

            if typed.lower() == "stop":
                break

            action = typed if typed else SCRIPTED_ACTIONS[idx]
            gt.write(json.dumps({"timestamp": ts, "action": action}) + "\n")
            gt.flush()
            print(f"  logged '{action}' @ {ts}")

            if not typed:
                idx += 1

        end_ts = time.time_ns() // 1_000_000
        gt.write(json.dumps({"timestamp": end_ts, "action": "end"}) + "\n")

    stop_flag.set()


async def run_scanner():
    scanner = BleakScanner(detection_callback=handle_ble)
    await scanner.start()
    print(f"Scanner started, filtering for {TARGET_ADDRESS}. Waiting for packets...\n")
    while not stop_flag.is_set():
        await asyncio.sleep(0.5)
    await scanner.stop()


def main():
    open(TELEMETRY_FILE, "w").close()

    progress_thread = threading.Thread(target=progress_printer, daemon=True)
    progress_thread.start()

    input_thread = threading.Thread(target=groundtruth_input_loop, daemon=True)
    input_thread.start()

    try:
        asyncio.run(run_scanner())
    except Exception as e:
        print(f"\nSCANNER ERROR: {type(e).__name__}: {e}")
        stop_flag.set()

    input_thread.join(timeout=1)
    print(f"\nCaptured {packet_count} phone packets.")
    print(f"Saved filtered telemetry to {TELEMETRY_FILE}")
    print(f"Saved ground truth to {GROUNDTRUTH_FILE}")


if __name__ == "__main__":
    main()
