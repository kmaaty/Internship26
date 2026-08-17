import argparse
import asyncio
import json
import time
from bleak import BleakScanner
import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TARGET_ADDRESS = "C36EC788-519D-E0E9-7C69-901AC19394B7"
SENTINEL_RSSI = 127


def build_publisher():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()
    return client


def make_handler(mqtt_client, node_id):
    topic = f"rtls/{node_id}"

    def handle_ble(device, advertisement_data):
        if device.address != TARGET_ADDRESS:
            return
        if advertisement_data.rssi == SENTINEL_RSSI:
            return
        payload = json.dumps(
            {
                "timestamp": time.time_ns() // 1_000_000,
                "rssi": advertisement_data.rssi,
            }
        )
        mqtt_client.publish(topic, payload)

    return handle_ble


async def run_node(node_id):
    mqtt_client = build_publisher()
    handler = make_handler(mqtt_client, node_id)

    scanner = BleakScanner(detection_callback=handler)
    await scanner.start()
    print(f"[{node_id}] Scanning and publishing to topic 'rtls/{node_id}'. Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await scanner.stop()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, help="Node identifier, e.g. node1, node2")
    args = parser.parse_args()

    try:
        asyncio.run(run_node(args.node))
    except KeyboardInterrupt:
        print(f"\n[{args.node}] Stopped.")
