import json
import time
from collections import defaultdict
import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC_FILTER = "rtls/#"

RAW_LOG_FILE = "fusion_raw_log.jsonl"
FUSED_LOG_FILE = "fusion_fused_log.jsonl"

FUSION_WINDOW_MS = 500

node_buffers = defaultdict(list)


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"Connected to broker (code={reason_code}). Subscribing to {TOPIC_FILTER}")
    client.subscribe(TOPIC_FILTER)


def on_message(client, userdata, msg):
    node_id = msg.topic.split("/")[-1]
    try:
        data = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        return

    record = {"node_id": node_id, "timestamp": data["timestamp"], "rssi": data["rssi"]}
    node_buffers[node_id].append(record)

    with open(RAW_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def fuse_log():
    latest_per_node = {}
    for node_id, readings in node_buffers.items():
        if readings:
            latest_per_node[node_id] = readings[-1]

    if not latest_per_node:
        return

    best_node = max(latest_per_node, key=lambda nid: latest_per_node[nid]["rssi"])
    fused_record = {
        "timestamp": time.time_ns() // 1_000_000,
        "fused_rssi": latest_per_node[best_node]["rssi"],
        "source_node": best_node,
        "all_nodes": {nid: r["rssi"] for nid, r in latest_per_node.items()},
    }

    with open(FUSED_LOG_FILE, "a") as f:
        f.write(json.dumps(fused_record) + "\n")

    print(
        f"[FUSED] {fused_record['fused_rssi']} dBm from {best_node} | "
        f"all nodes: {fused_record['all_nodes']}"
    )


def main():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    print("Hub running. Listening for node publishers... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(FUSION_WINDOW_MS / 1000)
            fuse_log()
    except KeyboardInterrupt:
        print("\nStopping hub.")
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    for f in (RAW_LOG_FILE, FUSED_LOG_FILE):
        open(f, "w").close()
    main()
