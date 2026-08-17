import asyncio
import threading
import time

from flask import Flask, jsonify, render_template_string
from bleak import BleakScanner

from extraction1 import extract_features, SENTINEL_RSSI
from main_week5 import DynamicProximityStateMachine, StdSmoother

TARGET_ADDRESS = "C36EC788-519D-E0E9-7C69-901AC19394B7"
WINDOW_MS = 3000
RSSI_FLOOR = -90
MIN_SAMPLES = 3

app = Flask(__name__)

state_lock = threading.Lock()
shared_state = {
    "fsm_state": "PRESENT",
    "last_rssi": None,
    "smoothed_std": 0.0,
    "active_grace": 0.0,
    "packet_count": 0,
    "last_seen": None,
    "dry_run": True,
    "lock_events": 0,
    "log": [],
}

window_buffer = []
window_lock = threading.Lock()


def log_event(message):
    ts = time.strftime("%H:%M:%S")
    with state_lock:
        shared_state["log"].insert(0, f"[{ts}] {message}")
        shared_state["log"] = shared_state["log"][:20]


def on_present():
    with state_lock:
        shared_state["fsm_state"] = "PRESENT"
    log_event("STATE -> PRESENT")


def on_unstable():
    with state_lock:
        shared_state["fsm_state"] = "UNSTABLE"
        grace = shared_state["active_grace"]
    log_event(f"STATE -> UNSTABLE (grace={grace:.2f}s)")


def on_absent():
    with state_lock:
        shared_state["fsm_state"] = "ABSENT"
        shared_state["lock_events"] += 1
        dry_run = shared_state["dry_run"]
    if dry_run:
        log_event("STATE -> ABSENT (DRY RUN: lock suppressed)")
    else:
        log_event("STATE -> ABSENT (LOCK TRIGGERED)")


fsm = DynamicProximityStateMachine(
    base_grace_seconds=3.0,
    k_factor=1.5,
    max_grace_seconds=15.0,
    on_present=on_present,
    on_unstable=on_unstable,
    on_absent=on_absent,
)
smoother = StdSmoother(history_length=5)


def handle_ble(device, advertising_data):
    if device.address.upper() != TARGET_ADDRESS.upper():
        return
    if advertising_data.rssi == SENTINEL_RSSI:
        return

    with window_lock:
        window_buffer.append(advertising_data.rssi)

    with state_lock:
        shared_state["packet_count"] += 1
        shared_state["last_rssi"] = advertising_data.rssi
        shared_state["last_seen"] = time.strftime("%H:%M:%S")


def process_window():
    while True:
        time.sleep(WINDOW_MS / 1000)
        with window_lock:
            vals = window_buffer.copy()
            window_buffer.clear()

        if len(vals) >= MIN_SAMPLES:
            avg_rssi = sum(vals) / len(vals)
            detected = avg_rssi >= RSSI_FLOOR
            std_val = extract_features(vals)["std"]
        else:
            detected = False
            std_val = None

        if detected and std_val is not None:
            smoothed = smoother.update(std_val)
        else:
            smoothed = smoother.smoothed_value

        with state_lock:
            shared_state["smoothed_std"] = smoothed

        fsm.update_state(detected, smoothed)

        with state_lock:
            shared_state["active_grace"] = fsm.active_grace_period


async def run_scanner():
    scanner = BleakScanner(detection_callback=handle_ble)
    await scanner.start()
    while True:
        await asyncio.sleep(0.5)


def start_background_threads():
    threading.Thread(target=lambda: asyncio.run(run_scanner()), daemon=True).start()
    threading.Thread(target=process_window, daemon=True).start()


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Proximity FSM Dashboard</title>
  <meta charset="utf-8">
  <style>
    body {
      font-family: -apple-system, sans-serif;
      background: #111;
      color: #eee;
      padding: 2rem;
    }
    .state {
      font-size: 2.5rem;
      font-weight: bold;
      padding: 1rem 2rem;
      border-radius: 8px;
      display: inline-block;
    }
    .PRESENT { background: #1a5c1a; }
    .UNSTABLE { background: #8a6d1a; }
    .ABSENT { background: #7a1a1a; }
    .metrics {
      margin-top: 1.5rem;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
    }
    .metric {
      background: #222;
      padding: 1rem;
      border-radius: 6px;
    }
    .metric .label { font-size: 0.8rem; color: #999; }
    .metric .value { font-size: 1.4rem; }
    .log {
      margin-top: 1.5rem;
      background: #1a1a1a;
      padding: 1rem;
      border-radius: 6px;
      height: 300px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 0.85rem;
    }
    .toggle { margin-top: 1rem; }
    button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
  </style>
</head>
<body>
  <h1>Proximity FSM Live Status</h1>
  <div id="state" class="state PRESENT">PRESENT</div>

  <div class="metrics">
    <div class="metric">
      <div class="label">Last RSSI</div>
      <div class="value" id="rssi">--</div>
    </div>
    <div class="metric">
      <div class="label">Smoothed Std</div>
      <div class="value" id="std">--</div>
    </div>
    <div class="metric">
      <div class="label">Active Grace</div>
      <div class="value" id="grace">--</div>
    </div>
    <div class="metric">
      <div class="label">Packets Seen</div>
      <div class="value" id="packets">--</div>
    </div>
    <div class="metric">
      <div class="label">Last Seen</div>
      <div class="value" id="lastseen">--</div>
    </div>
    <div class="metric">
      <div class="label">Lock Events</div>
      <div class="value" id="lockevents">--</div>
    </div>
  </div>

  <div class="toggle">
    <button onclick="toggleDryRun()">
      Toggle Dry Run (currently: <span id="dryrun">--</span>)
    </button>
  </div>

  <h3>Event Log</h3>
  <div class="log" id="log"></div>

  <script>
    async function refresh() {
      const res = await fetch('/status');
      const data = await res.json();

      const stateEl = document.getElementById('state');
      stateEl.textContent = data.fsm_state;
      stateEl.className = 'state ' + data.fsm_state;

      const rssiEl = document.getElementById('rssi');
      rssiEl.textContent = data.last_rssi !== null
        ? data.last_rssi + ' dBm'
        : '--';

      document.getElementById('std').textContent =
        data.smoothed_std.toFixed(2);
      document.getElementById('grace').textContent =
        data.active_grace.toFixed(2) + 's';
      document.getElementById('packets').textContent =
        data.packet_count;
      document.getElementById('lastseen').textContent =
        data.last_seen || '--';
      document.getElementById('lockevents').textContent =
        data.lock_events;
      document.getElementById('dryrun').textContent =
        data.dry_run ? 'ON' : 'OFF';

      document.getElementById('log').innerHTML = data.log
        .map(function (l) { return '<div>' + l + '</div>'; })
        .join('');
    }

    async function toggleDryRun() {
      await fetch('/toggle_dry_run', { method: 'POST' });
      refresh();
    }

    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/status")
def status():
    with state_lock:
        return jsonify(dict(shared_state))


@app.route("/toggle_dry_run", methods=["POST"])
def toggle_dry_run():
    with state_lock:
        shared_state["dry_run"] = not shared_state["dry_run"]
    return jsonify({"dry_run": shared_state["dry_run"]})


if __name__ == "__main__":
    start_background_threads()
    app.run(host="127.0.0.1", port=5000, debug=False)
