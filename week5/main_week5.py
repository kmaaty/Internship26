import time
import os
import platform
from collections import deque

from extraction1 import extract_window, SENTINEL_RSSI


class StdSmoother:

    def __init__(self, history_length=5):
        self.history_length = history_length
        self.history = deque(maxlen=history_length)

    def update(self, latest_std):
        self.history.append(latest_std)
        return self.smoothed_value

    @property
    def smoothed_value(self):
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    @property
    def is_warmed_up(self):
        return len(self.history) >= self.history_length


class DynamicProximityStateMachine:

    def __init__(
        self,
        base_grace_seconds=3.0,
        k_factor=1.5,
        max_grace_seconds=15.0,
        on_absent=None,
        on_present=None,
        on_unstable=None,
    ):
        self.current_state = "PRESENT"
        self.base_grace = base_grace_seconds
        self.k_factor = k_factor
        self.max_grace = max_grace_seconds
        self.unstable_start_time = None
        self.active_grace_period = base_grace_seconds

        self.on_absent = on_absent or self._default_lock_action
        self.on_present = on_present or (lambda: None)
        self.on_unstable = on_unstable or (lambda: None)

    def _compute_grace_period(self, smoothed_std):
        grace = self.base_grace + self.k_factor * smoothed_std
        return min(grace, self.max_grace)

    def update_state(self, is_user_detected: bool, smoothed_std: float):
        current_time = time.time()

        if is_user_detected:
            if self.current_state != "PRESENT":
                self.current_state = "PRESENT"
                self.unstable_start_time = None
                self.on_present()
            return self.current_state

        if self.current_state == "PRESENT":
            self.current_state = "UNSTABLE"
            self.unstable_start_time = current_time
            self.active_grace_period = self._compute_grace_period(smoothed_std)
            self.on_unstable()

        elif self.current_state == "UNSTABLE":
            elapsed = current_time - self.unstable_start_time
            if elapsed >= self.active_grace_period:
                self.current_state = "ABSENT"
                self.on_absent()

        return self.current_state


@staticmethod
def _default_lock_action():
    system = platform.system()
    if system == "Windows":
        os.system("rundll32.exe user32.dll,LockWorkStation")
    elif system == "Darwin":
        os.system(
            "/System/Library/CoreServices/Menu\\ Extras/User.menu/"
            "Contents/Resources/CGSession -suspend"
        )
    else:
        os.system("loginctl lock-session")


class DetectionBridge:

    def __init__(self, RSSI_FLOOR=-90, MIN_SAMPLES=3):
        self.RSSI_FLOOR = RSSI_FLOOR
        self.MIN_SAMPLES = MIN_SAMPLES
        self.window_buffer = []

    def add_sample(self, rssi_value):
        if rssi_value == SENTINEL_RSSI:
            return
        self.window_buffer.append(rssi_value)

    def process_window(self):
        if len(self.window_buffer) < self.MIN_SAMPLES:
            self.window_buffer = []
            return False, None

        avg_rssi = sum(self.window_buffer) / len(self.window_buffer)
        detected = avg_rssi >= self.RSSI_FLOOR
        features = extract_window(self.window_buffer)
        std_val = features["std"]

        self.window_buffer = []
        return detected, std_val


def main():
    def log_present():
        print(f"[{time.strftime('%H:%M:%S')}] STATE -> PRESENT")

    def log_unstable():
        print(
            f"[{time.strftime('%H:%M:%S')}] STATE -> UNSTABLE "
            f"(grace={fsm.active_grace_period:.2f}s, smoothed_std={smoother.smoothed_value:.2f})"
        )

    def log_absent():
        print(f"[{time.strftime('%H:%M:%S')}] STATE -> ABSENT (lock triggered)")

    smoother = StdSmoother(history_length=5)
    fsm = DynamicProximityStateMachine(
        base_grace_seconds=3.0,
        k_factor=1.5,
        max_grace_seconds=15.0,
        on_present=log_present,
        on_unstable=log_unstable,
        on_absent=log_absent,
    )

    simulated_feed = [(True, 6.0)] * 5 + [(False, 0.0)] * 8 + [(True, 0.5)] * 5 + [(False, 0.0)] * 8

    for tick, (detected, std_val) in enumerate(simulated_feed):
        if detected:
            smoothed = smoother.update(std_val)
        else:
            smoothed = smoother.smoothed_value

        state = fsm.update_state(detected, smoothed)
        print(
            f"tick={tick:02d}  detected={detected}  raw_std={std_val:.1f}  "
            f"smoothed_std={smoothed:.2f}  state={state}"
        )
        time.sleep(1)


if __name__ == "__main__":
    main()
