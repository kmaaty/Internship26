import json

rssi_values = []
with open("telemetry_raw.jsonl") as f:
    for line in f:
        rssi_values.append(json.loads(line)["rssi"])


class ReferenceFilterPipeline:
    def __init__(self, process_variance=0.1, measurement_variance=4.0):
        self.Q, self.R = process_variance, measurement_variance
        self.x, self.P = -60.0, 1.0

    def compute_kalman(self, raw_rssi: float) -> float:
        self.P = self.P + self.Q
        kalman_gain = self.P / (self.P + self.R)
        self.x = self.x + kalman_gain * (raw_rssi - self.x)
        self.P = (1 - kalman_gain) * self.P
        return self.x


kf = ReferenceFilterPipeline()
smoothed = [kf.compute_kalman(r) for r in rssi_values]
print(smoothed[:10])
