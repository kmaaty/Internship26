import asyncio
import json
import os
import time
import csv
import psutil
import matplotlib.pyplot as plt

TEST_DURATION_S = 2.0
FREQUENCIES_HZ = [20, 100, 500]
MAX_BACKLOG_MS = 50
BASELINE_FILE = "baseline_test.jsonl"
QUEUED_FILE = "queued_test.jsonl"


def make_payload(i):
    return json.dumps(
        {
            "timestamp": time.time_ns() // 1_000_000,
            "node_id": "receiver_node_01",
            "rssi": -60 - (i % 10),
        }
    )


def blocking_write(path, payload):
    with open(path, "a") as f:
        f.write(payload + "\n")
        f.flush()
        os.fsync(f.fileno())


async def run_baseline(rate_hz, duration_s, proc):
    if os.path.exists(BASELINE_FILE):
        os.remove(BASELINE_FILE)
    interval = 1.0 / rate_hz
    n_packets = int(rate_hz * duration_s)
    dropped = 0
    backlog_s = 0.0
    cpu_samples = []
    proc.cpu_percent(interval=None)
    start = time.perf_counter()
    for i in range(n_packets):
        next_arrival = start + i * interval
        now = time.perf_counter()
        sleep_time = next_arrival - now
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
            backlog_s = max(0.0, backlog_s - sleep_time)

        t0 = time.perf_counter()
        if backlog_s * 1000 > MAX_BACKLOG_MS:
            dropped += 1
        else:
            blocking_write(BASELINE_FILE, make_payload(i))
        t1 = time.perf_counter()
        proc_time = t1 - t0
        backlog_s += max(0.0, proc_time - interval)
        cpu_samples.append(proc.cpu_percent(interval=None))
    total_time = time.perf_counter() - start
    return dropped, sum(cpu_samples) / len(cpu_samples), total_time


async def run_queued(rate_hz, duration_s, proc):
    if os.path.exists(QUEUED_FILE):
        os.remove(QUEUED_FILE)
    queue = asyncio.Queue()
    interval = 1.0 / rate_hz
    n_packets = int(rate_hz * duration_s)
    cpu_samples = []
    proc.cpu_percent(interval=None)

    async def producer():
        start = time.perf_counter()
        for i in range(n_packets):
            next_arrival = start + i * interval
            now = time.perf_counter()
            sleep_time = next_arrival - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            queue.put_nowait(make_payload(i))
            cpu_samples.append(proc.cpu_percent(interval=None))

    async def consumer():
        while True:
            await asyncio.sleep(1.0)
            batch = []
            while not queue.empty():
                batch.append(queue.get_nowait())
            if batch:
                with open(QUEUED_FILE, "a") as f:
                    for p in batch:
                        f.write(p + "\n")

    start = time.perf_counter()
    consumer_task = asyncio.create_task(consumer())
    await producer()
    await asyncio.sleep(1.1)
    consumer_task.cancel()
    total_time = time.perf_counter() - start

    remaining = []
    while not queue.empty():
        remaining.append(queue.get_nowait())
    dropped = 0
    return dropped, sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0, total_time


async def main():
    proc = psutil.Process()
    results = []
    for rate in FREQUENCIES_HZ:
        b_dropped, b_cpu, b_time = await run_baseline(rate, TEST_DURATION_S, proc)
        results.append(("baseline", rate, b_dropped, round(b_cpu, 2), round(b_time, 3)))
        print(f"[BASELINE] {rate}Hz -> dropped={b_dropped}, avg_cpu={b_cpu:.2f}%")

        q_dropped, q_cpu, q_time = await run_queued(rate, TEST_DURATION_S, proc)
        results.append(("queued", rate, q_dropped, round(q_cpu, 2), round(q_time, 3)))
        print(f"[QUEUED]   {rate}Hz -> dropped={q_dropped}, avg_cpu={q_cpu:.2f}%")

    with open("burst_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["architecture", "frequency_hz", "dropped_packets", "avg_cpu_percent", "wall_time_s"]
        )
        writer.writerows(results)

    make_plot(results)


def make_plot(results):
    freqs = FREQUENCIES_HZ
    baseline = {r[1]: r for r in results if r[0] == "baseline"}
    queued = {r[1]: r for r in results if r[0] == "queued"}

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.set_xlabel("Input Frequency (Hz)")
    ax1.set_ylabel("Dropped Packets")
    ax1.plot(
        freqs,
        [baseline[f][2] for f in freqs],
        "o-",
        color="tab:red",
        label="Baseline - Dropped Packets",
    )
    ax1.plot(
        freqs,
        [queued[f][2] for f in freqs],
        "o--",
        color="tab:orange",
        label="Queued - Dropped Packets",
    )
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.set_ylabel("CPU Utilization (%)")
    ax2.plot(
        freqs, [baseline[f][3] for f in freqs], "s-", color="tab:blue", label="Baseline - CPU %"
    )
    ax2.plot(freqs, [queued[f][3] for f in freqs], "s--", color="tab:cyan", label="Queued - CPU %")
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Ingestion Architecture Stress Test: Baseline vs Async Queue")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("burst_results.png", dpi=150)
    print("\nSaved burst_results.csv and burst_results.png")


if __name__ == "__main__":
    asyncio.run(main())
