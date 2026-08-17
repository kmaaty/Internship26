import asyncio
from bleak import BleakScanner


def ble_packet(device, advertisement_data):
    print(
        f"{device.name or 'Unknown'} | {device.address} | "
        f"UUIDs: {advertisement_data.service_uuids} | "
        f"RSSI: {advertisement_data.rssi}"
    )


async def main():
    scanner = BleakScanner(detection_callback=ble_packet)
    await scanner.start()
    print("Listening... Press Ctrl+C to stop.")
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
