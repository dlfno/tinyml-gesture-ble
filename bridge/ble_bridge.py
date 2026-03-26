"""
TinyML BLE Bridge v1.1
Scans for the Arduino by SERVICE UUID (most reliable on macOS CoreBluetooth),
falls back to device name. Forwards JSON inference payloads to WebSocket clients.

Run: python bridge/ble_bridge.py
"""

import asyncio
import json
from datetime import datetime

from bleak import BleakClient, BleakScanner
import websockets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVICE_UUID        = "19b10000-e8f2-537e-4f6c-d104768a1214"
INFERENCE_CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"

DEVICE_NAME     = "TinyML-Sense"
WS_HOST         = "localhost"
WS_PORT         = 8765
RECONNECT_DELAY = 3  # seconds between BLE reconnect attempts

# ---------------------------------------------------------------------------
# WebSocket client registry
# ---------------------------------------------------------------------------
connected_clients: set = set()


async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"[WS]   Cliente conectado. Total: {len(connected_clients)}")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"[WS]   Cliente desconectado. Total: {len(connected_clients)}")


async def broadcast(message: str):
    if not connected_clients:
        return
    results = await asyncio.gather(
        *[client.send(message) for client in connected_clients],
        return_exceptions=True,
    )
    for client, result in zip(list(connected_clients), results):
        if isinstance(result, Exception):
            connected_clients.discard(client)


# ---------------------------------------------------------------------------
# BLE notification handler
# ---------------------------------------------------------------------------
def make_notification_handler(loop: asyncio.AbstractEventLoop):
    def handler(sender, data: bytearray):
        try:
            payload = data.decode("utf-8")
            parsed  = json.loads(payload)

            enriched = {
                "status":        "inference",
                "class":         parsed.get("c", "UNKNOWN"),
                "probabilities": parsed.get("p", []),
                "latency_ms":    parsed.get("l", 0),
                "device_time":   parsed.get("t", 0),
                "bridge_time":   datetime.now().isoformat(),
            }

            message = json.dumps(enriched)
            asyncio.run_coroutine_threadsafe(broadcast(message), loop)

            conf = max(parsed.get("p", [0]))
            print(f"[BLE]  {parsed.get('c', '?')} @ {conf:.1f}% | {parsed.get('l', '?')}ms")

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[BLE]  Error decodificando: {e}")

    return handler


# ---------------------------------------------------------------------------
# Device discovery — service UUID first, then name
# On macOS, CoreBluetooth hides MAC addresses; we scan by UUID to be reliable.
# ---------------------------------------------------------------------------
async def find_device():
    """Return the first BLE device that advertises our service UUID, or by name."""
    found_device = None

    def callback(device, adv_data):
        nonlocal found_device
        if found_device:
            return
        uuids = [str(u).lower() for u in (adv_data.service_uuids or [])]
        if SERVICE_UUID in uuids:
            print(f"[SCAN] Encontrado por UUID: {device.name!r} ({device.address})")
            found_device = device
        elif device.name == DEVICE_NAME:
            print(f"[SCAN] Encontrado por nombre: {device.name!r} ({device.address})")
            found_device = device

    print(f"[SCAN] Buscando '{DEVICE_NAME}' o service UUID {SERVICE_UUID[:8]}...")

    async with BleakScanner(callback) as scanner:
        # Wait up to 10 s for the device to appear
        for _ in range(100):
            await asyncio.sleep(0.1)
            if found_device:
                break

    return found_device


# ---------------------------------------------------------------------------
# BLE connection loop with auto-reconnect
# ---------------------------------------------------------------------------
async def ble_loop():
    loop = asyncio.get_running_loop()
    notification_handler = make_notification_handler(loop)

    while True:
        try:
            device = await find_device()

            if not device:
                print(f"[SCAN] No encontrado. Reintentando en {RECONNECT_DELAY}s...")
                await broadcast(json.dumps({"status": "scanning"}))
                await asyncio.sleep(RECONNECT_DELAY)
                continue

            async with BleakClient(device.address) as client:
                print(f"[BLE]  Conectado a {device.address}")
                await broadcast(json.dumps({"status": "connected",
                                            "device": device.name or DEVICE_NAME}))

                await client.start_notify(INFERENCE_CHAR_UUID, notification_handler)

                while client.is_connected:
                    await asyncio.sleep(1.0)

                print("[BLE]  Conexión perdida.")
                await broadcast(json.dumps({"status": "disconnected"}))

        except Exception as e:
            print(f"[BLE]  Error: {e}")
            await broadcast(json.dumps({"status": "error", "message": str(e)}))
            await asyncio.sleep(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    print("=" * 50)
    print("  TinyML BLE Bridge v1.1")
    print(f"  WebSocket: ws://{WS_HOST}:{WS_PORT}")
    print(f"  BLE scan:  service UUID {SERVICE_UUID[:18]}...")
    print("=" * 50)

    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"[WS]   Servidor en ws://{WS_HOST}:{WS_PORT}")

    await ble_loop()

    ws_server.close()
    await ws_server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[BRIDGE] Shutdown limpio.")
