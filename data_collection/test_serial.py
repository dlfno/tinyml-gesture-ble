# test_ble.py — Prueba rápida de conexión BLE al Arduino
import asyncio
import struct
from bleak import BleakClient, BleakScanner

DEVICE_NAME = "IMU-Capture"
SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
CMD_UUID     = "19b10001-e8f2-537e-4f6c-d104768a1214"
STATUS_UUID  = "19b10002-e8f2-537e-4f6c-d104768a1214"
DATA_UUID    = "19b10003-e8f2-537e-4f6c-d104768a1214"

samples = []


FRAME_SIZE = 12
ACC_SCALE  = 8192.0
GYRO_SCALE = 16.384

def on_data(sender, data: bytearray):
    n_frames = len(data) // FRAME_SIZE
    for i in range(n_frames):
        chunk = data[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        gx, gy, gz, ax, ay, az = struct.unpack('<6h', chunk)
        ts = len(samples) * 10
        samples.append((ts, ax/ACC_SCALE, ay/ACC_SCALE, az/ACC_SCALE,
                         gx/GYRO_SCALE, gy/GYRO_SCALE, gz/GYRO_SCALE))
        print(f"  {ts:6d} ms | ax={ax/ACC_SCALE:+.2f} ay={ay/ACC_SCALE:+.2f} "
              f"az={az/ACC_SCALE:+.2f} | gx={gx/GYRO_SCALE:+.1f} "
              f"gy={gy/GYRO_SCALE:+.1f} gz={gz/GYRO_SCALE:+.1f}")


def on_status(sender, data: bytearray):
    msg = data.decode('utf-8', errors='ignore').strip('\x00')
    print(f"[Estado] {msg}")


async def main():
    print(f"Buscando '{DEVICE_NAME}' por BLE...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    if not device:
        print("ERROR: No se encontró el dispositivo.")
        return

    print(f"Encontrado: {device.name} ({device.address})")

    async with BleakClient(device) as client:
        print("Conectado.\n")

        await client.start_notify(DATA_UUID, on_data)
        await client.start_notify(STATUS_UUID, on_status)

        print("Enviando 's' (start)...")
        await client.write_gatt_char(CMD_UUID, b's')

        await asyncio.sleep(3)

        print(f"\nRecibidas {len(samples)} muestras en 3s")

        print("\nEnviando 'x' (stop)...")
        await client.write_gatt_char(CMD_UUID, b'x')

        await asyncio.sleep(1)

    print("Desconectado. Test completado.")


if __name__ == "__main__":
    asyncio.run(main())
