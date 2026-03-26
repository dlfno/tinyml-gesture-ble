# capture.py — Captura IMU via BLE
import asyncio
import os
import struct
import sys

from bleak import BleakClient, BleakScanner

# ——— Configuración ———
DEVICE_NAME = "IMU-Capture"
SUJETO      = "sujeto_01"   # cambia antes de cada sesión
CLASE       = "circulo"     # cambia antes de cada clase
OUTPUT_DIR  = f"data/raw/{SUJETO}"

# UUIDs (deben coincidir con el sketch)
SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
CMD_UUID     = "19b10001-e8f2-537e-4f6c-d104768a1214"
STATUS_UUID  = "19b10002-e8f2-537e-4f6c-d104768a1214"
DATA_UUID    = "19b10003-e8f2-537e-4f6c-d104768a1214"

os.makedirs(OUTPUT_DIR, exist_ok=True)

buffer    = []
capturing = False


def next_filename():
    i = 1
    while True:
        fname = os.path.join(OUTPUT_DIR, f"{CLASE}_{i:02d}.csv")
        if not os.path.exists(fname):
            return fname
        i += 1


FRAME_SIZE = 12  # FIFO headerless: gyro(6) + accel(6)
ACC_SCALE  = 8192.0
GYRO_SCALE = 16.384

def on_data(sender, data: bytearray):
    global capturing
    if not capturing:
        return
    n_frames = len(data) // FRAME_SIZE
    for i in range(n_frames):
        chunk = data[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        gx, gy, gz, ax, ay, az = struct.unpack('<6h', chunk)
        ts = len(buffer) * 10
        buffer.append(
            f"{ts},{ax/ACC_SCALE:.4f},{ay/ACC_SCALE:.4f},{az/ACC_SCALE:.4f},"
            f"{gx/GYRO_SCALE:.2f},{gy/GYRO_SCALE:.2f},{gz/GYRO_SCALE:.2f}")


def on_status(sender, data: bytearray):
    global capturing
    msg = data.decode('utf-8', errors='ignore').strip('\x00')
    print(f"[Estado] {msg}")

    if msg == "START":
        capturing = True
        buffer.clear()
        print("⏺  Grabando...")
    elif msg == "STOP":
        capturing = False
        if buffer:
            fname = next_filename()
            with open(fname, "w") as f:
                f.write("timestamp_ms,ax,ay,az,gx,gy,gz\n")
                f.write("\n".join(buffer) + "\n")
            print(f"✅ Guardado: {fname}  ({len(buffer)} muestras)")
            print("Listo para el siguiente clip. Escribe 's' para iniciar\n")
        buffer.clear()


async def main():
    print(f"Buscando dispositivo BLE '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    if not device:
        print("ERROR: No se encontró el dispositivo. Verifica que el Arduino esté encendido.")
        sys.exit(1)

    print(f"Encontrado: {device.name} ({device.address})")
    print("Conectando...")

    async with BleakClient(device) as client:
        print(f"Conectado via BLE.  MTU={client.mtu_size}")
        print("Comandos: 's' = start, 'x' = stop, 'q' = salir\n")

        await client.start_notify(DATA_UUID, on_data)
        await client.start_notify(STATUS_UUID, on_status)

        loop = asyncio.get_event_loop()
        while True:
            cmd = await loop.run_in_executor(None, sys.stdin.readline)
            cmd = cmd.strip()
            if cmd == 's':
                await client.write_gatt_char(CMD_UUID, b's')
            elif cmd == 'x':
                await client.write_gatt_char(CMD_UUID, b'x')
            elif cmd == 'q':
                break

    print("\nCaptura terminada.")


if __name__ == "__main__":
    asyncio.run(main())
