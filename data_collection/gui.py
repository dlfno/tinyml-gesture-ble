#!/usr/bin/env python3
"""
Interfaz gráfica de captura de datos IMU via BLE
Arduino Nano 33 BLE Sense REV2
"""

import tkinter as tk
from tkinter import ttk, messagebox
import asyncio
import threading
import os
import struct

from bleak import BleakClient, BleakScanner

# ── BLE UUIDs ────────────────────────────────────────
SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
CMD_UUID     = "19b10001-e8f2-537e-4f6c-d104768a1214"
STATUS_UUID  = "19b10002-e8f2-537e-4f6c-d104768a1214"
DATA_UUID    = "19b10003-e8f2-537e-4f6c-d104768a1214"
DEVICE_NAME  = "IMU-Capture"

# ── Configuración ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# ── Paleta de colores ──────────────────────────────────
BG        = "#1e1e2e"
SURFACE   = "#2a2a3e"
OVERLAY   = "#313244"
ACCENT    = "#89b4fa"
GREEN     = "#a6e3a1"
RED       = "#f38ba8"
YELLOW    = "#f9e2af"
TEXT      = "#cdd6f4"
SUBTEXT   = "#7f849c"
FONT_BODY = ("SF Pro Display", 13)
FONT_SM   = ("SF Pro Display", 11)
FONT_LG   = ("SF Pro Display", 16, "bold")
FONT_MONO = ("SF Mono", 11)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Captura IMU — BLE")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.client        = None
        self.capturing     = False
        self.buffer        = []
        self.running       = True
        self.timer_job     = None
        self.ble_connected = False

        # Asyncio loop en hilo separado
        self.ble_loop = asyncio.new_event_loop()
        self.ble_thread = threading.Thread(
            target=self._run_ble_loop, daemon=True)
        self.ble_thread.start()

        self._build_ui()
        self.after(300, self._connect)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _run_ble_loop(self):
        asyncio.set_event_loop(self.ble_loop)
        self.ble_loop.run_forever()

    def _ble_submit(self, coro):
        """Envía una coroutine al loop BLE."""
        return asyncio.run_coroutine_threadsafe(coro, self.ble_loop)

    # ── UI ────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=20, pady=8)

        # Título
        tk.Label(self, text="Captura IMU — BLE", font=FONT_LG,
                 bg=BG, fg=ACCENT).grid(row=0, column=0, columnspan=2,
                                         pady=(20, 4))

        # Estado conexión
        conn_frame = tk.Frame(self, bg=SURFACE, padx=12, pady=8)
        conn_frame.grid(row=1, column=0, columnspan=2, sticky="ew",
                        padx=20, pady=4)
        self.conn_dot = tk.Label(conn_frame, text="●", font=FONT_BODY,
                                  bg=SURFACE, fg=RED)
        self.conn_dot.pack(side="left")
        self.conn_label = tk.Label(conn_frame, text="Desconectado",
                                    font=FONT_SM, bg=SURFACE, fg=SUBTEXT)
        self.conn_label.pack(side="left", padx=6)
        tk.Button(conn_frame, text="Reconectar", font=FONT_SM,
                  bg=OVERLAY, fg=TEXT, relief="flat", cursor="hand2",
                  command=self._connect).pack(side="right")

        # ── Sujeto ───────────────────────────────────────
        self._section("Sujeto", 2)
        suj_frame = tk.Frame(self, bg=BG)
        suj_frame.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        self.suj_var = tk.StringVar()
        self.suj_cb  = ttk.Combobox(suj_frame, textvariable=self.suj_var,
                                     font=FONT_BODY, width=22, state="readonly")
        self.suj_cb.pack(side="left")
        self.suj_cb.bind("<<ComboboxSelected>>", lambda _: self._on_subject_change())

        tk.Button(suj_frame, text="+ Nuevo", font=FONT_SM,
                  bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                  command=self._new_subject).pack(side="left", padx=10)

        # ── Clase ────────────────────────────────────────
        self._section("Clase / Gesto", 4)
        cls_frame = tk.Frame(self, bg=BG)
        cls_frame.grid(row=5, column=0, columnspan=2, sticky="ew", **pad)

        self.cls_var = tk.StringVar()
        self.cls_cb  = ttk.Combobox(cls_frame, textvariable=self.cls_var,
                                     font=FONT_BODY, width=22)
        self.cls_cb.pack(side="left")
        self.cls_cb.bind("<<ComboboxSelected>>", lambda _: self._update_preview())
        self.cls_cb.bind("<KeyRelease>", lambda _: self._update_preview())

        tk.Button(cls_frame, text="+ Nueva", font=FONT_SM,
                  bg=OVERLAY, fg=TEXT, relief="flat", cursor="hand2",
                  command=self._new_class).pack(side="left", padx=10)

        # ── Registro ─────────────────────────────────────
        self._section("Registro", 6)
        reg_frame = tk.Frame(self, bg=BG)
        reg_frame.grid(row=7, column=0, columnspan=2, sticky="ew", **pad)

        self.reg_mode = tk.StringVar(value="nuevo")
        rb_nuevo = tk.Radiobutton(reg_frame, text="Nuevo (auto)", font=FONT_BODY,
                                   variable=self.reg_mode, value="nuevo",
                                   bg=BG, fg=TEXT, selectcolor=OVERLAY,
                                   activebackground=BG, activeforeground=TEXT,
                                   command=self._update_preview)
        rb_nuevo.pack(side="left")
        rb_sobre = tk.Radiobutton(reg_frame, text="Sobrescribir:", font=FONT_BODY,
                                   variable=self.reg_mode, value="sobrescribir",
                                   bg=BG, fg=TEXT, selectcolor=OVERLAY,
                                   activebackground=BG, activeforeground=TEXT,
                                   command=self._update_preview)
        rb_sobre.pack(side="left", padx=(16, 4))

        self.over_var = tk.StringVar()
        self.over_cb  = ttk.Combobox(reg_frame, textvariable=self.over_var,
                                      font=FONT_SM, width=16, state="readonly")
        self.over_cb.pack(side="left")
        self.over_cb.bind("<<ComboboxSelected>>", lambda _: self._update_preview())

        # ── Modo de captura ──────────────────────────────
        self._section("Modo de captura", 8)
        mode_frame = tk.Frame(self, bg=BG)
        mode_frame.grid(row=9, column=0, columnspan=2, sticky="ew", **pad)

        self.mode_var = tk.StringVar(value="manual")
        tk.Radiobutton(mode_frame, text="Manual", font=FONT_BODY,
                       variable=self.mode_var, value="manual",
                       bg=BG, fg=TEXT, selectcolor=OVERLAY,
                       activebackground=BG, activeforeground=TEXT,
                       command=self._on_mode_change).pack(side="left")
        tk.Radiobutton(mode_frame, text="Timer", font=FONT_BODY,
                       variable=self.mode_var, value="timer",
                       bg=BG, fg=TEXT, selectcolor=OVERLAY,
                       activebackground=BG, activeforeground=TEXT,
                       command=self._on_mode_change).pack(side="left", padx=(16, 6))

        self.timer_entry = tk.Entry(mode_frame, font=FONT_BODY, width=4,
                                     bg=OVERLAY, fg=TEXT, insertbackground=TEXT,
                                     relief="flat", justify="center")
        self.timer_entry.insert(0, "5")
        self.timer_entry.pack(side="left")
        self.timer_lbl = tk.Label(mode_frame, text="seg", font=FONT_SM,
                                   bg=BG, fg=SUBTEXT)
        self.timer_lbl.pack(side="left", padx=4)
        self.timer_entry.config(state="disabled")
        self.timer_lbl.config(fg=OVERLAY)

        # ── Preview del archivo ──────────────────────────
        prev_frame = tk.Frame(self, bg=OVERLAY, padx=12, pady=8)
        prev_frame.grid(row=10, column=0, columnspan=2, sticky="ew",
                        padx=20, pady=(4, 8))
        tk.Label(prev_frame, text="Guardará en:", font=FONT_SM,
                 bg=OVERLAY, fg=SUBTEXT).pack(side="left")
        self.preview_lbl = tk.Label(prev_frame, text="—", font=FONT_MONO,
                                     bg=OVERLAY, fg=YELLOW)
        self.preview_lbl.pack(side="left", padx=8)

        # ── Barra de progreso (timer) ────────────────────
        self.progress_frame = tk.Frame(self, bg=BG)
        self.progress_frame.grid(row=11, column=0, columnspan=2,
                                  sticky="ew", padx=20, pady=0)
        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Accent.Horizontal.TProgressbar",
                         troughcolor=OVERLAY, background=ACCENT,
                         bordercolor=OVERLAY, lightcolor=ACCENT, darkcolor=ACCENT)
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=360,
                                             variable=self.progress_var,
                                             style="Accent.Horizontal.TProgressbar")
        self.progress_bar.pack(side="left")
        self.progress_lbl = tk.Label(self.progress_frame, text="",
                                      font=FONT_SM, bg=BG, fg=SUBTEXT)
        self.progress_lbl.pack(side="left", padx=8)
        self.progress_frame.grid_remove()

        # ── Botones principales ──────────────────────────
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.grid(row=12, column=0, columnspan=2, pady=12)

        self.start_btn = tk.Button(btn_frame, text="▶  INICIAR",
                                    font=("SF Pro Display", 14, "bold"),
                                    bg=GREEN, fg=BG, relief="flat",
                                    cursor="hand2", width=14, pady=8,
                                    command=self._iniciar)
        self.start_btn.pack(side="left", padx=8)

        self.stop_btn = tk.Button(btn_frame, text="■  DETENER",
                                   font=("SF Pro Display", 14, "bold"),
                                   bg=RED, fg=BG, relief="flat",
                                   cursor="hand2", width=14, pady=8,
                                   state="disabled",
                                   command=self._detener)
        self.stop_btn.pack(side="left", padx=8)

        # ── Log ──────────────────────────────────────────
        self._section("Log", 13)
        log_frame = tk.Frame(self, bg=OVERLAY, padx=2, pady=2)
        log_frame.grid(row=14, column=0, columnspan=2,
                        padx=20, pady=(0, 20))
        self.log = tk.Text(log_frame, height=8, width=54, font=FONT_MONO,
                            bg=OVERLAY, fg=TEXT, insertbackground=TEXT,
                            relief="flat", state="disabled", wrap="word")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left")
        scroll.pack(side="left", fill="y")

        self._refresh_subjects()

    def _section(self, title, row):
        tk.Label(self, text=title.upper(), font=("SF Pro Display", 10, "bold"),
                 bg=BG, fg=SUBTEXT).grid(row=row, column=0, columnspan=2,
                                          sticky="w", padx=20, pady=(10, 0))

    # ── BLE ─────────────────────────────────────────────
    def _connect(self):
        self._log("Buscando IMU-Capture por BLE...")
        self._set_conn(False)
        self._ble_submit(self._ble_connect())

    async def _ble_connect(self):
        try:
            if self.client and self.client.is_connected:
                await self.client.disconnect()

            device = await BleakScanner.find_device_by_name(
                DEVICE_NAME, timeout=10.0)
            if not device:
                self.after(0, lambda: self._log("No se encontró IMU-Capture"))
                return

            self.client = BleakClient(
                device, disconnected_callback=self._on_ble_disconnect)
            await self.client.connect()
            self.ble_connected = True

            await self.client.start_notify(DATA_UUID, self._on_ble_data)
            await self.client.start_notify(STATUS_UUID, self._on_ble_status)

            mtu = self.client.mtu_size
            self.after(0, lambda: self._set_conn(True))
            self.after(0, lambda: self._log(
                f"Conectado BLE: {device.name} ({device.address})  MTU={mtu}"))

        except Exception as e:
            self.after(0, lambda: self._log(f"Error BLE: {e}"))
            self.after(0, lambda: self._set_conn(False))

    def _on_ble_disconnect(self, client):
        self.ble_connected = False
        self.after(0, lambda: self._set_conn(False))
        self.after(0, lambda: self._log("BLE desconectado"))
        if self.capturing:
            self.capturing = False
            self.after(0, self._save_file)

    def _on_ble_data(self, sender, data: bytearray):
        if not self.capturing:
            return
        # FIFO headerless: gyro(6) + accel(6) = 12 bytes por frame
        FRAME_SIZE = 12
        # BMI270 raw scales: accel ±4g = /8192, gyro ±2000dps = /16.384
        ACC_SCALE  = 8192.0
        GYRO_SCALE = 16.384
        n_frames = len(data) // FRAME_SIZE
        for i in range(n_frames):
            chunk = data[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
            gx, gy, gz, ax, ay, az = struct.unpack('<6h', chunk)
            ts = len(self.buffer) * 10  # reconstruir timestamp a 100 Hz
            self.buffer.append(
                f"{ts},{ax/ACC_SCALE:.4f},{ay/ACC_SCALE:.4f},{az/ACC_SCALE:.4f},"
                f"{gx/GYRO_SCALE:.2f},{gy/GYRO_SCALE:.2f},{gz/GYRO_SCALE:.2f}")

    def _on_ble_status(self, sender, data: bytearray):
        msg = data.decode('utf-8', errors='ignore').strip('\x00')
        self.after(0, lambda: self._log(f"[Arduino] {msg}"))

        if msg == "START":
            self.capturing = True
            self.buffer = []
            self.after(0, lambda: self._log("⏺  Grabando..."))
        elif msg == "STOP":
            self.capturing = False
            self.after(0, self._save_file)

    async def _ble_send(self, cmd: bytes):
        try:
            if self.client and self.client.is_connected:
                await self.client.write_gatt_char(CMD_UUID, cmd)
        except Exception as e:
            self.after(0, lambda: self._log(f"Error al enviar: {e}"))

    def _send(self, cmd: bytes):
        self._ble_submit(self._ble_send(cmd))

    def _set_conn(self, ok):
        if ok:
            self.conn_dot.config(fg=GREEN)
            self.conn_label.config(text="Conectado BLE", fg=GREEN)
        else:
            self.conn_dot.config(fg=RED)
            self.conn_label.config(text="Desconectado", fg=SUBTEXT)

    # ── Sujeto / Clase ───────────────────────────────────
    def _refresh_subjects(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        subjects = sorted([
            d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))
        ])
        self.suj_cb["values"] = subjects
        if subjects and not self.suj_var.get():
            self.suj_var.set(subjects[0])
            self._on_subject_change()

    def _on_subject_change(self):
        self._refresh_classes()
        self._update_preview()

    def _refresh_classes(self):
        sujeto = self.suj_var.get()
        if not sujeto:
            return
        folder = os.path.join(DATA_DIR, sujeto)
        clases = sorted(set(
            f.rsplit("_", 1)[0]
            for f in os.listdir(folder) if f.endswith(".csv")
        )) if os.path.isdir(folder) else []
        self.cls_cb["values"] = clases
        self._update_overwrite_list()

    def _update_overwrite_list(self):
        sujeto = self.suj_var.get()
        clase  = self.cls_var.get()
        if not sujeto or not clase:
            self.over_cb["values"] = []
            return
        folder = os.path.join(DATA_DIR, sujeto)
        files  = sorted(f for f in os.listdir(folder)
                        if f.startswith(clase + "_") and f.endswith(".csv")) \
                 if os.path.isdir(folder) else []
        self.over_cb["values"] = files
        if files:
            self.over_cb.current(len(files) - 1)

    def _next_filename(self):
        sujeto = self.suj_var.get()
        clase  = self.cls_var.get()
        folder = os.path.join(DATA_DIR, sujeto)
        i = 1
        while True:
            f = os.path.join(folder, f"{clase}_{i:02d}.csv")
            if not os.path.exists(f):
                return f
            i += 1

    def _target_filename(self):
        if self.reg_mode.get() == "nuevo":
            return self._next_filename()
        else:
            fname = self.over_var.get()
            if not fname:
                return self._next_filename()
            return os.path.join(DATA_DIR, self.suj_var.get(), fname)

    def _update_preview(self):
        self._update_overwrite_list()
        sujeto = self.suj_var.get()
        clase  = self.cls_var.get()
        if not sujeto or not clase:
            self.preview_lbl.config(text="—")
            return
        path = self._target_filename()
        short = os.path.relpath(path, BASE_DIR)
        self.preview_lbl.config(text=short)

    def _new_subject(self):
        dialog = _InputDialog(self, "Nuevo sujeto", "Nombre del sujeto:")
        name = dialog.result
        if name:
            folder = os.path.join(DATA_DIR, name)
            os.makedirs(folder, exist_ok=True)
            self._refresh_subjects()
            self.suj_var.set(name)
            self._on_subject_change()

    def _new_class(self):
        dialog = _InputDialog(self, "Nueva clase", "Nombre de la clase / gesto:")
        name = dialog.result
        if name:
            self.cls_var.set(name)
            self._update_preview()

    # ── Captura ──────────────────────────────────────────
    def _iniciar(self):
        if not self.suj_var.get():
            messagebox.showwarning("Falta sujeto", "Selecciona o crea un sujeto.")
            return
        if not self.cls_var.get():
            messagebox.showwarning("Falta clase", "Selecciona o escribe una clase.")
            return
        if not self.ble_connected:
            messagebox.showerror("Sin conexión", "El Arduino no está conectado por BLE.")
            return

        self.start_btn.config(state="disabled")
        self._send(b's')

        if self.mode_var.get() == "timer":
            try:
                secs = float(self.timer_entry.get())
            except ValueError:
                secs = 5.0
            self.progress_frame.grid()
            self._run_timer(secs)
        else:
            self.stop_btn.config(state="normal")

    def _detener(self):
        self._send(b'x')
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")

    def _run_timer(self, total):
        import time
        start = time.time()

        def tick():
            elapsed = time.time() - start
            remaining = total - elapsed
            pct = min(elapsed / total * 100, 100)
            self.progress_var.set(pct)
            self.progress_lbl.config(
                text=f"{elapsed:.1f}s / {total:.0f}s")
            if remaining > 0:
                self.timer_job = self.after(50, tick)
            else:
                self._send(b'x')
                self.progress_var.set(0)
                self.progress_lbl.config(text="")
                self.progress_frame.grid_remove()
                self.start_btn.config(state="normal")

        tick()

    def _save_file(self):
        if not self.buffer:
            self._log("Sin datos — archivo no guardado.")
            self.start_btn.config(state="normal")
            return

        sujeto = self.suj_var.get()
        folder = os.path.join(DATA_DIR, sujeto)
        os.makedirs(folder, exist_ok=True)

        path = self._target_filename()
        with open(path, "w") as f:
            f.write("timestamp_ms,ax,ay,az,gx,gy,gz\n")
            f.write("\n".join(self.buffer) + "\n")

        short = os.path.relpath(path, BASE_DIR)
        self._log(f"✅  {short}  ({len(self.buffer)} muestras)")
        self.buffer = []
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._refresh_classes()
        self._update_preview()

    # ── Utilidades ───────────────────────────────────────
    def _on_mode_change(self):
        if self.mode_var.get() == "timer":
            self.timer_entry.config(state="normal")
            self.timer_lbl.config(fg=SUBTEXT)
            self.stop_btn.config(state="disabled")
        else:
            self.timer_entry.config(state="disabled")
            self.timer_lbl.config(fg=OVERLAY)

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _on_close(self):
        self.running = False
        if self.timer_job:
            self.after_cancel(self.timer_job)
        if self.client and self.client.is_connected:
            self._ble_submit(self.client.disconnect())
        self.ble_loop.call_soon_threadsafe(self.ble_loop.stop)
        self.destroy()


class _InputDialog(tk.Toplevel):
    """Diálogo simple de texto."""
    def __init__(self, parent, title, prompt):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=BG)
        self.resizable(False, False)
        self.result = None
        self.grab_set()

        tk.Label(self, text=prompt, font=FONT_BODY,
                 bg=BG, fg=TEXT).pack(padx=20, pady=(16, 4))
        self.entry = tk.Entry(self, font=FONT_BODY, bg=OVERLAY, fg=TEXT,
                               insertbackground=TEXT, relief="flat", width=24)
        self.entry.pack(padx=20, pady=4)
        self.entry.focus()
        self.entry.bind("<Return>", lambda _: self._ok())

        btn_f = tk.Frame(self, bg=BG)
        btn_f.pack(pady=12)
        tk.Button(btn_f, text="Cancelar", font=FONT_SM,
                  bg=OVERLAY, fg=TEXT, relief="flat",
                  command=self.destroy).pack(side="left", padx=6)
        tk.Button(btn_f, text="Crear", font=FONT_SM,
                  bg=ACCENT, fg=BG, relief="flat",
                  command=self._ok).pack(side="left", padx=6)

        self.wait_window()

    def _ok(self):
        val = self.entry.get().strip()
        if val:
            self.result = val
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
