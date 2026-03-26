#include <Arduino_BMI270_BMM150.h>
#include <ArduinoBLE.h>
#include <Wire.h>

// ——— Configuración ———
const unsigned long CLIP_MS = 10000;
const int BTN_PIN = 2;

// ——— BMI270 FIFO vía I2C directo ———
#define BMI_ADDR     0x68
#define BMI_WIRE     Wire1
#define REG_FIFO_LEN 0x24
#define REG_FIFO_DAT 0x26
#define REG_FIFO_CF0 0x48
#define REG_FIFO_CF1 0x49
#define REG_CMD      0x7E

const int FRAME_SIZE = 12;  // gyro(6) + accel(6) = 12 bytes

void bmiWrite(uint8_t reg, uint8_t val) {
  BMI_WIRE.beginTransmission(BMI_ADDR);
  BMI_WIRE.write(reg);
  BMI_WIRE.write(val);
  BMI_WIRE.endTransmission();
}

void setupFIFO() {
  bmiWrite(REG_CMD, 0xB0);
  delay(2);
  bmiWrite(REG_FIFO_CF0, 0x00);
  bmiWrite(REG_FIFO_CF1, 0x70);
}

void flushFIFO() {
  bmiWrite(REG_CMD, 0xB0);
}

int readFIFOLen() {
  BMI_WIRE.beginTransmission(BMI_ADDR);
  BMI_WIRE.write(REG_FIFO_LEN);
  BMI_WIRE.endTransmission(false);
  BMI_WIRE.requestFrom((uint8_t)BMI_ADDR, (uint8_t)2);
  uint8_t lo = BMI_WIRE.read();
  uint8_t hi = BMI_WIRE.read();
  return ((hi & 0x3F) << 8) | lo;
}

// ——— BLE ———
BLEService imuService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic cmdChar("19B10001-E8F2-537E-4F6C-D104768A1214",
                          BLEWrite, 1);
BLECharacteristic statusChar("19B10002-E8F2-537E-4F6C-D104768A1214",
                              BLERead | BLENotify, 20);
BLECharacteristic dataChar("19B10003-E8F2-537E-4F6C-D104768A1214",
                            BLENotify, 244);

bool bleConnected = false;
bool recording = false;
unsigned long clipStart = 0;
unsigned long lastBLEPoll = 0; // [arduino-optimizer] 2026-03-17 — cadenciar BLE.poll() para reducir overhead en fallback serial

// ——— Buffer de acumulación ———
// Acumula frames del FIFO en RAM antes de enviar por BLE.
// Esto desacopla la lectura del FIFO (frecuente) del envío BLE (bloqueante).
uint8_t accBuf[240];  // max 20 frames
int accFrames = 0;
unsigned long lastBLESend = 0;

// Lee frames disponibles del FIFO hacia accBuf
void drainFIFO() {
  int fifoLen = readFIFOLen();
  int available = fifoLen / FRAME_SIZE;
  if (available <= 0 || accFrames >= 20) return;

  int toRead = min(available, 20 - accFrames);
  int bytes = toRead * FRAME_SIZE;

  BMI_WIRE.beginTransmission(BMI_ADDR);
  BMI_WIRE.write(REG_FIFO_DAT);
  BMI_WIRE.endTransmission(false);

  int pos = accFrames * FRAME_SIZE;
  int end = pos + bytes;
  while (pos < end) {
    int chunk = min(end - pos, 240);
    BMI_WIRE.requestFrom((uint8_t)BMI_ADDR, (uint8_t)chunk);
    for (int i = 0; i < chunk && BMI_WIRE.available(); i++) {
      accBuf[pos++] = BMI_WIRE.read();
    }
  }
  accFrames += toRead;
}

// Envía accBuf por BLE y resetea
void sendAccBuf() {
  if (accFrames > 0) {
    dataChar.writeValue(accBuf, accFrames * FRAME_SIZE);
    accFrames = 0;
    lastBLESend = millis();
  }
}

void sendStatus(const char* msg) {
  statusChar.writeValue((const uint8_t*)msg, strlen(msg));
  Serial.print("# ");
  Serial.println(msg);
}

void startRecording() {
  flushFIFO();
  accFrames = 0;
  lastBLESend = millis();
  clipStart = millis();
  recording = true;
  sendStatus("START");
  digitalWrite(LED_BUILTIN, HIGH);
}

void stopRecording() {
  recording = false;

  if (bleConnected) {
    // Drenar FIFO restante al accBuf y enviar
    drainFIFO();
    sendAccBuf();

    // Dar tiempo al stack BLE para notificaciones pendientes
    for (int i = 0; i < 3; i++) {
      BLE.poll();
      delay(20);
    }
  }

  flushFIFO();
  sendStatus("STOP");
  BLE.poll();
  delay(20);
  BLE.poll();
  digitalWrite(LED_BUILTIN, LOW);
}

void onBLEConnected(BLEDevice central) {
  bleConnected = true;
  Serial.print("# BLE conectado: ");
  Serial.println(central.address());
}

void onBLEDisconnected(BLEDevice central) {
  bleConnected = false;
  Serial.println("# BLE desconectado");
  if (recording) {
    recording = false;
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(BTN_PIN, INPUT_PULLUP);

  Serial.println("# Iniciando IMU...");
  if (!IMU.begin()) {
    Serial.println("# ERROR: No se pudo iniciar el IMU");
    while (1);
  }

  BMI_WIRE.setClock(400000);
  setupFIFO();
  delay(10);

  BMI_WIRE.beginTransmission(BMI_ADDR);
  BMI_WIRE.write(REG_FIFO_CF1);
  BMI_WIRE.endTransmission(false);
  BMI_WIRE.requestFrom((uint8_t)BMI_ADDR, (uint8_t)1);
  uint8_t cfg1 = BMI_WIRE.read();
  Serial.print("# FIFO_CONFIG_1 = 0x");
  Serial.println(cfg1, HEX);
  Serial.println(cfg1 == 0x70 ? "# FIFO OK (acc+gyr headerless)" : "# FIFO ERROR!");

  Serial.println("# Iniciando BLE...");
  if (!BLE.begin()) {
    Serial.println("# ERROR: No se pudo iniciar BLE");
    while (1);
  }

  BLE.setConnectionInterval(0x0010, 0x0020); // 20ms – 40ms
  BLE.setLocalName("IMU-Capture");
  BLE.setAdvertisedService(imuService);

  imuService.addCharacteristic(cmdChar);
  imuService.addCharacteristic(statusChar);
  imuService.addCharacteristic(dataChar);
  BLE.addService(imuService);

  BLE.setEventHandler(BLEConnected, onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);

  BLE.advertise();
  sendStatus("READY");
  Serial.println("# LISTO. BLE activo como 'IMU-Capture'");
}

void loop() {
  // [arduino-optimizer] 2026-03-17 — Capturar millis() una sola vez al inicio del loop y
  // reutilizarlo en todos los checks del ciclo (BLE cadencia, debounce boton, auto-stop,
  // BLE send timeout). Evita N llamadas al timer del sistema por iteracion.
  unsigned long ahora = millis(); // [arduino-optimizer] 2026-03-17 — un solo millis() por iteracion

  // [arduino-optimizer] 2026-03-17 — Omitir BLE.poll() durante grabacion serial (sin central).
  // Razon: BLE.poll() sin central conectada bloquea ~7-8 ms por llamada (stack Cordio procesando
  // cola HCI de advertising). Ninguna frecuencia de cadencia resuelve esto: cada llamada destruye
  // uno o mas ciclos IMU de ~10 ms. El advertising es manejado por hardware; BLE.poll() solo es
  // necesario para procesar datos/comandos de una central conectada. Cuando recording==true y
  // bleConnected==false, saltar poll completamente elimina el 100% del overhead BLE durante
  // la captura. Start/stop siguen disponibles via serial y boton fisico.
  // Historial: 5ms=2307m, 15ms=2231m, 50ms/5ms-diferenciado=2268m. Ninguna cadencia supero 2375.
  // [arduino-optimizer] 2026-03-17 — Condicion: solo hacer BLE.poll() si hay central conectada
  //   O si no estamos grabando (para detectar nuevas conexiones entre sesiones).
  if (!recording || bleConnected) { // [arduino-optimizer] 2026-03-17 — skip poll durante captura serial
    unsigned long intervaloBLE = bleConnected ? 5 : 50;
    if (ahora - lastBLEPoll >= intervaloBLE) { // [arduino-optimizer] 2026-03-17 — reusar ahora en vez de millis()
      lastBLEPoll = ahora;
      BLE.poll();
      if (cmdChar.written()) {
        uint8_t cmd;
        cmdChar.readValue(cmd);
        if (cmd == 's' && !recording) startRecording();
        else if (cmd == 'x' && recording) stopRecording();
      }
    }
  }

  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 's' && !recording) startRecording();
    else if (cmd == 'x' && recording) stopRecording();
  }

  static bool lastBtn = HIGH;
  static unsigned long lastDebounce = 0;
  bool currentBtn = digitalRead(BTN_PIN);
  if (currentBtn == LOW && lastBtn == HIGH && ahora - lastDebounce > 200) { // [arduino-optimizer] 2026-03-17 — reusar ahora
    lastDebounce = ahora; // [arduino-optimizer] 2026-03-17 — reusar ahora
    if (!recording) startRecording();
    else            stopRecording();
  }
  lastBtn = currentBtn;

  if (recording && ahora - clipStart >= CLIP_MS) { // [arduino-optimizer] 2026-03-17 — reusar ahora
    stopRecording();
    return;
  }

  // ——— Acumulación + envío por BLE ———
  // 1) Drenar FIFO al buffer de acumulación (rápido, sin BLE)
  // 2) Enviar cuando hay 3+ frames O pasaron 25ms (lo que pase primero)
  //
  // Esto evita el problema de writeValue bloqueante:
  //   - writeValue bloquea ~15ms (1 connection event)
  //   - Durante ese bloqueo, el FIFO acumula frames
  //   - Al desbloquearse, drainFIFO lee todo lo pendiente
  //   - Cuando accBuf tiene 3+ frames, un solo writeValue los envía todos
  //   - 3 frames por event × 67 events/s ≈ 100 Hz
  // [arduino-optimizer] 2026-03-17 — drainFIFO() solo cuando bleConnected==true.
  // Razon: sin BLE, las 2 transacciones I2C de drainFIFO (~0.3 ms cada una por FIFO vacio)
  // se ejecutaban en cada iteracion del loop, sumando overhead innecesario al fallback serial.
  if (recording && bleConnected) {
    drainFIFO();

    bool enoughFrames = (accFrames >= 3);
    bool timeout = (ahora - lastBLESend >= 25); // [arduino-optimizer] 2026-03-17 — reusar ahora

    if (enoughFrames || (accFrames > 0 && timeout)) {
      sendAccBuf();
    }
  }

  // Serial fallback
  // [arduino-optimizer] 2026-03-17 — calcular millis() una sola vez y reutilizar para
  // timestamp y auto-stop, evitando llamadas redundantes al timer del sistema.
  if (recording && !bleConnected && IMU.accelerationAvailable()) {
    float ax, ay, az, gx, gy, gz;
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    uint32_t ts = ahora - clipStart; // [arduino-optimizer] 2026-03-17 — reusar ahora en vez de llamar millis() otra vez
    char buf[80];
    int len = snprintf(buf, sizeof(buf) - 1, "%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                       ts, ax, ay, az, gx, gy, gz);
    // [arduino-optimizer] 2026-03-17 — usar Serial.write() no-blocking en lugar de Serial.println()
    // bloqueante. Serial.availableForWrite() retorna bytes libres en el buffer USB CDC (512 bytes).
    // Si no hay espacio suficiente, se descarta la muestra en lugar de bloquear el loop ~0.5-1 ms.
    // Esto elimina los stalls periodicos del flush CDC que ocurren cada ~12 muestras.
    if (len > 0 && Serial.availableForWrite() >= len) {
      Serial.write(buf, len); // [arduino-optimizer] 2026-03-17 — write() no bloquea si hay espacio; println() bloquea siempre en flush
    }
  }
}
