// TinyML Gesture Classification — Arduino Nano 33 BLE Sense Rev2
// Model: Conv1D (float32 I/O), input [1, 100, 6], output [1, 4]
// BLE: GATT Notify, JSON payload
// IMU: BMI270 FIFO raw (mismo pipeline que recoleccion.ino → mismo formato que training data)

#include <Arduino_BMI270_BMM150.h>
#include <ArduinoBLE.h>
#include <Wire.h>
#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model_cnn.h"

// ---------------------------------------------------------------------------
// BLE UUIDs
// ---------------------------------------------------------------------------
#define SERVICE_UUID        "19b10000-e8f2-537e-4f6c-d104768a1214"
#define INFERENCE_CHAR_UUID "19b10001-e8f2-537e-4f6c-d104768a1214"

BLEService inferenceService(SERVICE_UUID);
BLEStringCharacteristic inferenceChar(INFERENCE_CHAR_UUID, BLERead | BLENotify, 128);

// ---------------------------------------------------------------------------
// BMI270 FIFO — lectura directa por I2C (idéntico a recoleccion.ino)
// Evita el swap de ejes que aplica la librería en readAcceleration/readGyroscope.
// Frame headerless: GYR_X(2) GYR_Y(2) GYR_Z(2) ACC_X(2) ACC_Y(2) ACC_Z(2) = 12 bytes
// capture.py guarda: ax=ACC_X/8192, ay=ACC_Y/8192, az=ACC_Z/8192,
//                    gx=GYR_X/16.384, gy=GYR_Y/16.384, gz=GYR_Z/16.384
// ---------------------------------------------------------------------------
#define BMI_ADDR      0x68
#define BMI_WIRE      Wire1
#define REG_FIFO_LEN  0x24
#define REG_FIFO_DAT  0x26
#define REG_FIFO_CF0  0x48
#define REG_FIFO_CF1  0x49
#define REG_CMD       0x7E
#define FRAME_SIZE    12
#define ACC_SCALE     8192.0f    // LSB/g  (±4 g)
#define GYRO_SCALE    16.384f    // LSB/°/s (±2000 dps)

static void bmiWrite(uint8_t reg, uint8_t val) {
    BMI_WIRE.beginTransmission(BMI_ADDR);
    BMI_WIRE.write(reg);
    BMI_WIRE.write(val);
    BMI_WIRE.endTransmission();
}

static void setupFIFO() {
    bmiWrite(REG_CMD,      0xB0);  // flush
    delay(2);
    bmiWrite(REG_FIFO_CF0, 0x00);
    bmiWrite(REG_FIFO_CF1, 0x70);  // acc + gyr headerless
}

static void flushFIFO() {
    bmiWrite(REG_CMD, 0xB0);
}

static int readFIFOLen() {
    BMI_WIRE.beginTransmission(BMI_ADDR);
    BMI_WIRE.write(REG_FIFO_LEN);
    BMI_WIRE.endTransmission(false);
    BMI_WIRE.requestFrom((uint8_t)BMI_ADDR, (uint8_t)2);
    uint8_t lo = BMI_WIRE.read();
    uint8_t hi = BMI_WIRE.read();
    return ((hi & 0x3F) << 8) | lo;
}

// ---------------------------------------------------------------------------
// Model / Inference
// ---------------------------------------------------------------------------
const int   NUM_CLASSES  = 4;
const char* CLASS_NAMES[] = { "CIRCULO", "DEFAULT", "LADO", "QUIETO" };

const int   NUM_SAMPLES  = 100;  // ventana = 1000 ms a 100 Hz
const int   NUM_FEATURES = 6;    // ax, ay, az, gx, gy, gz

// StandardScaler — entrenado sobre training data (FIFO raw / mismas escalas)
const float SCALER_MEAN[] = { -3.8690f,  0.0855f, -0.1367f,   -5.5769f,   5.6856f, -15.5715f };
const float SCALER_STD[]  = {  0.0112f,  0.6942f,  0.6692f, 1150.2933f, 1162.4168f, 1150.0352f };

constexpr int kTensorArenaSize = 100 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static const tflite::Model*      tfl_model    = nullptr;
static tflite::MicroInterpreter* interp       = nullptr;
static TfLiteTensor*             input_tensor = nullptr;
static TfLiteTensor*             output_tensor= nullptr;
static bool tfl_ready = false;

static float  imu_buf[NUM_SAMPLES][NUM_FEATURES];
static int    sample_count = 0;
static char   json_buf[160];

// ---------------------------------------------------------------------------
// readFIFOFrames — lee frames disponibles y los acumula en imu_buf
// ---------------------------------------------------------------------------
static void readFIFOFrames() {
    int fifoLen  = readFIFOLen();
    int available = fifoLen / FRAME_SIZE;
    if (available <= 0) return;

    int toRead     = min(available, NUM_SAMPLES - sample_count);
    int totalBytes = toRead * FRAME_SIZE;

    // Apuntar al registro FIFO_DATA
    BMI_WIRE.beginTransmission(BMI_ADDR);
    BMI_WIRE.write(REG_FIFO_DAT);
    BMI_WIRE.endTransmission(false);

    static uint8_t rawBuf[240];  // 20 frames máx por lote
    int bytesProcessed = 0;

    while (bytesProcessed < totalBytes) {
        int chunk = min(totalBytes - bytesProcessed, (int)sizeof(rawBuf));
        // chunk debe ser múltiplo de FRAME_SIZE para parseo limpio
        chunk = (chunk / FRAME_SIZE) * FRAME_SIZE;
        if (chunk == 0) break;

        BMI_WIRE.requestFrom((uint8_t)BMI_ADDR, (uint8_t)chunk);
        int got = 0;
        while (got < chunk && BMI_WIRE.available()) {
            rawBuf[got++] = BMI_WIRE.read();
        }

        for (int i = 0; i + FRAME_SIZE <= got; i += FRAME_SIZE) {
            if (sample_count >= NUM_SAMPLES) break;
            uint8_t* f = rawBuf + i;

            // Frame: GYR_X GYR_Y GYR_Z ACC_X ACC_Y ACC_Z (little-endian int16)
            int16_t gx_raw = (int16_t)(f[0]  | (f[1]  << 8));
            int16_t gy_raw = (int16_t)(f[2]  | (f[3]  << 8));
            int16_t gz_raw = (int16_t)(f[4]  | (f[5]  << 8));
            int16_t ax_raw = (int16_t)(f[6]  | (f[7]  << 8));
            int16_t ay_raw = (int16_t)(f[8]  | (f[9]  << 8));
            int16_t az_raw = (int16_t)(f[10] | (f[11] << 8));

            // Mismas escalas que capture.py → mismas unidades que el training data
            float vals[NUM_FEATURES] = {
                ax_raw / ACC_SCALE,
                ay_raw / ACC_SCALE,
                az_raw / ACC_SCALE,
                gx_raw / GYRO_SCALE,
                gy_raw / GYRO_SCALE,
                gz_raw / GYRO_SCALE
            };

            for (int fi = 0; fi < NUM_FEATURES; fi++) {
                imu_buf[sample_count][fi] = (vals[fi] - SCALER_MEAN[fi]) / SCALER_STD[fi];
            }
            sample_count++;
        }

        bytesProcessed += got;
    }
}

// ---------------------------------------------------------------------------
// run_inference
// ---------------------------------------------------------------------------
static void run_inference() {
    float* ptr = input_tensor->data.f;
    for (int s = 0; s < NUM_SAMPLES; s++)
        for (int f = 0; f < NUM_FEATURES; f++)
            ptr[s * NUM_FEATURES + f] = imu_buf[s][f];

    uint32_t t0 = micros();
    if (interp->Invoke() != kTfLiteOk) {
        Serial.println("[ERROR] Invoke() failed");
        return;
    }
    float latency_ms = (micros() - t0) / 1000.0f;

    float* out_ptr = output_tensor->data.f;
    float probs[NUM_CLASSES];
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = out_ptr[i] < 0.0f ? 0.0f : out_ptr[i];
        sum += probs[i];
    }
    if (sum < 1e-9f) sum = 1.0f;

    float pct[NUM_CLASSES];
    int best = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        pct[i] = (probs[i] / sum) * 100.0f;
        if (pct[i] > pct[best]) best = i;
    }

    snprintf(json_buf, sizeof(json_buf),
        "{\"c\":\"%s\",\"p\":[%.1f,%.1f,%.1f,%.1f],\"l\":%.1f,\"t\":%lu}",
        CLASS_NAMES[best],
        pct[0], pct[1], pct[2], pct[3],
        latency_ms,
        (unsigned long)(millis() & 0xFFFFFFFFUL));

    inferenceChar.writeValue(json_buf);
    Serial.print("[INF]  "); Serial.print(CLASS_NAMES[best]);
    Serial.print(" @ "); Serial.print(pct[best], 1);
    Serial.print("% | "); Serial.print(latency_ms, 1); Serial.println("ms");
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("[BOOT] TinyML Sense v3.0 (Conv1D, FIFO raw)");

    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(LEDR, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    digitalWrite(LEDR, HIGH);

    // BLE
    if (!BLE.begin()) {
        Serial.println("[ERROR] BLE.begin() failed — halting");
        while (true) { digitalWrite(LEDR, LOW); delay(100); digitalWrite(LEDR, HIGH); delay(100); }
    }
    BLE.setConnectionInterval(0x0010, 0x0020);
    BLE.setLocalName("TinyML-Sense");
    BLE.setDeviceName("TinyML-Sense");
    BLE.setAdvertisedService(inferenceService);
    inferenceService.addCharacteristic(inferenceChar);
    BLE.addService(inferenceService);
    inferenceChar.writeValue("{}");
    BLE.advertise();
    Serial.print("[BLE]  Service: "); Serial.println(SERVICE_UUID);
    Serial.println("[BLE]  Advertising as 'TinyML-Sense'");

    // IMU — librería necesaria para inicializar el chip BMI270
    if (!IMU.begin()) {
        Serial.println("[ERROR] IMU init failed — inference disabled");
        return;
    }
    // Reconfigurar FIFO igual que recoleccion.ino
    BMI_WIRE.setClock(400000);
    setupFIFO();
    delay(10);
    Serial.println("[IMU]  BMI270 FIFO raw (acc+gyr headerless, 100 Hz)");

    // TFLite Micro
    tfl_model = tflite::GetModel(g_model_data);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("[ERROR] Schema mismatch: model="); Serial.print(tfl_model->version());
        Serial.print(" lib="); Serial.println(TFLITE_SCHEMA_VERSION);
        return;
    }
    Serial.print("[TFL]  Model loaded ("); Serial.print(g_model_data_len); Serial.println(" bytes)");

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interp(
        tfl_model, resolver, tensor_arena, kTensorArenaSize);
    interp = &static_interp;

    if (interp->AllocateTensors() != kTfLiteOk) {
        Serial.print("[ERROR] AllocateTensors() failed — arena=");
        Serial.print(kTensorArenaSize / 1024); Serial.println(" KB");
        return;
    }

    input_tensor  = interp->input(0);
    output_tensor = interp->output(0);
    tfl_ready = true;

    Serial.print("[TFL]  Arena ok ("); Serial.print(kTensorArenaSize / 1024); Serial.println(" KB)");
    Serial.println("[TFL]  Classes: 0=CIRCULO 1=DEFAULT 2=LADO 3=QUIETO");
    Serial.println("[SYS]  Ready. Waiting for BLE central...");
    digitalWrite(LED_BUILTIN, HIGH);
}

// ---------------------------------------------------------------------------
// loop
// ---------------------------------------------------------------------------
void loop() {
    BLEDevice central = BLE.central();

    if (central) {
        Serial.print("[BLE]  Connected: "); Serial.println(central.address());
        digitalWrite(LED_BUILTIN, HIGH);
        sample_count = 0;
        flushFIFO();

        while (central.connected()) {
            BLE.poll();
            if (!tfl_ready) { delay(10); continue; }

            readFIFOFrames();

            if (sample_count >= NUM_SAMPLES) {
                run_inference();
                sample_count = 0;
                flushFIFO();
            }
        }

        Serial.println("[BLE]  Disconnected");
        sample_count = 0;

    } else {
        static unsigned long last_blink = 0;
        unsigned long now = millis();
        if (now - last_blink >= 1000) {
            last_blink = now;
            digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        }
        BLE.poll();
        delay(10);
    }
}
