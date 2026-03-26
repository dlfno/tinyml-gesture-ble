#!/usr/bin/env python3
"""
Entrenador de Modelo — Entrena CNN 1D para clasificación de gestos IMU.
Arquitectura: Conv1D 16→32→64 + GlobalAvgPool + Dense(32) → Dense(4)
Exporta: model.keras, model_float.tflite, model_quantized.tflite
Artefactos: scaler_params.json, training_history.json, predictions.npz, class_names.json
"""
import os, sys, json, unicodedata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks as keras_callbacks
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ── Reproducibilidad ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'data')
MODEL_DIR   = os.path.join(SCRIPT_DIR, '..', 'models', 'cnn')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hiperparámetros ───────────────────────────────────────────────────────────
FEATURE_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
WINDOW_SIZE  = 100  # muestras a 100 Hz = 1000 ms (óptimo real-time: Banos et al. 2014)
STEP_SIZE    = 50   # 50 % de overlap
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15  # del train_full
BATCH_SIZE   = 32
MAX_EPOCHS   = 150


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_label(raw: str) -> str:
    """Normaliza etiqueta a ASCII uppercase: Círculo→CIRCULO, etc."""
    nfkd = unicodedata.normalize('NFKD', raw)
    ascii_str = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_str.upper()


# ── 1. Cargar datos ───────────────────────────────────────────────────────────
print("=" * 60)
print("FASE 1 — Carga de datos")
print("=" * 60)

recordings = []  # list of (array_features, label_str)
file_count = 0
class_file_counts = {}

for root, _, files in os.walk(DATA_DIR):
    for fname in sorted(files):
        if not fname.endswith('.csv'):
            continue
        raw_label = fname.split('_')[0]
        label = normalize_label(raw_label)
        fpath = os.path.join(root, fname)
        try:
            df = pd.read_csv(fpath)
            if not all(c in df.columns for c in FEATURE_COLS):
                print(f"  [WARN] columnas faltantes: {fpath}")
                continue
            arr = df[FEATURE_COLS].values.astype(np.float32)
            recordings.append((arr, label))
            class_file_counts[label] = class_file_counts.get(label, 0) + 1
            file_count += 1
        except Exception as e:
            print(f"  [WARN] {fpath}: {e}")

if not recordings:
    print("ERROR: No se encontraron CSVs válidos.")
    sys.exit(1)

print(f"Archivos cargados  : {file_count}")
print(f"Archivos por clase : {class_file_counts}")
print()

# ── 2. Crear ventanas deslizantes ─────────────────────────────────────────────
print("FASE 2 — Creación de ventanas (window={}, step={})".format(WINDOW_SIZE, STEP_SIZE))

X_all, y_all = [], []
for arr, label in recordings:
    n = len(arr)
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        X_all.append(arr[start: start + WINDOW_SIZE])
        y_all.append(label)

X_all = np.array(X_all, dtype=np.float32)  # (N, 100, 6)
y_str = np.array(y_all)

print(f"Ventanas totales   : {len(X_all):,}  shape={X_all.shape}")
unique, counts = np.unique(y_str, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {cls:10s}: {cnt:,}")
print()

# ── 3. Label encoding ─────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y_str)
class_names = list(le.classes_)
print(f"Clases (orden LabelEncoder): {class_names}")

# ── 4. Split estratificado ────────────────────────────────────────────────────
print("\nFASE 3 — Split de datos")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_all, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_train_full)

print(f"Train : {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

# ── 5. Normalización (scaler ajustado SOLO en train) ─────────────────────────
print("\nFASE 4 — Normalización (StandardScaler fit en train)")
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, len(FEATURE_COLS))
scaler.fit(X_train_flat)

def apply_scaler(X):
    return scaler.transform(X.reshape(-1, len(FEATURE_COLS))).reshape(X.shape)

X_train = apply_scaler(X_train)
X_val   = apply_scaler(X_val)
X_test  = apply_scaler(X_test)

print(f"SCALER_MEAN : {scaler.mean_.tolist()}")
print(f"SCALER_STD  : {scaler.scale_.tolist()}")

scaler_path = os.path.join(MODEL_DIR, 'scaler_params.json')
with open(scaler_path, 'w') as f:
    json.dump({"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}, f, indent=2)
print(f"Guardado: {scaler_path}")

# ── 6. Pesos de clase (corrige imbalance DEFAULT) ─────────────────────────────
class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}
print(f"\nClass weights: {class_weight_dict}")

# ── 7. Arquitectura CNN 1D ────────────────────────────────────────────────────
print("\nFASE 5 — Construcción del modelo CNN 1D")

def build_cnn_model(input_shape=(WINDOW_SIZE, len(FEATURE_COLS)), num_classes=len(class_names)):
    model = models.Sequential([
        # Bloque 1 — patrones locales
        layers.Conv1D(16, kernel_size=3, activation='relu', padding='same',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),           # 100 → 50

        # Bloque 2 — patrones de nivel medio
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),           # 50 → 25

        # Bloque 3 — patrones globales
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),            # 25×64 → 64

        # Clasificador
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ], name='gesture_cnn')
    return model

model = build_cnn_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── 8. Entrenamiento ──────────────────────────────────────────────────────────
print("\nFASE 6 — Entrenamiento")
keras_path = os.path.join(MODEL_DIR, 'model.keras')

cb_list = [
    keras_callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    keras_callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    keras_callbacks.ModelCheckpoint(
        keras_path, monitor='val_accuracy', save_best_only=True, verbose=0),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=cb_list,
    verbose=1
)

# Reload best weights (ModelCheckpoint)
model = tf.keras.models.load_model(keras_path)
print(f"Modelo cargado desde checkpoint: {keras_path}")

# ── 9. Evaluación — modelo Keras (float32) ───────────────────────────────────
print("\nFASE 7 — Evaluación float32")
y_pred_float = model.predict(X_test, verbose=0).argmax(axis=1)
acc_float = accuracy_score(y_test, y_pred_float)
print(f"Float32 Accuracy: {acc_float*100:.2f}%")
print(classification_report(y_test, y_pred_float, target_names=class_names))

# ── 10. Exportar TFLite float32 ───────────────────────────────────────────────
print("FASE 8 — Exportar TFLite float32")
float_path = os.path.join(MODEL_DIR, 'model_float.tflite')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float = converter.convert()
with open(float_path, 'wb') as f:
    f.write(tflite_float)
print(f"Float TFLite: {os.path.getsize(float_path)/1024:.1f} KB → {float_path}")

# ── 11. Exportar TFLite INT8 cuantizado ──────────────────────────────────────
print("FASE 9 — Exportar TFLite INT8 cuantizado")
quant_path = os.path.join(MODEL_DIR, 'model_quantized.tflite')

def representative_dataset():
    n = min(200, len(X_train))
    for i in range(n):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_quant = converter.convert()
with open(quant_path, 'wb') as f:
    f.write(tflite_quant)
print(f"Quant TFLite: {os.path.getsize(quant_path)/1024:.1f} KB → {quant_path}")

size_kb = os.path.getsize(quant_path) / 1024
if size_kb > 60:
    print(f"[WARN] Modelo cuantizado {size_kb:.1f} KB supera el objetivo de 60 KB.")

# ── 12. Evaluación — TFLite cuantizado ───────────────────────────────────────
print("\nFASE 10 — Evaluación TFLite cuantizado")
interp = tf.lite.Interpreter(model_path=quant_path)
interp.allocate_tensors()
inp_det = interp.get_input_details()
out_det = interp.get_output_details()

y_pred_quant = []
for i in range(len(X_test)):
    interp.set_tensor(inp_det[0]['index'], X_test[i:i+1].astype(np.float32))
    interp.invoke()
    y_pred_quant.append(int(interp.get_tensor(out_det[0]['index']).argmax()))
y_pred_quant = np.array(y_pred_quant)

acc_quant = accuracy_score(y_test, y_pred_quant)
print(f"Quant Accuracy: {acc_quant*100:.2f}%")
print(classification_report(y_test, y_pred_quant, target_names=class_names))

delta = abs(acc_float - acc_quant)
if delta > 0.03:
    print(f"[WARN] Degradación por cuantización {delta*100:.2f} pp (>3%). Considerar QAT.")
else:
    print(f"[OK] Degradación por cuantización: {delta*100:.2f} pp")

# ── 13. Guardar artefactos ────────────────────────────────────────────────────
print("\nFASE 11 — Guardar artefactos")

# Historial
history_path = os.path.join(MODEL_DIR, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f, indent=2)
print(f"Guardado: {history_path}")

# Predicciones
preds_path = os.path.join(MODEL_DIR, 'predictions.npz')
np.savez(preds_path,
         y_test=y_test,
         y_pred_float=y_pred_float,
         y_pred_quant=y_pred_quant,
         class_names=np.array(class_names))
print(f"Guardado: {preds_path}")

# Nombres de clases
cn_path = os.path.join(MODEL_DIR, 'class_names.json')
with open(cn_path, 'w') as f:
    json.dump(class_names, f)
print(f"Guardado: {cn_path}")

# ── Resumen final ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"Clases        : {class_names}")
print(f"Float Acc     : {acc_float*100:.2f}%")
print(f"Quant Acc     : {acc_quant*100:.2f}%")
print(f"Float size    : {os.path.getsize(float_path)/1024:.1f} KB")
print(f"Quant size    : {os.path.getsize(quant_path)/1024:.1f} KB")
print(f"Epochs        : {len(history.history['loss'])}")
val_acc = history.history.get('val_accuracy', [])
if val_acc:
    best_ep = int(np.argmax(val_acc)) + 1
    print(f"Mejor epoch   : {best_ep}  (val_acc={max(val_acc)*100:.2f}%)")
print("=" * 60)
print("\n// Arduino — copiar a tinyml_ble_cnn.ino si scaler_params.json no se lee automáticamente:")
mean_str = ', '.join(f'{v:.4f}f' for v in scaler.mean_)
std_str  = ', '.join(f'{v:.4f}f' for v in scaler.scale_)
print(f"const float SCALER_MEAN[] = {{ {mean_str} }};")
print(f"const float SCALER_STD[]  = {{ {std_str} }};")
print("\nEntrenamiento completado exitosamente.")
