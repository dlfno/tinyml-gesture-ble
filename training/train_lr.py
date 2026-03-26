#!/usr/bin/env python3
"""
Entrenador Regresión Logística — Clasificador de gestos IMU.

Rol en el proyecto: baseline lineal de referencia.
  - Sin dependencia de TFLite Micro (~200 KB de flash ahorro)
  - El modelo exportado es código C++ puro (~1-2 KB)
  - Latencia de inferencia < 0.1 ms (128 multiplicaciones + softmax)
  - Establece el piso de rendimiento: si SVM o RF superan a Logística
    en más de 5 pp, eso justifica su complejidad adicional.

Pipeline:
  1. Ventanas deslizantes idénticas a CNN/RF (WINDOW_SIZE=100, STEP_SIZE=50)
  2. Extracción de 32 features estadísticas por ventana
  3. StandardScaler sobre los 32 features (LR requiere features escalados)
  4. LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs')
  5. Exporta model_lr.h (C++ para Arduino via micromlgen, coefs ya en espacio escalado)
  6. Guarda scaler_params_lr.json (mean/std de los 32 features para el firmware)
  7. Guarda predictions_lr.npz, class_names_lr.json

Nota sobre despliegue en Arduino:
  El firmware debe aplicar Z-score a los 32 features extraídos usando los
  valores de scaler_params_lr.json antes de llamar a LRModel::predict().

Exporta: model_lr.h (Arduino C++)
Artefactos: predictions_lr.npz, class_names_lr.json, scaler_params_lr.json
"""
import os
import sys
import json
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)

# ── Reproducibilidad ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
MODEL_DIR  = os.path.join(SCRIPT_DIR, '..', 'models', 'lr')
EVAL_DIR   = os.path.join(SCRIPT_DIR, '..', 'eval', 'lr')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ── Hiperparámetros — mismo pipeline de ventanas que CNN/RF ───────────────────
FEATURE_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
WINDOW_SIZE  = 100   # muestras a 100 Hz = 1000 ms
STEP_SIZE    = 50    # 50% de solapamiento
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15

# Hiperparámetros del clasificador
C_REG    = 1.0
MAX_ITER = 5000   # LR con lbfgs converge en ~300-500 iter con features escalados

# ── Nombres de features (32 en total) ─────────────────────────────────────────
# Debe coincidir EXACTAMENTE con extractFeatures() en el firmware Arduino.
FEATURE_NAMES = []
for axis in FEATURE_COLS:
    for stat in ['mean', 'std', 'min', 'max', 'rms']:
        FEATURE_NAMES.append(f'{axis}_{stat}')
FEATURE_NAMES.append('sma_accel')
FEATURE_NAMES.append('sma_gyro')
N_FEATURES = len(FEATURE_NAMES)     # 32


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_label(raw: str) -> str:
    nfkd = unicodedata.normalize('NFKD', raw)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).upper()


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extrae 32 features estadísticas de una ventana (WINDOW_SIZE × 6).
    Contrato idéntico al de train_rf.py y al firmware Arduino.
    """
    feats = []
    for i in range(len(FEATURE_COLS)):
        col = window[:, i]
        feats.append(float(np.mean(col)))
        feats.append(float(np.std(col)))
        feats.append(float(np.min(col)))
        feats.append(float(np.max(col)))
        feats.append(float(np.sqrt(np.mean(col ** 2))))   # RMS
    # Signal Magnitude Area
    feats.append(float(np.mean(
        np.abs(window[:, 0]) + np.abs(window[:, 1]) + np.abs(window[:, 2]))))
    feats.append(float(np.mean(
        np.abs(window[:, 3]) + np.abs(window[:, 4]) + np.abs(window[:, 5]))))
    return np.array(feats, dtype=np.float32)


# ── 1. Cargar datos ───────────────────────────────────────────────────────────
print('=' * 60)
print('FASE 1 — Carga de datos')
print('=' * 60)

recordings = []
class_file_counts = {}
file_count = 0

for root, _, files in os.walk(DATA_DIR):
    for fname in sorted(files):
        if not fname.endswith('.csv'):
            continue
        label = normalize_label(fname.split('_')[0])
        fpath = os.path.join(root, fname)
        try:
            df = pd.read_csv(fpath)
            if not all(c in df.columns for c in FEATURE_COLS):
                print(f'  [WARN] columnas faltantes: {fpath}')
                continue
            arr = df[FEATURE_COLS].values.astype(np.float32)
            recordings.append((arr, label))
            class_file_counts[label] = class_file_counts.get(label, 0) + 1
            file_count += 1
        except Exception as e:
            print(f'  [WARN] {fpath}: {e}')

if not recordings:
    print('ERROR: No se encontraron CSVs válidos.')
    sys.exit(1)

print(f'Archivos cargados  : {file_count}')
print(f'Archivos por clase : {class_file_counts}')

# ── 2. Ventanas deslizantes ───────────────────────────────────────────────────
print(f'\nFASE 2 — Ventanas (size={WINDOW_SIZE}, step={STEP_SIZE})')

raw_windows, y_str = [], []
for arr, label in recordings:
    for start in range(0, len(arr) - WINDOW_SIZE + 1, STEP_SIZE):
        raw_windows.append(arr[start:start + WINDOW_SIZE])
        y_str.append(label)

raw_windows = np.array(raw_windows, dtype=np.float32)
print(f'Ventanas totales   : {len(raw_windows)}')

# ── 3. Extracción de features ─────────────────────────────────────────────────
print(f'\nFASE 3 — Extracción de {N_FEATURES} features estadísticas por ventana')
print(f'  Features: {FEATURE_NAMES}')

X_all = np.array([extract_features(w) for w in raw_windows], dtype=np.float32)
print(f'  Matriz de features: {X_all.shape}')

le = LabelEncoder()
y_all = le.fit_transform(y_str)
class_names = list(le.classes_)
print(f'  Clases: {class_names}')

# ── 4. Split train / val / test ───────────────────────────────────────────────
print('\nFASE 4 — División de datos')

X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_temp)

print(f'  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}')

# ── 5. Escalado de features ───────────────────────────────────────────────────
# LR requiere features escalados. RF es invariante a escala, LR no.
# El scaler se ajusta SOLO sobre X_train para evitar data leakage.
print('\nFASE 5a — StandardScaler (fit en train, transform en todos los splits)')

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
X_trainval_s = scaler.transform(np.vstack([X_train, X_val]))
print(f'  Scaler ajustado. Mean[0]={scaler.mean_[0]:.4f}, Std[0]={scaler.scale_[0]:.4f}')

# ── 6. Entrenamiento ──────────────────────────────────────────────────────────
print(f'\nFASE 5b — Entrenamiento LR (C={C_REG}, max_iter={MAX_ITER}, solver=lbfgs)')

clf = LogisticRegression(
    C=C_REG,
    max_iter=MAX_ITER,
    solver='lbfgs',
    class_weight='balanced',
    random_state=RANDOM_SEED,
)
clf.fit(X_train_s, y_train)
print('  Entrenamiento completado.')
print(f'  Iteraciones convergencia: {clf.n_iter_[0]}')

# ── 7. Evaluación ─────────────────────────────────────────────────────────────
print('\nFASE 6 — Evaluación')

y_pred_train = clf.predict(X_train_s)
y_pred_val   = clf.predict(X_val_s)
y_pred_test  = clf.predict(X_test_s)

acc_train = accuracy_score(y_train, y_pred_train)
acc_val   = accuracy_score(y_val,   y_pred_val)
acc_test  = accuracy_score(y_test,  y_pred_test)

print(f'  Train accuracy : {acc_train*100:.2f}%')
print(f'  Val   accuracy : {acc_val*100:.2f}%')
print(f'  Test  accuracy : {acc_test*100:.2f}%')
print()
print(classification_report(y_test, y_pred_test, target_names=class_names))

# ── 8. Validación cruzada ─────────────────────────────────────────────────────
print('FASE 7 — Validación cruzada (5-fold sobre train+val, con scaler dentro del fold)')

from sklearn.pipeline import Pipeline as SKPipeline

y_trainval = np.hstack([y_train, y_val])
# Pipeline interno para CV: scaler + LR en cada fold (evita data leakage)
cv_pipe = SKPipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        C=C_REG, max_iter=MAX_ITER, solver='lbfgs',
        class_weight='balanced', random_state=RANDOM_SEED)),
])
cv_scores = cross_val_score(
    cv_pipe,
    np.vstack([X_train, X_val]), y_trainval,
    cv=5, scoring='accuracy', n_jobs=-1)
print(f'  CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%')

# ── 9. Guardar artefactos ─────────────────────────────────────────────────────
print('\nFASE 8 — Guardar artefactos')

class_names_path = os.path.join(MODEL_DIR, 'class_names_lr.json')
with open(class_names_path, 'w') as f:
    json.dump(class_names, f, indent=2)
print(f'  Guardado: {class_names_path}')

predictions_path = os.path.join(MODEL_DIR, 'predictions_lr.npz')
np.savez(predictions_path,
         y_test=y_test,
         y_pred=y_pred_test,
         class_names=np.array(class_names))
print(f'  Guardado: {predictions_path}')

# Parámetros del scaler para el firmware Arduino
# El firmware debe aplicar Z-score a los 32 features antes de llamar a LRModel::predict()
scaler_params_path = os.path.join(MODEL_DIR, 'scaler_params_lr.json')
with open(scaler_params_path, 'w') as f:
    json.dump({
        'feature_names': FEATURE_NAMES,
        'mean':  scaler.mean_.tolist(),
        'std':   scaler.scale_.tolist(),
    }, f, indent=2)
print(f'  Guardado: {scaler_params_path}')

# ── 9. Exportar modelo como C++ para Arduino ──────────────────────────────────
print('\nFASE 9 — Exportar modelo C++ para Arduino (micromlgen)')

model_h_path = os.path.join(MODEL_DIR, 'model_lr.h')
try:
    from micromlgen import port
    classmap = {i: name for i, name in enumerate(class_names)}
    c_code = port(clf, classname='LRModel', classmap=classmap)
    with open(model_h_path, 'w') as f:
        f.write(c_code)
    model_h_kb = os.path.getsize(model_h_path) / 1024
    print(f'  Guardado: {model_h_path} ({model_h_kb:.2f} KB)')
except Exception as e:
    print(f'  [ERROR] micromlgen falló: {e}')
    sys.exit(1)

# ── 10. Gráficas de evaluación ────────────────────────────────────────────────
print('\nFASE 10 — Gráficas de evaluación')

BG_DARK   = '#0a0f1a'
BG_CARD   = '#111827'
SECONDARY = '#6b7280'
TEXT      = '#e5e7eb'
PRIMARY   = '#a855f7'   # violeta para Regresión Logística

plt.rcParams.update({
    'figure.facecolor': BG_DARK, 'axes.facecolor': BG_CARD,
    'axes.edgecolor': SECONDARY, 'axes.labelcolor': TEXT,
    'text.color': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
    'grid.color': '#1f2937', 'grid.alpha': 0.5,
    'font.family': 'monospace', 'font.size': 11,
    'legend.facecolor': BG_CARD, 'legend.edgecolor': SECONDARY,
})

from matplotlib.colors import LinearSegmentedColormap

# Confusion matrix
cm      = confusion_matrix(y_test, y_pred_test)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
n       = len(class_names)

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_CARD)
cmap = LinearSegmentedColormap.from_list('c', [BG_CARD, PRIMARY], N=256)
im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
for i in range(n):
    for j in range(n):
        c_val = 'white' if cm_norm[i, j] > 0.5 else TEXT
        ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)',
                ha='center', va='center', color=c_val,
                fontsize=9, fontweight='bold' if i == j else 'normal')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(class_names, fontsize=9)
ax.set_yticklabels(class_names, fontsize=9)
ax.set_xlabel('Predicted', fontweight='bold')
ax.set_ylabel('True', fontweight='bold')
ax.set_title(f'Regresión Logística — Test Accuracy: {acc_test:.1%}',
             fontweight='bold', color=TEXT, pad=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
cm_path = os.path.join(EVAL_DIR, 'confusion_matrix_lr.png')
fig.savefig(cm_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(cm_path)}')

# Per-class metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG_DARK)
prf = precision_recall_fscore_support(y_test, y_pred_test, average=None)
for ax, (mname, vals) in zip(axes, [
    ('Precision', prf[0]), ('Recall', prf[1]), ('F1-Score', prf[2])
]):
    ax.set_facecolor(BG_CARD)
    bars = ax.bar(range(n), vals, color=PRIMARY, alpha=0.9)
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f'{v:.2f}',
                ha='center', va='bottom', fontsize=9, color=TEXT)
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title(mname, fontweight='bold', color=TEXT)
    ax.set_ylabel(mname)
    ax.grid(True, axis='y', alpha=0.4)
fig.suptitle('Regresión Logística — Per-Class Metrics', fontsize=14,
             fontweight='bold', color=TEXT)
plt.tight_layout()
metrics_path = os.path.join(EVAL_DIR, 'per_class_metrics_lr.png')
fig.savefig(metrics_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(metrics_path)}')

# Coeficientes del modelo (heatmap: clases × features)
# Muestra qué features empuja el modelo para cada clase.
fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_CARD)
coef = clf.coef_   # (n_classes, n_features)
vmax = np.abs(coef).max()
cmap_coef = LinearSegmentedColormap.from_list(
    'coef', ['#ef4444', BG_CARD, PRIMARY], N=256)
im = ax.imshow(coef, aspect='auto', cmap=cmap_coef, vmin=-vmax, vmax=vmax)
ax.set_xticks(range(N_FEATURES))
ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
ax.set_yticks(range(len(class_names)))
ax.set_yticklabels(class_names, fontsize=9)
ax.set_title('Regresión Logística — Coeficientes por Clase (rojo=negativo, violeta=positivo)',
             fontweight='bold', color=TEXT, pad=10)
plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
plt.tight_layout()
coef_path = os.path.join(EVAL_DIR, 'coefficients_lr.png')
fig.savefig(coef_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(coef_path)}')

# ── 11. Reporte Markdown ──────────────────────────────────────────────────────
print('\nFASE 11 — Reporte Markdown')

report_str = classification_report(y_test, y_pred_test, target_names=class_names)

md = f"""# Reporte de Evaluación — Regresión Logística

## Resumen
- **Algoritmo:** Logistic Regression (solver=lbfgs, multinomial)
- **Hiperparámetros:** C={C_REG}, max_iter={MAX_ITER}, class_weight='balanced'
- **Iteraciones de convergencia:** {clf.n_iter_[0]}
- **Preprocesamiento:** StandardScaler sobre los 32 features (fit en train)
- **Features:** {N_FEATURES} (5 estadísticos × 6 ejes + 2 SMA)
- **Ventana:** {WINDOW_SIZE} muestras @ 100 Hz ({WINDOW_SIZE * 10} ms)
- **Rol:** Baseline lineal de referencia
- **Dependencia de TFLite:** ninguna (C++ puro via micromlgen)
- **Formato de exportación:** `model_lr.h` (Arduino C++)
- **Nota de firmware:** aplicar Z-score con `scaler_params_lr.json` antes de `LRModel::predict()`

## Resultados
| Conjunto | Accuracy |
|----------|----------|
| Train | {acc_train*100:.2f}% |
| Val   | {acc_val*100:.2f}% |
| Test  | {acc_test*100:.2f}% |
| CV 5-fold | {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}% |

## Classification Report
```
{report_str}
```

## Tamaño del modelo C++
| Archivo | Tamaño |
|---------|--------|
| `model_lr.h` | {model_h_kb:.2f} KB (texto, no compilado) |
| `scaler_params_lr.json` | < 5 KB (parámetros del StandardScaler para firmware) |

## Archivos Generados
- `confusion_matrix_lr.png`
- `per_class_metrics_lr.png`
- `coefficients_lr.png`
"""

report_path = os.path.join(EVAL_DIR, 'report.md')
with open(report_path, 'w') as f:
    f.write(md)
print(f'  Guardado: {os.path.relpath(report_path)}')

# ── Resumen final ─────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('Entrenamiento LR completado.')
print('=' * 60)
print(f'  Test accuracy : {acc_test*100:.2f}%')
print(f'  CV accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%')
print(f'  model_lr.h    : {model_h_kb:.2f} KB')
print()
print('Próximo paso — copiar model_lr.h al sketch Arduino:')
print(f'  cp models/lr/model_lr.h arduino/tinyml_ble_lr/model_lr.h')
