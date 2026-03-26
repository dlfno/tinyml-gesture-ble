#!/usr/bin/env python3
"""
Entrenador Gaussian Naive Bayes — Clasificador de gestos IMU.

Rol en el proyecto: referencia de complejidad mínima.
  - Modelo más pequeño posible: ~1 KB (256 parámetros μ/σ²)
  - Sin dependencia de TFLite Micro ni StandardScaler
  - Latencia de inferencia < 0.1 ms
  - Si el accuracy de GNB está cerca del de RF, eso indicaría que
    las 32 features estadísticas capturan casi toda la información
    necesaria y que la estructura de interacciones es débil.

Pipeline:
  1. Ventanas deslizantes idénticas a CNN/RF/LR (WINDOW_SIZE=100, STEP_SIZE=50)
  2. Extracción de 32 features estadísticas por ventana
  3. GaussianNB() — sin escalado (GNB es invariante a escala)
  4. Exporta model_nb.h (C++ para Arduino via micromlgen)
  5. Guarda predictions_nb.npz, class_names_nb.json

Exporta: model_nb.h (Arduino C++)
Artefactos: predictions_nb.npz, class_names_nb.json
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

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
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
MODEL_DIR  = os.path.join(SCRIPT_DIR, '..', 'models', 'nb')
EVAL_DIR   = os.path.join(SCRIPT_DIR, '..', 'eval', 'nb')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ── Hiperparámetros — mismo pipeline de ventanas que CNN/RF/LR ────────────────
FEATURE_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
WINDOW_SIZE  = 100   # muestras a 100 Hz = 1000 ms
STEP_SIZE    = 50    # 50% de solapamiento
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15

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
    Contrato idéntico al de train_rf.py/train_lr.py y al firmware Arduino.
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

# ── 5. Entrenamiento ──────────────────────────────────────────────────────────
# GaussianNB no requiere StandardScaler: modela distribuciones gaussianas
# por feature por clase, por lo que la escala absoluta no afecta la inferencia.
print('\nFASE 5 — Entrenamiento GaussianNB')

clf = GaussianNB()
clf.fit(X_train, y_train)
print('  Entrenamiento completado.')
print(f'  Parámetros del modelo: {len(class_names)} clases × {N_FEATURES} features × 2 (μ, σ²) = '
      f'{len(class_names) * N_FEATURES * 2} valores')

# ── 6. Evaluación ─────────────────────────────────────────────────────────────
print('\nFASE 6 — Evaluación')

y_pred_train = clf.predict(X_train)
y_pred_val   = clf.predict(X_val)
y_pred_test  = clf.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_val   = accuracy_score(y_val,   y_pred_val)
acc_test  = accuracy_score(y_test,  y_pred_test)

print(f'  Train accuracy : {acc_train*100:.2f}%')
print(f'  Val   accuracy : {acc_val*100:.2f}%')
print(f'  Test  accuracy : {acc_test*100:.2f}%')
print()
print(classification_report(y_test, y_pred_test, target_names=class_names))

# ── 7. Validación cruzada ─────────────────────────────────────────────────────
print('FASE 7 — Validación cruzada (5-fold sobre train+val)')

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])
cv_scores = cross_val_score(
    GaussianNB(),
    X_trainval, y_trainval, cv=5, scoring='accuracy', n_jobs=-1)
print(f'  CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%')

# ── 8. Guardar artefactos ─────────────────────────────────────────────────────
print('\nFASE 8 — Guardar artefactos')

class_names_path = os.path.join(MODEL_DIR, 'class_names_nb.json')
with open(class_names_path, 'w') as f:
    json.dump(class_names, f, indent=2)
print(f'  Guardado: {class_names_path}')

predictions_path = os.path.join(MODEL_DIR, 'predictions_nb.npz')
np.savez(predictions_path,
         y_test=y_test,
         y_pred=y_pred_test,
         class_names=np.array(class_names))
print(f'  Guardado: {predictions_path}')

# ── 9. Exportar modelo como C++ para Arduino ──────────────────────────────────
print('\nFASE 9 — Exportar modelo C++ para Arduino (micromlgen)')

model_h_path = os.path.join(MODEL_DIR, 'model_nb.h')
try:
    from micromlgen import port
    # sklearn >= 1.0 renombró sigma_ a var_; micromlgen espera el atributo sigma_
    if not hasattr(clf, 'sigma_') and hasattr(clf, 'var_'):
        clf.sigma_ = clf.var_
    classmap = {i: name for i, name in enumerate(class_names)}
    c_code = port(clf, classname='NBModel', classmap=classmap)
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
PRIMARY   = '#f97316'   # naranja para Naive Bayes

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
ax.set_title(f'Gaussian Naive Bayes — Test Accuracy: {acc_test:.1%}',
             fontweight='bold', color=TEXT, pad=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
cm_path = os.path.join(EVAL_DIR, 'confusion_matrix_nb.png')
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
fig.suptitle('Gaussian Naive Bayes — Per-Class Metrics', fontsize=14,
             fontweight='bold', color=TEXT)
plt.tight_layout()
metrics_path = os.path.join(EVAL_DIR, 'per_class_metrics_nb.png')
fig.savefig(metrics_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(metrics_path)}')

# Medias condicionales por clase (heatmap: clases × features)
# Visualiza qué valores de features caracterizan a cada clase según GNB.
# Se normaliza cada feature por su rango para hacerlo comparable entre features.
fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_CARD)
means = clf.theta_                                   # (n_classes, n_features)
f_min = means.min(axis=0, keepdims=True)
f_max = means.max(axis=0, keepdims=True)
means_norm = (means - f_min) / np.where(f_max - f_min > 1e-8, f_max - f_min, 1)
cmap_means = LinearSegmentedColormap.from_list(
    'means', [BG_CARD, PRIMARY], N=256)
im = ax.imshow(means_norm, aspect='auto', cmap=cmap_means, vmin=0, vmax=1)
ax.set_xticks(range(N_FEATURES))
ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
ax.set_yticks(range(len(class_names)))
ax.set_yticklabels(class_names, fontsize=9)
ax.set_title('Gaussian Naive Bayes — Medias condicionales por clase (normalizadas por feature)',
             fontweight='bold', color=TEXT, pad=10)
plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label='Relativo (0=min, 1=max)')
plt.tight_layout()
means_path = os.path.join(EVAL_DIR, 'class_means_nb.png')
fig.savefig(means_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(means_path)}')

# ── 11. Reporte Markdown ──────────────────────────────────────────────────────
print('\nFASE 11 — Reporte Markdown')

report_str = classification_report(y_test, y_pred_test, target_names=class_names)

md = f"""# Reporte de Evaluación — Gaussian Naive Bayes

## Resumen
- **Algoritmo:** Gaussian Naive Bayes (independencia condicional entre features)
- **Hiperparámetros:** var_smoothing=1e-9 (default)
- **Parámetros del modelo:** {len(class_names)} clases × {N_FEATURES} features × 2 (μ, σ²) = {len(class_names) * N_FEATURES * 2} valores
- **Preprocesamiento:** ninguno — GNB es invariante a escala
- **Features:** {N_FEATURES} (5 estadísticos × 6 ejes + 2 SMA)
- **Ventana:** {WINDOW_SIZE} muestras @ 100 Hz ({WINDOW_SIZE * 10} ms)
- **Rol:** Referencia de complejidad mínima
- **Dependencia de TFLite:** ninguna (C++ puro via micromlgen)
- **Formato de exportación:** `model_nb.h` (Arduino C++)

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
| `model_nb.h` | {model_h_kb:.2f} KB (texto, no compilado) |

## Archivos Generados
- `confusion_matrix_nb.png`
- `per_class_metrics_nb.png`
- `class_means_nb.png`
"""

report_path = os.path.join(EVAL_DIR, 'report.md')
with open(report_path, 'w') as f:
    f.write(md)
print(f'  Guardado: {os.path.relpath(report_path)}')

# ── Resumen final ─────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('Entrenamiento NB completado.')
print('=' * 60)
print(f'  Test accuracy : {acc_test*100:.2f}%')
print(f'  CV accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%')
print(f'  model_nb.h    : {model_h_kb:.2f} KB')
print()
print('Próximo paso — copiar model_nb.h al sketch Arduino:')
print(f'  cp models/nb/model_nb.h arduino/tinyml_ble_nb/model_nb.h')
