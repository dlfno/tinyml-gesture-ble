#!/usr/bin/env python3
"""
Entrenador Random Forest — Clasificador de gestos IMU.

Ventaja sobre CNN/LSTM en Arduino:
  - Sin dependencia de TFLite Micro (~200 KB de flash ahorro)
  - El modelo exportado es código C++ puro (if/else anidados)
  - Inferencia en microsegundos vs ~14 ms del CNN
  - Interpretable: importancia de features disponible

Pipeline:
  1. Ventanas deslizantes idénticas a CNN/LSTM (WINDOW_SIZE=100, STEP_SIZE=50)
  2. Extracción de 32 features estadísticas por ventana
  3. RandomForestClassifier(n_estimators=20, max_depth=8)
  4. Exporta model_rf.h (C++ para Arduino via micromlgen)
  5. Guarda predictions_rf.npz, feature_names_rf.json

Exporta: model_rf.h (Arduino C++)
Artefactos: predictions_rf.npz, feature_names_rf.json, feature_importance_rf.json
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight

# ── Reproducibilidad ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
MODEL_DIR  = os.path.join(SCRIPT_DIR, '..', 'models', 'rf')
EVAL_DIR   = os.path.join(SCRIPT_DIR, '..', 'eval', 'rf')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ── Hiperparámetros — mismo pipeline de ventanas que CNN/LSTM ─────────────────
FEATURE_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
WINDOW_SIZE  = 100   # muestras a 100 Hz = 1000 ms
STEP_SIZE    = 50    # 50% de solapamiento
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15
N_ESTIMATORS = 20
MAX_DEPTH    = 8

# ── Nombres de features (32 en total) ─────────────────────────────────────────
# Formato: {eje}_{estadístico} + señales de magnitud
# Debe coincidir EXACTAMENTE con extractFeatures() en el firmware Arduino.
FEATURE_NAMES = []
for axis in FEATURE_COLS:
    for stat in ['mean', 'std', 'min', 'max', 'rms']:
        FEATURE_NAMES.append(f'{axis}_{stat}')
FEATURE_NAMES.append('sma_accel')   # Signal Magnitude Area acelerómetro
FEATURE_NAMES.append('sma_gyro')    # Signal Magnitude Area giroscopio
N_FEATURES = len(FEATURE_NAMES)     # 32


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_label(raw: str) -> str:
    nfkd = unicodedata.normalize('NFKD', raw)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).upper()


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extrae 32 features estadísticas de una ventana (WINDOW_SIZE × 6).
    Esta función define el contrato con el firmware Arduino:
    cada feature debe calcularse de forma idéntica en C++.
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

raw_windows = np.array(raw_windows, dtype=np.float32)   # (N, 100, 6)
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

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
sample_weights = np.array([class_weight_dict[y] for y in y_train])

# ── 5. Entrenamiento ──────────────────────────────────────────────────────────
print(f'\nFASE 5 — Entrenamiento RF (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})')

clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
clf.fit(X_train, y_train)
print('  Entrenamiento completado.')

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
    RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
    X_trainval, y_trainval, cv=5, scoring='accuracy', n_jobs=-1)
print(f'  CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%')

# ── 8. Feature importance ─────────────────────────────────────────────────────
print('\nFASE 8 — Importancia de features (top 10)')

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for rank, i in enumerate(sorted_idx[:10]):
    print(f'  #{rank+1:2d}  {FEATURE_NAMES[i]:20s}  {importances[i]:.4f}')

# ── 9. Guardar artefactos ─────────────────────────────────────────────────────
print('\nFASE 9 — Guardar artefactos')

# Feature names
feature_names_path = os.path.join(MODEL_DIR, 'feature_names_rf.json')
with open(feature_names_path, 'w') as f:
    json.dump(FEATURE_NAMES, f, indent=2)
print(f'  Guardado: {feature_names_path}')

# Feature importance
importance_path = os.path.join(MODEL_DIR, 'feature_importance_rf.json')
with open(importance_path, 'w') as f:
    json.dump({
        'features': FEATURE_NAMES,
        'importances': importances.tolist(),
        'sorted_indices': sorted_idx.tolist(),
    }, f, indent=2)
print(f'  Guardado: {importance_path}')

# Class names
class_names_path = os.path.join(MODEL_DIR, 'class_names_rf.json')
with open(class_names_path, 'w') as f:
    json.dump(class_names, f, indent=2)
print(f'  Guardado: {class_names_path}')

# Predictions
predictions_path = os.path.join(MODEL_DIR, 'predictions_rf.npz')
np.savez(predictions_path,
         y_test=y_test,
         y_pred=y_pred_test,
         class_names=np.array(class_names))
print(f'  Guardado: {predictions_path}')

# ── 10. Exportar modelo como C++ para Arduino ─────────────────────────────────
print('\nFASE 10 — Exportar modelo C++ para Arduino (micromlgen)')

model_h_path = os.path.join(MODEL_DIR, 'model_rf.h')
try:
    from micromlgen import port
    classmap = {i: name for i, name in enumerate(class_names)}
    c_code = port(clf, classname='RFModel', classmap=classmap)
    with open(model_h_path, 'w') as f:
        f.write(c_code)
    model_h_kb = os.path.getsize(model_h_path) / 1024
    print(f'  Guardado: {model_h_path} ({model_h_kb:.1f} KB)')
except Exception as e:
    print(f'  [ERROR] micromlgen falló: {e}')
    sys.exit(1)

# ── 11. Gráficas de evaluación ────────────────────────────────────────────────
print('\nFASE 11 — Gráficas de evaluación')

BG_DARK   = '#0a0f1a'
BG_CARD   = '#111827'
SECONDARY = '#6b7280'
TEXT      = '#e5e7eb'
PRIMARY   = '#10b981'   # verde para RF

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
ax.set_title(f'Random Forest — Test Accuracy: {acc_test:.1%}',
             fontweight='bold', color=TEXT, pad=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
cm_path = os.path.join(EVAL_DIR, 'confusion_matrix_rf.png')
fig.savefig(cm_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(cm_path)}')

# Feature importance (top 15)
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_CARD)
top_n = 15
top_idx = sorted_idx[:top_n]
ax.barh(range(top_n), importances[top_idx][::-1], color=PRIMARY, alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels([FEATURE_NAMES[i] for i in top_idx[::-1]], fontsize=9)
ax.set_xlabel('Importancia (Gini)')
ax.set_title('Random Forest — Top 15 Features por Importancia',
             fontweight='bold', color=TEXT)
ax.grid(True, axis='x', alpha=0.4)
plt.tight_layout()
fi_path = os.path.join(EVAL_DIR, 'feature_importance_rf.png')
fig.savefig(fi_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(fi_path)}')

# Per-class bar chart
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
fig.suptitle('Random Forest — Per-Class Metrics', fontsize=14,
             fontweight='bold', color=TEXT)
plt.tight_layout()
metrics_path = os.path.join(EVAL_DIR, 'per_class_metrics_rf.png')
fig.savefig(metrics_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(metrics_path)}')

# ── 12. Reporte Markdown ──────────────────────────────────────────────────────
print('\nFASE 12 — Reporte Markdown')

report_str = classification_report(y_test, y_pred_test, target_names=class_names)
top_features_md = '\n'.join(
    f'| {rank+1} | `{FEATURE_NAMES[i]}` | {importances[i]:.4f} |'
    for rank, i in enumerate(sorted_idx[:10])
)

md = f"""# Reporte de Evaluación — Random Forest

## Resumen
- **Algoritmo:** Random Forest Classifier
- **Árboles:** {N_ESTIMATORS}   |   **Profundidad máx.:** {MAX_DEPTH}
- **Features:** {N_FEATURES} (5 estadísticos × 6 ejes + 2 SMA)
- **Ventana:** {WINDOW_SIZE} muestras @ 100 Hz ({WINDOW_SIZE * 10} ms)
- **Dependencia de TFLite:** ninguna (C++ puro)
- **Formato de exportación:** `model_rf.h` (Arduino C++)

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

## Top 10 Features más Importantes
| # | Feature | Importancia |
|---|---------|-------------|
{top_features_md}

## Tamaño del modelo C++
| Archivo | Tamaño |
|---------|--------|
| `model_rf.h` | {model_h_kb:.1f} KB (texto, no compilado) |

## Archivos Generados
- `confusion_matrix_rf.png`
- `feature_importance_rf.png`
- `per_class_metrics_rf.png`
"""

report_path = os.path.join(EVAL_DIR, 'report.md')
with open(report_path, 'w') as f:
    f.write(md)
print(f'  Guardado: {os.path.relpath(report_path)}')

# ── Resumen final ─────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('Entrenamiento RF completado.')
print('=' * 60)
print(f'  Test accuracy : {acc_test*100:.2f}%')
print(f'  CV accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%')
print(f'  model_rf.h    : {model_h_kb:.1f} KB')
print()
print('Próximo paso — copiar model_rf.h al sketch Arduino:')
print(f'  cp models/rf/model_rf.h arduino/tinyml_ble_rf/model_rf.h')
