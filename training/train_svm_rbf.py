#!/usr/bin/env python3
"""
Entrenador SVM RBF — Clasificador de gestos IMU.

Rol en el proyecto: clasificador no-lineal de máximo margen (Prioridad 1).
  - SVC(kernel='rbf') proyecta los 32 features a un espacio de alta dimensión
    donde la separación lineal es más factible — ideal para CIRCULO vs LADO,
    cuyas distribuciones marginales se solapan pero se separan en el espacio conjunto.
  - Sin dependencia de TFLite Micro
  - Latencia de inferencia < 2 ms
  - Candidato a competir con CNN 1D (94–97% estimado según análisis de viabilidad)

Pipeline:
  1. Ventanas deslizantes idénticas a CNN/RF/LR/NB/SVM_L (WINDOW_SIZE=100, STEP_SIZE=50)
  2. Extracción de 32 features estadísticas por ventana
  3. StandardScaler sobre los 32 features (SVC es sensible a la escala)
  4. SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
  5. Exporta model_svm_rbf.h (C++ para Arduino via micromlgen)

Nota de rendimiento:
  - Entrenamiento del modelo final: ~1–3 minutos
  - Validación cruzada 5-fold: ~5–15 minutos (paralelizable con n_jobs=-1)

Exporta: model_svm_rbf.h (Arduino C++)
Artefactos: predictions_svm_rbf.npz, class_names_svm_rbf.json,
            scaler_params_svm_rbf.json, sv_info_svm_rbf.json
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

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline as SKPipeline

# ── Reproducibilidad ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
MODEL_DIR  = os.path.join(SCRIPT_DIR, '..', 'models', 'svm_rbf')
EVAL_DIR   = os.path.join(SCRIPT_DIR, '..', 'eval', 'svm_rbf')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ── Hiperparámetros ───────────────────────────────────────────────────────────
FEATURE_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
WINDOW_SIZE  = 100
STEP_SIZE    = 50
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15
C_REG        = 10.0
GAMMA        = 'scale'

# ── Nombres de features (32 en total) ─────────────────────────────────────────
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
    feats = []
    for i in range(len(FEATURE_COLS)):
        col = window[:, i]
        feats.append(float(np.mean(col)))
        feats.append(float(np.std(col)))
        feats.append(float(np.min(col)))
        feats.append(float(np.max(col)))
        feats.append(float(np.sqrt(np.mean(col ** 2))))
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

# ── 5a. Escalado de features ──────────────────────────────────────────────────
print('\nFASE 5a — StandardScaler (fit en train, transform en todos los splits)')

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
print(f'  Scaler ajustado. Mean[0]={scaler.mean_[0]:.4f}, Std[0]={scaler.scale_[0]:.4f}')

# ── 5b. Entrenamiento ─────────────────────────────────────────────────────────
print(f"\nFASE 5b — Entrenamiento SVC(kernel='rbf', C={C_REG}, gamma='{GAMMA}')")
print('  Esto puede tomar 1–3 minutos en este dataset...')

clf = SVC(
    kernel='rbf',
    C=C_REG,
    gamma=GAMMA,
    class_weight='balanced',
    random_state=RANDOM_SEED,
)
clf.fit(X_train_s, y_train)
print('  Entrenamiento completado.')
n_sv = int(clf.n_support_.sum())
print(f'  Support vectors totales: {n_sv}')
for i, cn in enumerate(class_names):
    print(f'    {cn}: {clf.n_support_[i]} SVs')

# ── 6. Evaluación ─────────────────────────────────────────────────────────────
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

# ── 7. Validación cruzada ─────────────────────────────────────────────────────
print('FASE 7 — Validación cruzada (5-fold, scaler dentro del fold)')
print('  Esto puede tomar 5–15 minutos con n_jobs=-1...')

cv_pipe = SKPipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',
        C=C_REG, gamma=GAMMA,
        class_weight='balanced', random_state=RANDOM_SEED)),
])
cv_scores = cross_val_score(
    cv_pipe,
    np.vstack([X_train, X_val]),
    np.hstack([y_train, y_val]),
    cv=5, scoring='accuracy', n_jobs=-1)
print(f'  CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%')

# ── 8. Guardar artefactos ─────────────────────────────────────────────────────
print('\nFASE 8 — Guardar artefactos')

json.dump(class_names,
          open(os.path.join(MODEL_DIR, 'class_names_svm_rbf.json'), 'w'), indent=2)
print(f'  Guardado: class_names_svm_rbf.json')

np.savez(os.path.join(MODEL_DIR, 'predictions_svm_rbf.npz'),
         y_test=y_test, y_pred=y_pred_test,
         class_names=np.array(class_names))
print(f'  Guardado: predictions_svm_rbf.npz')

json.dump({'feature_names': FEATURE_NAMES,
           'mean': scaler.mean_.tolist(),
           'std':  scaler.scale_.tolist()},
          open(os.path.join(MODEL_DIR, 'scaler_params_svm_rbf.json'), 'w'), indent=2)
print(f'  Guardado: scaler_params_svm_rbf.json')

train_per_class = {cn: int(np.sum(y_train == i)) for i, cn in enumerate(class_names)}
sv_per_class    = {class_names[i]: int(clf.n_support_[i]) for i in range(len(class_names))}
sv_fraction     = {cn: sv_per_class[cn] / train_per_class[cn] for cn in class_names}
sv_info = {
    'n_support_vectors': n_sv,
    'n_support_per_class': sv_per_class,
    'n_train_per_class': train_per_class,
    'sv_fraction': sv_fraction,
    'C': C_REG,
    'gamma': GAMMA,
}
json.dump(sv_info,
          open(os.path.join(MODEL_DIR, 'sv_info_svm_rbf.json'), 'w'), indent=2)
print(f'  Guardado: sv_info_svm_rbf.json')

# ── 9. Exportar modelo C++ para Arduino ──────────────────────────────────────
print('\nFASE 9 — Exportar modelo C++ para Arduino (micromlgen)')
print('  SVC(kernel=rbf) es soportado nativamente por micromlgen.')
print('  El modelo C++ computa distancias kernel contra todos los support vectors.')

model_h_path = os.path.join(MODEL_DIR, 'model_svm_rbf.h')
try:
    from micromlgen import port

    classmap = {i: name for i, name in enumerate(class_names)}
    c_code = port(clf, classname='SVMRBFModel', classmap=classmap)
    with open(model_h_path, 'w') as f:
        f.write(c_code)
    model_h_kb = os.path.getsize(model_h_path) / 1024
    print(f'  Guardado: {model_h_path} ({model_h_kb:.2f} KB)')
    data_kb = n_sv * N_FEATURES * 4 / 1024
    print(f'  Estimación de datos puros (SVs): {n_sv} × {N_FEATURES} × 4 bytes = {data_kb:.1f} KB')
except Exception as e:
    print(f'  [ERROR] micromlgen falló: {e}')
    sys.exit(1)

# ── 10. Gráficas de evaluación ────────────────────────────────────────────────
print('\nFASE 10 — Gráficas de evaluación')

BG_DARK   = '#0a0f1a'
BG_CARD   = '#111827'
SECONDARY = '#6b7280'
TEXT      = '#e5e7eb'
PRIMARY   = '#ec4899'   # rosa/magenta para SVM RBF

plt.rcParams.update({
    'figure.facecolor': BG_DARK, 'axes.facecolor': BG_CARD,
    'axes.edgecolor': SECONDARY, 'axes.labelcolor': TEXT,
    'text.color': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT,
    'grid.color': '#1f2937', 'grid.alpha': 0.5,
    'font.family': 'monospace', 'font.size': 11,
    'legend.facecolor': BG_CARD, 'legend.edgecolor': SECONDARY,
})

from matplotlib.colors import LinearSegmentedColormap

# ── Confusion matrix ──────────────────────────────────────────────────────────
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
ax.set_title(f'SVM RBF — Test Accuracy: {acc_test:.1%}',
             fontweight='bold', color=TEXT, pad=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
cm_path = os.path.join(EVAL_DIR, 'confusion_matrix_svm_rbf.png')
fig.savefig(cm_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(cm_path)}')

# ── Per-class metrics ─────────────────────────────────────────────────────────
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
fig.suptitle('SVM RBF — Per-Class Metrics', fontsize=14,
             fontweight='bold', color=TEXT)
plt.tight_layout()
metrics_path = os.path.join(EVAL_DIR, 'per_class_metrics_svm_rbf.png')
fig.savefig(metrics_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(metrics_path)}')

# ── Support Vector Distribution — específico para SVM RBF ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG_DARK)

ax = axes[0]
ax.set_facecolor(BG_CARD)
sv_counts = [sv_per_class[cn] for cn in class_names]
bars = ax.barh(class_names, sv_counts, color=PRIMARY, alpha=0.9)
for bar in bars:
    v = bar.get_width()
    ax.text(v + max(sv_counts) * 0.02, bar.get_y() + bar.get_height() / 2,
            f'{v:,}', va='center', fontsize=10, color=TEXT)
ax.set_xlabel('Support Vectors')
ax.set_title('SVs por clase (absoluto)', fontweight='bold', color=TEXT)
ax.set_xlim(0, max(sv_counts) * 1.20)
ax.grid(True, axis='x', alpha=0.4)
ax.invert_yaxis()

ax = axes[1]
ax.set_facecolor(BG_CARD)
sv_fracs = [sv_fraction[cn] * 100 for cn in class_names]
bars = ax.barh(class_names, sv_fracs, color=PRIMARY, alpha=0.9)
for bar in bars:
    v = bar.get_width()
    ax.text(v + max(sv_fracs) * 0.02, bar.get_y() + bar.get_height() / 2,
            f'{v:.1f}%', va='center', fontsize=10, color=TEXT)
ax.set_xlabel('% de muestras de entrenamiento')
ax.set_title('SVs como fracción del train (% por clase)', fontweight='bold', color=TEXT)
ax.set_xlim(0, max(sv_fracs) * 1.20)
ax.grid(True, axis='x', alpha=0.4)
ax.invert_yaxis()

fig.suptitle(f'SVM RBF — Distribución de Support Vectors (total: {n_sv}, C={C_REG})',
             fontsize=14, fontweight='bold', color=TEXT)
plt.tight_layout()
sv_path = os.path.join(EVAL_DIR, 'sv_distribution_svm_rbf.png')
fig.savefig(sv_path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close()
print(f'  Guardado: {os.path.relpath(sv_path)}')

# ── 11. Reporte Markdown ──────────────────────────────────────────────────────
print('\nFASE 11 — Reporte Markdown')

report_str = classification_report(y_test, y_pred_test, target_names=class_names)

sv_rows = '\n'.join(
    f'| {cn} | {sv_per_class[cn]} | {train_per_class[cn]} | {sv_fraction[cn]*100:.1f}% |'
    for cn in class_names
)

md = f"""# Reporte de Evaluación — SVM RBF

## Resumen
- **Algoritmo:** SVC(kernel='rbf') — máximo margen no-lineal
- **Hiperparámetros:** C={C_REG}, gamma='{GAMMA}', class_weight='balanced'
- **Support vectors totales:** {n_sv}
- **Preprocesamiento:** StandardScaler sobre los 32 features (fit en train)
- **Features:** {N_FEATURES} (5 estadísticos × 6 ejes + 2 SMA)
- **Ventana:** {WINDOW_SIZE} muestras @ 100 Hz ({WINDOW_SIZE * 10} ms)
- **Rol:** Clasificador no-lineal de máximo margen (Prioridad 1 en análisis de viabilidad)
- **Dependencia de TFLite:** ninguna (C++ puro via micromlgen)
- **Formato:** `model_svm_rbf.h` (Arduino C++)
- **Nota de firmware:** aplicar Z-score con `scaler_params_svm_rbf.json` antes de `SVMRBFModel::predict()`

## Resultados
| Conjunto | Accuracy |
|----------|----------|
| Train | {acc_train*100:.2f}% |
| Val   | {acc_val*100:.2f}% |
| Test  | {acc_test*100:.2f}% |
| CV 5-fold | {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}% |

## Support Vectors por Clase
| Clase | SVs | Total Train | Fracción |
|-------|-----|-------------|----------|
{sv_rows}

## Classification Report
```
{report_str}
```

## Tamaño del modelo C++
| Archivo | Tamaño |
|---------|--------|
| `model_svm_rbf.h` | {model_h_kb:.2f} KB (texto, no compilado) |
| `scaler_params_svm_rbf.json` | < 5 KB |

## Archivos Generados
- `confusion_matrix_svm_rbf.png`
- `per_class_metrics_svm_rbf.png`
- `sv_distribution_svm_rbf.png`
"""

report_path = os.path.join(EVAL_DIR, 'report.md')
with open(report_path, 'w') as f:
    f.write(md)
print(f'  Guardado: {os.path.relpath(report_path)}')

# ── Resumen final ─────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('Entrenamiento SVM RBF completado.')
print('=' * 60)
print(f'  Test accuracy  : {acc_test*100:.2f}%')
print(f'  CV accuracy    : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%')
print(f'  Support vectors: {n_sv}')
print(f'  model_svm_rbf.h: {model_h_kb:.2f} KB')
print()
print('Próximos pasos:')
print(f'  python3 eval/generate_eval.py --model svm_rbf')
print(f'  python3 eval/generate_eval.py --model cnn rf lr nb svm_l svm_rbf')
