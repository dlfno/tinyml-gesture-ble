#!/usr/bin/env python3
"""
Script de Evaluación — TinyML Gesture Classifier
Genera visualizaciones, reportes y comparativa de modelos.

Prerequisito: ejecutar el script de entrenamiento correspondiente primero.

Uso:
    python eval/generate_eval.py --model cnn        # evalúa CNN (default)
    python eval/generate_eval.py --model rf         # evalúa Random Forest
    python eval/generate_eval.py --model cnn rf     # evalúa ambos y genera comparativa
"""
import argparse
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score
)

# ── Rutas base ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, '..')

# ── Configuración por modelo ──────────────────────────────────────────────────
MODEL_CONFIGS = {
    'cnn': {
        'label':         'CNN 1D',
        'model_dir':     os.path.join(ROOT_DIR, 'models', 'cnn'),
        'eval_dir':      os.path.join(SCRIPT_DIR, 'cnn'),
        'predictions':   'predictions.npz',
        'history':       'training_history.json',
        'tflite_float':  'model_float.tflite',
        'tflite_quant':  'model_quantized.tflite',
        'primary':       '#0050cb',
        'quant_color':   '#f59e0b',
        'arch_summary':  'Conv1D 16→32→64 + GlobalAvgPool + Dense(32) → Dense(4)',
        'window_ms':     1000,
        'ble_name':      'TinyML-Sense',
    },
    'rf': {
        'label':        'Random Forest',
        'model_dir':    os.path.join(ROOT_DIR, 'models', 'rf'),
        'eval_dir':     os.path.join(SCRIPT_DIR, 'rf'),
        'predictions':  'predictions_rf.npz',
        'model_h':      'model_rf.h',
        'primary':      '#10b981',
        'arch_summary': 'RF(n_estimators=20, max_depth=8) + 32 features estadísticas',
        'window_ms':    1000,
        'ble_name':     'TinyML-RF',
        'model_type':   'rf',
    },
    'lr': {
        'label':        'Regresión Logística',
        'model_dir':    os.path.join(ROOT_DIR, 'models', 'lr'),
        'eval_dir':     os.path.join(SCRIPT_DIR, 'lr'),
        'predictions':  'predictions_lr.npz',
        'model_h':      'model_lr.h',
        'primary':      '#a855f7',
        'arch_summary': 'LogisticRegression(C=1.0, max_iter=5000) + StandardScaler + 32 features estadísticas',
        'window_ms':    1000,
        'ble_name':     'TinyML-LR',
        'model_type':   'lr',
    },
    'nb': {
        'label':        'Gaussian Naive Bayes',
        'model_dir':    os.path.join(ROOT_DIR, 'models', 'nb'),
        'eval_dir':     os.path.join(SCRIPT_DIR, 'nb'),
        'predictions':  'predictions_nb.npz',
        'model_h':      'model_nb.h',
        'primary':      '#f97316',
        'arch_summary': 'GaussianNB() + 32 features estadísticas (sin scaler)',
        'window_ms':    1000,
        'ble_name':     'TinyML-NB',
        'model_type':   'nb',
    },
    'svm_l': {
        'label':        'SVM Lineal',
        'model_dir':    os.path.join(ROOT_DIR, 'models', 'svm_l'),
        'eval_dir':     os.path.join(SCRIPT_DIR, 'svm_l'),
        'predictions':  'predictions_svm_linear.npz',
        'model_h':      'model_svm_linear.h',
        'primary':      '#06b6d4',
        'arch_summary': 'LinearSVC(C=1.0, max_iter=5000) + StandardScaler + 32 features estadísticas',
        'window_ms':    1000,
        'ble_name':     'TinyML-SVML',
        'model_type':   'svm_l',
    },
    'svm_rbf': {
        'label':        'SVM RBF',
        'model_dir':    os.path.join(ROOT_DIR, 'models', 'svm_rbf'),
        'eval_dir':     os.path.join(SCRIPT_DIR, 'svm_rbf'),
        'predictions':  'predictions_svm_rbf.npz',
        'model_h':      'model_svm_rbf.h',
        'primary':      '#ec4899',
        'arch_summary': "SVC(kernel='rbf', C=10, gamma='scale') + StandardScaler + 32 features estadísticas",
        'window_ms':    1000,
        'ble_name':     'TinyML-SVMR',
        'model_type':   'svm_rbf',
    },
}

# ── Paleta Blueprint Ethos ────────────────────────────────────────────────────
BG_DARK       = '#0a0f1a'
BG_CARD       = '#111827'
SECONDARY     = '#6b7280'
TEXT          = '#e5e7eb'
ACCENT_GREEN  = '#10b981'
ACCENT_RED    = '#ef4444'
ACCENT_YELLOW = '#f59e0b'

plt.rcParams.update({
    'figure.facecolor': BG_DARK,
    'axes.facecolor':   BG_CARD,
    'axes.edgecolor':   SECONDARY,
    'axes.labelcolor':  TEXT,
    'text.color':       TEXT,
    'xtick.color':      TEXT,
    'ytick.color':      TEXT,
    'grid.color':       '#1f2937',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
    'font.size':        11,
    'legend.facecolor': BG_CARD,
    'legend.edgecolor': SECONDARY,
})


# ── Helpers ───────────────────────────────────────────────────────────────────
def _required_files(cfg: dict) -> list[str]:
    if cfg.get('model_type') in ('rf', 'lr', 'nb', 'svm_l', 'svm_rbf'):
        return [cfg['predictions']]
    return [cfg['predictions'], cfg['history'], cfg['tflite_float'], cfg['tflite_quant']]


def check_artifacts(cfg: dict) -> bool:
    """Verifica que existan todos los artefactos necesarios. Retorna False si falta alguno."""
    missing = []
    for fname in _required_files(cfg):
        path = os.path.join(cfg['model_dir'], fname)
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        for p in missing:
            print(f"  [MISSING] {p}")
        return False
    return True


def load_artifacts(cfg: dict) -> dict:
    """Carga predictions y tamaños de modelo. Para RF no hay history ni TFLite."""
    mdir = cfg['model_dir']
    preds = np.load(os.path.join(mdir, cfg['predictions']), allow_pickle=True)

    if cfg.get('model_type') in ('rf', 'lr', 'nb', 'svm_l', 'svm_rbf'):
        model_h_file = cfg.get('model_h', 'model_rf.h')
        model_h_kb = os.path.getsize(os.path.join(mdir, model_h_file)) / 1024
        y_pred     = preds['y_pred']
        acc        = accuracy_score(preds['y_test'], y_pred)
        return {
            'history':      None,
            'y_test':       preds['y_test'],
            'y_pred_float': y_pred,
            'y_pred_quant': y_pred,
            'class_names':  [str(c) for c in preds['class_names']],
            'float_kb':     model_h_kb,
            'quant_kb':     model_h_kb,
            'compression':  None,
            'acc_float':    acc,
            'acc_quant':    acc,
        }

    history  = json.load(open(os.path.join(mdir, cfg['history'])))
    float_kb = os.path.getsize(os.path.join(mdir, cfg['tflite_float'])) / 1024
    quant_kb = os.path.getsize(os.path.join(mdir, cfg['tflite_quant'])) / 1024

    return {
        'history':      history,
        'y_test':       preds['y_test'],
        'y_pred_float': preds['y_pred_float'],
        'y_pred_quant': preds['y_pred_quant'],
        'class_names':  [str(c) for c in preds['class_names']],
        'float_kb':     float_kb,
        'quant_kb':     quant_kb,
        'compression':  (1 - quant_kb / float_kb) * 100,
        'acc_float':    accuracy_score(preds['y_test'], preds['y_pred_float']),
        'acc_quant':    accuracy_score(preds['y_test'], preds['y_pred_quant']),
    }


def _draw_cm(ax, y_true, y_pred, names, subtitle, color):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    n       = len(names)
    ax.set_facecolor(BG_CARD)
    cmap = LinearSegmentedColormap.from_list('c', [BG_CARD, color], N=256)
    im   = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            cnt  = cm[i, j]
            pct  = cm_norm[i, j] * 100
            c    = 'white' if pct > 50 else TEXT
            w    = 'bold'  if i == j  else 'normal'
            ax.text(j, i, f'{cnt}\n({pct:.1f}%)',
                    ha='center', va='center', color=c, fontsize=9, fontweight=w)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Predicted', fontweight='bold', color=TEXT)
    ax.set_ylabel('True',      fontweight='bold', color=TEXT)
    ax.set_title(subtitle, fontsize=12, fontweight='bold', color=TEXT, pad=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    acc = accuracy_score(y_true, y_pred)
    ax.text(0.5, -0.14, f'Accuracy: {acc:.1%}',
            transform=ax.transAxes, ha='center', fontsize=11,
            color=ACCENT_GREEN, fontweight='bold')


def _save(fig, eval_dir, filename):
    os.makedirs(eval_dir, exist_ok=True)
    path = os.path.join(eval_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Guardado: {os.path.relpath(path)}")


# ── Generadores de plots ──────────────────────────────────────────────────────
def plot_confusion_matrices(data: dict, cfg: dict) -> None:
    primary      = cfg['primary']
    quant_color  = cfg.get('quant_color', cfg['primary'])
    eval_dir     = cfg['eval_dir']
    label        = cfg['label']
    names        = data['class_names']

    # Float individual
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG_DARK)
    _draw_cm(ax, data['y_test'], data['y_pred_float'], names,
             f'{label} — Float32', primary)
    plt.tight_layout()
    _save(fig, eval_dir, 'confusion_matrix_float.png')

    # Quant individual
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG_DARK)
    _draw_cm(ax, data['y_test'], data['y_pred_quant'], names,
             f'{label} — INT8 Quantized', quant_color)
    plt.tight_layout()
    _save(fig, eval_dir, 'confusion_matrix_quant.png')

    # Comparativa lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_DARK)
    _draw_cm(axes[0], data['y_test'], data['y_pred_float'], names, 'Float32',    primary)
    _draw_cm(axes[1], data['y_test'], data['y_pred_quant'], names, 'INT8 Quant', primary)
    fig.suptitle(
        f'{label} — Float: {data["acc_float"]:.1%}  vs  INT8: {data["acc_quant"]:.1%}',
        fontsize=13, fontweight='bold', color=TEXT, y=1.01
    )
    plt.tight_layout()
    _save(fig, eval_dir, 'confusion_matrix_comparison.png')


def plot_training_curves(data: dict, cfg: dict) -> None:
    h        = data['history']
    primary  = cfg['primary']
    label    = cfg['label']
    eval_dir = cfg['eval_dir']

    loss     = h.get('loss', [])
    val_loss = h.get('val_loss', [])
    acc      = h.get('accuracy', [])
    val_acc  = h.get('val_accuracy', [])
    lr       = h.get('lr', [])
    epochs   = range(1, len(loss) + 1)

    best_loss = int(np.argmin(val_loss)) if val_loss else 0
    best_acc  = int(np.argmax(val_acc))  if val_acc  else 0
    has_lr    = len(lr) > 0

    # Combinado
    n_panels = 3 if has_lr else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.patch.set_facecolor(BG_DARK)

    ax = axes[0]
    ax.set_facecolor(BG_CARD)
    ax.plot(epochs, loss,     color=primary,       lw=2, label='Train Loss')
    ax.plot(epochs, val_loss, color=ACCENT_YELLOW, lw=2, label='Val Loss')
    ax.axvline(best_loss + 1, color=ACCENT_GREEN, ls='--', lw=1.5,
               label=f'Best epoch {best_loss + 1}')
    ax.plot(best_loss + 1, val_loss[best_loss], 'o', color=ACCENT_GREEN, ms=8, zorder=5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold', color=TEXT)
    ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.set_facecolor(BG_CARD)
    ax.plot(epochs, acc,     color=primary,       lw=2, label='Train Acc')
    ax.plot(epochs, val_acc, color=ACCENT_YELLOW, lw=2, label='Val Acc')
    ax.axvline(best_acc + 1, color=ACCENT_GREEN, ls='--', lw=1.5,
               label=f'Best epoch {best_acc + 1}')
    ax.plot(best_acc + 1, val_acc[best_acc], 'o', color=ACCENT_GREEN, ms=8, zorder=5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy', fontweight='bold', color=TEXT)
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True)

    if has_lr:
        ax = axes[2]
        ax.set_facecolor(BG_CARD)
        ax.plot(epochs, lr, color=cfg['primary'], lw=2)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate', fontweight='bold', color=TEXT)
        ax.grid(True)

    fig.suptitle(f'{label} — Training History', fontsize=14, fontweight='bold', color=TEXT)
    plt.tight_layout()
    _save(fig, eval_dir, 'training_combined.png')

    # Loss individual
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_DARK); ax.set_facecolor(BG_CARD)
    ax.plot(epochs, loss,     color=primary,       lw=2, label='Train Loss')
    ax.plot(epochs, val_loss, color=ACCENT_YELLOW, lw=2, label='Val Loss')
    ax.axvline(best_loss + 1, color=ACCENT_GREEN, ls='--', lw=1.5)
    ax.plot(best_loss + 1, val_loss[best_loss], 'o', color=ACCENT_GREEN, ms=8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'{label} — Training Loss', fontweight='bold', color=TEXT)
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    _save(fig, eval_dir, 'training_loss.png')

    # Accuracy individual
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_DARK); ax.set_facecolor(BG_CARD)
    ax.plot(epochs, acc,     color=primary,       lw=2, label='Train Acc')
    ax.plot(epochs, val_acc, color=ACCENT_YELLOW, lw=2, label='Val Acc')
    ax.axvline(best_acc + 1, color=ACCENT_GREEN, ls='--', lw=1.5)
    ax.plot(best_acc + 1, val_acc[best_acc], 'o', color=ACCENT_GREEN, ms=8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{label} — Training Accuracy', fontweight='bold', color=TEXT)
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True)
    plt.tight_layout()
    _save(fig, eval_dir, 'training_accuracy.png')


def plot_quantization_impact(data: dict, cfg: dict) -> None:
    names    = data['class_names']
    primary  = cfg['primary']
    label    = cfg['label']
    eval_dir = cfg['eval_dir']

    mf = precision_recall_fscore_support(data['y_test'], data['y_pred_float'], average=None)
    mq = precision_recall_fscore_support(data['y_test'], data['y_pred_quant'], average=None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(BG_DARK)
    x     = np.arange(len(names))
    width = 0.35

    for ax, (mname, vf, vq) in zip(axes, [
        ('Precision', mf[0], mq[0]),
        ('Recall',    mf[1], mq[1]),
        ('F1-Score',  mf[2], mq[2]),
    ]):
        ax.set_facecolor(BG_CARD)
        bf = ax.bar(x - width / 2, vf, width, label='Float32',   color=primary,       alpha=0.9)
        bq = ax.bar(x + width / 2, vq, width, label='INT8 Quant', color=ACCENT_YELLOW, alpha=0.9)
        for bar in [*bf, *bq]:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=9, color=TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(mname, fontweight='bold', color=TEXT)
        ax.set_ylabel(mname)
        ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.4)

    fig.suptitle(f'{label} — Quantization Impact: Float32 vs INT8',
                 fontsize=14, fontweight='bold', color=TEXT)
    plt.tight_layout()
    _save(fig, eval_dir, 'quantization_impact.png')


# ── Reporte Markdown ──────────────────────────────────────────────────────────
def _classic_artifacts_md(cfg: dict) -> str:
    """Retorna la lista de archivos generados según el tipo de modelo clásico."""
    if cfg.get('model_type') == 'rf':
        return (
            "- `confusion_matrix_rf.png`\n"
            "- `feature_importance_rf.png`\n"
            "- `per_class_metrics_rf.png`"
        )
    if cfg.get('model_type') == 'lr':
        return (
            "- `confusion_matrix_lr.png`\n"
            "- `per_class_metrics_lr.png`\n"
            "- `coefficients_lr.png`"
        )
    if cfg.get('model_type') == 'nb':
        return (
            "- `confusion_matrix_nb.png`\n"
            "- `per_class_metrics_nb.png`\n"
            "- `class_means_nb.png`"
        )
    if cfg.get('model_type') == 'svm_l':
        return (
            "- `confusion_matrix_svm_linear.png`\n"
            "- `per_class_metrics_svm_linear.png`\n"
            "- `coefficients_svm_linear.png`"
        )
    if cfg.get('model_type') == 'svm_rbf':
        return (
            "- `confusion_matrix_svm_rbf.png`\n"
            "- `per_class_metrics_svm_rbf.png`\n"
            "- `sv_distribution_svm_rbf.png`"
        )
    return ""


def generate_report(data: dict, cfg: dict) -> None:
    eval_dir = cfg['eval_dir']
    label    = cfg['label']
    names    = data['class_names']
    is_classic = cfg.get('model_type') in ('rf', 'lr', 'nb', 'svm_l', 'svm_rbf')

    report_str = classification_report(data['y_test'], data['y_pred_float'], target_names=names)

    if is_classic:
        model_h_file = cfg.get('model_h', 'model_rf.h')
        md = f"""# Reporte de Evaluación — {label}

## Resumen
- **Arquitectura:** {cfg['arch_summary']}
- **Formato de despliegue:** C++ header (`{model_h_file}`) — sin dependencia de TFLite
- **Window size:** 100 muestras @ 100 Hz ({cfg['window_ms']} ms)
- **BLE device name:** `{cfg['ble_name']}`

## Tamaño del Modelo
| Archivo | Tamaño | Notas |
|---------|--------|-------|
| `{model_h_file}` | {data['float_kb']:.1f} KB | Texto C++; compilado según modelo |

## Accuracy en Test Set
| Accuracy |
|----------|
| {data['acc_float'] * 100:.2f}% |

## Classification Report
```
{report_str}
```

## Archivos Generados
{_classic_artifacts_md(cfg)}"""
    else:
        h       = data['history']
        val_loss = h.get('val_loss', [])
        val_acc  = h.get('val_accuracy', [])
        n_epochs = len(h.get('loss', []))
        best_ep       = int(np.argmax(val_acc)) + 1 if val_acc else '—'
        best_val_loss = f"{min(val_loss):.4f}"       if val_loss else 'N/A'
        best_val_acc  = f"{max(val_acc) * 100:.2f}%" if val_acc  else 'N/A'
        report_quant  = classification_report(data['y_test'], data['y_pred_quant'], target_names=names)
        comp_str      = f"{data['compression']:.1f}%" if data['compression'] is not None else "—"

        md = f"""# Reporte de Evaluación — {label}

## Resumen del Entrenamiento
- **Arquitectura:** {cfg['arch_summary']}
- **Epochs entrenados:** {n_epochs} (mejor epoch: {best_ep})
- **Mejor val_loss:** {best_val_loss}
- **Mejor val_accuracy:** {best_val_acc}
- **Window size:** 100 muestras @ 100 Hz ({cfg['window_ms']} ms)
- **BLE device name:** `{cfg['ble_name']}`

## Tamaño del Modelo
| Versión | Tamaño | Reducción |
|---------|--------|-----------|
| Float32 | {data['float_kb']:.1f} KB | — |
| Quantized | {data['quant_kb']:.1f} KB | {comp_str} |

## Accuracy en Test Set
| Modelo | Accuracy | Diferencia |
|--------|----------|------------|
| Float32 | {data['acc_float'] * 100:.2f}% | — |
| Quantized | {data['acc_quant'] * 100:.2f}% | {(data['acc_quant'] - data['acc_float']) * 100:+.2f} pp |

## Classification Report — Float32
```
{report_str}
```

## Classification Report — Quantized
```
{report_quant}
```

## Archivos Generados
- `confusion_matrix_float.png`
- `confusion_matrix_quant.png`
- `confusion_matrix_comparison.png`
- `training_loss.png`
- `training_accuracy.png`
- `training_combined.png`
- `quantization_impact.png`"""

    path = os.path.join(eval_dir, 'report.md')
    os.makedirs(eval_dir, exist_ok=True)
    with open(path, 'w') as f:
        f.write(md)
    print(f"  Guardado: {os.path.relpath(path)}")


# ── Comparativa CNN vs RF ─────────────────────────────────────────────────────
def generate_comparison(results: dict[str, dict]) -> None:
    """Genera eval/model_comparison.md con todos los modelos evaluados."""
    comp_path = os.path.join(SCRIPT_DIR, 'model_comparison.md')

    rows_size = ""
    rows_acc  = ""
    recommendation = ""

    for key, (data, cfg) in results.items():
        if cfg.get('model_type') in ('rf', 'lr', 'nb', 'svm_l', 'svm_rbf'):
            rows_size += f"| {cfg['label']} | C++ header | {data['float_kb']:.1f} KB | — |\n"
            rows_acc  += f"| {cfg['label']} | C++ | {data['acc_float'] * 100:.2f}% |\n"
        else:
            comp_str = f"{data['compression']:.1f}%" if data['compression'] is not None else "—"
            rows_size += (
                f"| {cfg['label']} | Float32 | {data['float_kb']:.1f} KB | — |\n"
                f"| {cfg['label']} | INT8    | {data['quant_kb']:.1f} KB | {comp_str} |\n"
            )
            rows_acc += (
                f"| {cfg['label']} | Float32 | {data['acc_float'] * 100:.2f}% |\n"
                f"| {cfg['label']} | INT8    | {data['acc_quant'] * 100:.2f}% |\n"
            )

    if 'cnn' in results and 'rf' in results:
        cnn_acc = results['cnn'][0]['acc_quant']
        rf_acc  = results['rf'][0]['acc_float']
        cnn_kb  = results['cnn'][0]['quant_kb']
        rf_kb   = results['rf'][0]['float_kb']
        recommendation = f"""
## Recommendation

| Criterion | Winner |
|-----------|--------|
| Accuracy | **CNN** ({cnn_acc * 100:.2f}% INT8 vs {rf_acc * 100:.2f}% RF) |
| Inference latency | **Random Forest** (<1 ms vs ~14 ms) |
| TFLite dependency | **Random Forest** (pure C++, no interpreter) |
| Recommended for production | **CNN** (higher accuracy, smaller compiled footprint) |

> Use **CNN** when accuracy is the priority.
> Use **Random Forest** when flash is critically constrained or inference latency must be <1 ms.
"""

    md = f"""# Model Comparison — TinyML Gesture Classifier

## Model Size
| Model | Format | Size | Compression |
|-------|--------|------|-------------|
{rows_size}
## Test Accuracy
| Model | Format | Accuracy |
|-------|--------|----------|
{rows_acc}
## Evaluation Artifacts
"""
    for key, (data, cfg) in results.items():
        label = cfg['label']
        rel   = os.path.relpath(cfg['eval_dir'], SCRIPT_DIR)
        md += f"\n### {label}\n"
        if cfg.get('model_type') == 'rf':
            for fname in ['confusion_matrix_rf.png', 'feature_importance_rf.png',
                          'per_class_metrics_rf.png', 'report.md']:
                md += f"- [`{rel}/{fname}`]({rel}/{fname})\n"
        elif cfg.get('model_type') == 'lr':
            for fname in ['confusion_matrix_lr.png', 'per_class_metrics_lr.png',
                          'coefficients_lr.png', 'report.md']:
                md += f"- [`{rel}/{fname}`]({rel}/{fname})\n"
        elif cfg.get('model_type') == 'nb':
            for fname in ['confusion_matrix_nb.png', 'per_class_metrics_nb.png',
                          'class_means_nb.png', 'report.md']:
                md += f"- [`{rel}/{fname}`]({rel}/{fname})\n"
        elif cfg.get('model_type') == 'svm_l':
            for fname in ['confusion_matrix_svm_linear.png', 'per_class_metrics_svm_linear.png',
                          'coefficients_svm_linear.png', 'report.md']:
                md += f"- [`{rel}/{fname}`]({rel}/{fname})\n"
        elif cfg.get('model_type') == 'svm_rbf':
            for fname in ['confusion_matrix_svm_rbf.png', 'per_class_metrics_svm_rbf.png',
                          'sv_distribution_svm_rbf.png', 'report.md']:
                md += f"- [`{rel}/{fname}`]({rel}/{fname})\n"
        else:
            for fname in ['confusion_matrix_comparison.png', 'training_combined.png',
                          'quantization_impact.png', 'report.md']:
                md += f"- [`{rel}/{fname}`]({rel}/{fname})\n"

    md += recommendation

    with open(comp_path, 'w') as f:
        f.write(md)
    print(f"\n  Guardado: {os.path.relpath(comp_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate_model(model_key: str) -> tuple[dict, dict] | None:
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'=' * 55}")
    print(f"Evaluando: {cfg['label']}")
    print(f"{'=' * 55}")

    print("Verificando artefactos...")
    if not check_artifacts(cfg):
        label  = cfg['label']
        scripts = {'cnn': 'train_cnn.py', 'rf': 'train_rf.py', 'lr': 'train_lr.py',
                   'nb': 'train_nb.py', 'svm_l': 'train_svm_linear.py',
                   'svm_rbf': 'train_svm_rbf.py'}
        print(f"  ERROR: faltan artefactos para {label}.")
        print(f"  Ejecuta primero: python training/{scripts.get(model_key, 'train_' + model_key + '.py')}")
        return None

    data = load_artifacts(cfg)

    if cfg.get('model_type') in ('rf', 'lr', 'nb', 'svm_l', 'svm_rbf'):
        model_h_file = cfg.get('model_h', 'model_rf.h')
        print(f"  Accuracy    : {data['acc_float'] * 100:.2f}%  ({model_h_file}: {data['float_kb']:.1f} KB)")
        print("\nGenerando plots...")
        plot_confusion_matrices(data, cfg)
        generate_report(data, cfg)
    else:
        print(f"  Float32 acc : {data['acc_float'] * 100:.2f}%  ({data['float_kb']:.1f} KB)")
        comp = f"{data['compression']:.1f}%" if data['compression'] is not None else "—"
        print(f"  Quant acc   : {data['acc_quant'] * 100:.2f}%  ({data['quant_kb']:.1f} KB, {comp} compression)")
        print("\nGenerando plots...")
        plot_confusion_matrices(data, cfg)
        plot_training_curves(data, cfg)
        plot_quantization_impact(data, cfg)
        generate_report(data, cfg)

    return data, cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera visualizaciones y reporte de evaluación del modelo TinyML."
    )
    parser.add_argument(
        '--model',
        nargs='+',
        choices=['cnn', 'rf', 'lr', 'nb', 'svm_l', 'svm_rbf'],
        default=['cnn'],
        help="Modelo(s) a evaluar. Ej: --model cnn rf lr nb svm_l",
    )
    args = parser.parse_args()

    results = {}
    for key in args.model:
        result = evaluate_model(key)
        if result is not None:
            results[key] = result

    if not results:
        print("\nNo se generó ningún reporte. Verifica los artefactos de entrenamiento.")
        sys.exit(1)

    print("\nGenerando model_comparison.md...")
    generate_comparison(results)

    print(f"\n{'=' * 55}")
    print("Evaluación completada.")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    main()
