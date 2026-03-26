#!/usr/bin/env python3
"""Calcula SCALER_MEAN y SCALER_STD desde todos los CSVs de entrenamiento."""
import os, sys, numpy as np, pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), 'data')
FEATURE_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

all_frames = []
class_counts = {}
for root, _, files in os.walk(BASE_DIR):
    subject = os.path.basename(root)
    for file in sorted(files):
        if not file.endswith('.csv'):
            continue
        label = file.split('_')[0]
        class_counts[label] = class_counts.get(label, 0) + 1
        try:
            df = pd.read_csv(os.path.join(root, file))
            all_frames.append(df[FEATURE_COLS])
        except Exception as e:
            print(f"[WARN] {file}: {e}", file=sys.stderr)

if not all_frames:
    print("ERROR: No se encontraron CSVs en", BASE_DIR, file=sys.stderr)
    sys.exit(1)

combined = pd.concat(all_frames, ignore_index=True)

print(f"Total filas: {len(combined):,}")
print(f"Archivos por clase: {class_counts}")
print()
print("// Para copiar al sketch Arduino:")
print(f"const float SCALER_MEAN[] = {{ {', '.join(f'{v:.4f}f' for v in combined.mean())} }};")
print(f"const float SCALER_STD[]  = {{ {', '.join(f'{v:.4f}f' for v in combined.std())} }};")
print()
print("# Para copiar al script de entrenamiento Python:")
print(f"SCALER_MEAN = {combined.mean().tolist()}")
print(f"SCALER_STD = {combined.std().tolist()}")
