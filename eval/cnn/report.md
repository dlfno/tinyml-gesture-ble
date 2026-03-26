# Reporte de Evaluación — CNN 1D

## Resumen del Entrenamiento
- **Arquitectura:** Conv1D 16→32→64 + GlobalAvgPool + Dense(32) → Dense(4)
- **Epochs entrenados:** 50 (mejor epoch: 35)
- **Mejor val_loss:** 0.0794
- **Mejor val_accuracy:** 97.68%
- **Window size:** 100 muestras @ 100 Hz (1000 ms)
- **BLE device name:** `TinyML-Sense`

## Tamaño del Modelo
| Versión | Tamaño | Reducción |
|---------|--------|-----------|
| Float32 | 52.1 KB | — |
| Quantized | 26.9 KB | 48.4% |

## Accuracy en Test Set
| Modelo | Accuracy | Diferencia |
|--------|----------|------------|
| Float32 | 97.93% | — |
| Quantized | 97.59% | -0.34 pp |

## Classification Report — Float32
```
              precision    recall  f1-score   support

     CIRCULO       0.99      0.97      0.98       934
     DEFAULT       0.94      0.89      0.91       150
        LADO       0.96      0.99      0.97       933
      QUIETO       1.00      0.99      1.00       933

    accuracy                           0.98      2950
   macro avg       0.97      0.96      0.96      2950
weighted avg       0.98      0.98      0.98      2950

```

## Classification Report — Quantized
```
              precision    recall  f1-score   support

     CIRCULO       0.99      0.97      0.98       934
     DEFAULT       0.93      0.89      0.91       150
        LADO       0.95      0.99      0.97       933
      QUIETO       1.00      0.98      0.99       933

    accuracy                           0.98      2950
   macro avg       0.97      0.96      0.96      2950
weighted avg       0.98      0.98      0.98      2950

```

## Archivos Generados
- `confusion_matrix_float.png`
- `confusion_matrix_quant.png`
- `confusion_matrix_comparison.png`
- `training_loss.png`
- `training_accuracy.png`
- `training_combined.png`
- `quantization_impact.png`