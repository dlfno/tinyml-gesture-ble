# Reporte de Evaluación — Regresión Logística

## Resumen
- **Arquitectura:** LogisticRegression(C=1.0, max_iter=5000) + StandardScaler + 32 features estadísticas
- **Formato de despliegue:** C++ header (`model_lr.h`) — sin dependencia de TFLite
- **Window size:** 100 muestras @ 100 Hz (1000 ms)
- **BLE device name:** `TinyML-LR`

## Tamaño del Modelo
| Archivo | Tamaño | Notas |
|---------|--------|-------|
| `model_lr.h` | 4.8 KB | Texto C++; compilado según modelo |

## Accuracy en Test Set
| Accuracy |
|----------|
| 90.85% |

## Classification Report
```
              precision    recall  f1-score   support

     CIRCULO       0.92      0.89      0.90       934
     DEFAULT       0.53      0.71      0.61       150
        LADO       0.89      0.88      0.89       933
      QUIETO       0.99      0.99      0.99       933

    accuracy                           0.91      2950
   macro avg       0.83      0.87      0.85      2950
weighted avg       0.91      0.91      0.91      2950

```

## Archivos Generados
- `confusion_matrix_lr.png`
- `per_class_metrics_lr.png`
- `coefficients_lr.png`