# Reporte de Evaluación — Gaussian Naive Bayes

## Resumen
- **Arquitectura:** GaussianNB() + 32 features estadísticas (sin scaler)
- **Formato de despliegue:** C++ header (`model_nb.h`) — sin dependencia de TFLite
- **Window size:** 100 muestras @ 100 Hz (1000 ms)
- **BLE device name:** `TinyML-NB`

## Tamaño del Modelo
| Archivo | Tamaño | Notas |
|---------|--------|-------|
| `model_nb.h` | 8.6 KB | Texto C++; compilado según modelo |

## Accuracy en Test Set
| Accuracy |
|----------|
| 63.93% |

## Classification Report
```
              precision    recall  f1-score   support

     CIRCULO       0.50      0.91      0.65       934
     DEFAULT       0.31      0.46      0.37       150
        LADO       0.80      0.31      0.45       933
      QUIETO       1.00      0.73      0.84       933

    accuracy                           0.64      2950
   macro avg       0.65      0.60      0.58      2950
weighted avg       0.75      0.64      0.63      2950

```

## Archivos Generados
- `confusion_matrix_nb.png`
- `per_class_metrics_nb.png`
- `class_means_nb.png`