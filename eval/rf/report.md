# Reporte de Evaluación — Random Forest

## Resumen
- **Arquitectura:** RF(n_estimators=20, max_depth=8) + 32 features estadísticas
- **Formato de despliegue:** C++ header (`model_rf.h`) — sin dependencia de TFLite
- **Window size:** 100 muestras @ 100 Hz (1000 ms)
- **BLE device name:** `TinyML-RF`

## Tamaño del Modelo
| Archivo | Tamaño | Notas |
|---------|--------|-------|
| `model_rf.h` | 576.5 KB | Texto C++; compilado ~80-120 KB en ARM |

## Accuracy en Test Set
| Accuracy |
|----------|
| 94.03% |

## Classification Report
```
              precision    recall  f1-score   support

     CIRCULO       0.94      0.93      0.93       934
     DEFAULT       0.59      0.83      0.69       150
        LADO       0.96      0.92      0.94       933
      QUIETO       1.00      0.99      1.00       933

    accuracy                           0.94      2950
   macro avg       0.87      0.92      0.89      2950
weighted avg       0.95      0.94      0.94      2950

```

## Archivos Generados
- `confusion_matrix_rf.png`
- `feature_importance_rf.png`
- `per_class_metrics_rf.png`