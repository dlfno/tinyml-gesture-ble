# Reporte de Evaluación — SVM Lineal

## Resumen
- **Arquitectura:** LinearSVC(C=1.0, max_iter=5000) + StandardScaler + 32 features estadísticas
- **Formato de despliegue:** C++ header (`model_svm_linear.h`) — sin dependencia de TFLite
- **Window size:** 100 muestras @ 100 Hz (1000 ms)
- **BLE device name:** `TinyML-SVML`

## Tamaño del Modelo
| Archivo | Tamaño | Notas |
|---------|--------|-------|
| `model_svm_linear.h` | 4.8 KB | Texto C++; compilado según modelo |

## Accuracy en Test Set
| Accuracy |
|----------|
| 91.36% |

## Classification Report
```
              precision    recall  f1-score   support

     CIRCULO       0.90      0.91      0.91       934
     DEFAULT       0.65      0.63      0.64       150
        LADO       0.90      0.88      0.89       933
      QUIETO       0.97      1.00      0.98       933

    accuracy                           0.91      2950
   macro avg       0.86      0.85      0.86      2950
weighted avg       0.91      0.91      0.91      2950

```

## Archivos Generados
- `confusion_matrix_svm_linear.png`
- `per_class_metrics_svm_linear.png`
- `coefficients_svm_linear.png`