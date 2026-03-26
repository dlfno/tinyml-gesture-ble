<div align="center">

# TinyML — Clasificación de Gestos

**Reconocimiento de movimiento en tiempo real en el dispositivo — del sensor IMU al dashboard web por BLE.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://docs.python.org/3/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21%2B-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/api_docs)
[![TFLite](https://img.shields.io/badge/TFLite-INT8%20Cuantizado-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite/guide)
[![Arduino](https://img.shields.io/badge/Arduino-Nano%2033%20BLE-00878A?style=flat-square&logo=arduino&logoColor=white)](https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/stable/user_guide.html)
[![Bleak](https://img.shields.io/badge/Bleak-Cliente%20BLE-4B8BBE?style=flat-square)](https://bleak.readthedocs.io/en/latest/)
[![WebSockets](https://img.shields.io/badge/WebSockets-12.0%2B-010101?style=flat-square)](https://websockets.readthedocs.io/en/stable/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-3.x-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/docs)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-22c55e?style=flat-square)](./LICENSE)
[![CNN INT8](https://img.shields.io/badge/CNN%20INT8-97.6%25%20acc-22c55e?style=flat-square)](eval/model_comparison.md)
[![Tamaño del modelo](https://img.shields.io/badge/Tamaño-26.9%20KB-6366f1?style=flat-square)](models/cnn/)
[![C++](https://img.shields.io/badge/C%2B%2B-Arduino-00878A?style=flat-square&logo=cplusplus&logoColor=white)](arduino/)

</div>

---

## Descripción general

Este proyecto implementa un **pipeline TinyML de extremo a extremo** para clasificar gestos humanos en tiempo real. Un modelo CNN 1D (26.9 KB, INT8) corre directamente en un **Arduino Nano 33 BLE Sense Rev2**, lee datos IMU de 6 ejes (acelerómetro + giroscopio) y transmite los resultados de inferencia por Bluetooth Low Energy hacia un puente Python y un dashboard web en vivo.

```
┌──────────────────────┐   BLE GATT   ┌─────────────────┐  WebSocket  ┌─────────────────────┐
│  Arduino Nano 33 BLE │  ──────────► │  Puente Python  │ ──────────► │   Dashboard Web     │
│  ─────────────────── │              │  ble_bridge.py  │             │   dashboard/        │
│  BMI270 IMU (100 Hz) │              │  localhost:8765 │             │   UI en tiempo real │
│  CNN 1D (26.9 KB)    │              └─────────────────┘             └─────────────────────┘
│  INT8 TFLite Micro   │
└──────────────────────┘
```

### Clases de gestos

| Clase | Descripción |
|---|---|
| `CIRCULO` | Movimiento circular |
| `LADO` | Movimiento lateral de lado a lado |
| `QUIETO` | Dispositivo en reposo |
| `DEFAULT` | Cualquier otro movimiento |

---

## Rendimiento

| Modelo | Formato | Tamaño | Exactitud | Latencia |
|---|---|---|---|---|
| CNN 1D | Float32 TFLite | 52.1 KB | **97.93%** | ~14 ms |
| CNN 1D | INT8 TFLite | **26.9 KB** | **97.59%** | ~14 ms |
| Random Forest | C++ header (sin TFLite) | 576 KB¹ | 94.03% | **<1 ms** |
| Regresión Logística | C++ header (sin TFLite) | **4.8 KB** | 90.85% | **<0.1 ms** |
| Gaussian Naive Bayes | C++ header (sin TFLite) | 8.7 KB | 63.93% | **<0.1 ms** |
| SVM Lineal | C++ header (sin TFLite) | **4.8 KB** | 91.36% | **<0.1 ms** |
| SVM RBF | C++ header (sin TFLite) | ~40–70 KB² | *pendiente* | **<2 ms** |

> La cuantización INT8 del CNN reduce el tamaño del modelo un **48%** con sólo **0.34 pp** de pérdida de exactitud.
>
> ¹ `model_rf.h` son 576 KB de código fuente C++; compilado a ARM Thumb2 resulta en ~80–120 KB.
> Sin la librería TFLite Micro (~200 KB), el firmware RF tiene una huella total de flash menor que el CNN.
>
> La Regresión Logística actúa como **baseline lineal de referencia**: su exactitud del 90.85% confirma que el problema tiene un componente no-lineal significativo que justifica la complejidad adicional de RF y CNN.
>
> Gaussian Naive Bayes alcanza solo el 63.93% — el 58% de las muestras `LADO` se clasifican como `CIRCULO` — confirmando que el supuesto de independencia condicional es gravemente violado por los ejes del IMU, que están altamente correlacionados.
>
> ² El tamaño de `model_svm_rbf.h` depende del número de support vectors (~200–600 esperados). Ejecutar `python training/train_svm_rbf.py` para obtener resultados; exactitud estimada 94–97% (ver `docs/model_viability_analysis.md`).

---

## Estructura del repositorio

```
.
├── arduino/
│   ├── tinyml_ble_cnn/        # Firmware de inferencia CNN (.ino + model_cnn.h)
│   └── tinyml_ble_rf/         # Firmware de inferencia Random Forest (.ino + model_rf.h)
├── bridge/
│   ├── ble_bridge.py          # Relay BLE → WebSocket (Python)
│   └── requirements.txt
├── dashboard/
│   └── index.html             # Dashboard web en tiempo real (Tailwind + Vanilla JS)
├── data_collection/
│   ├── capture.py             # Script de captura de datos por BLE
│   ├── gui.py                 # Interfaz Tkinter para sesiones de captura
│   ├── recoleccion.ino        # Firmware Arduino para recolección de datos
│   └── requirements.txt
├── docs/
│   ├── technical_report.md    # Decisiones de arquitectura, bugs y soluciones
│   ├── model_viability_analysis.md  # Análisis de viabilidad de modelos candidatos
│   └── design.md              # Especificación del sistema de diseño
├── eval/
│   ├── cnn/                   # Gráficas y matrices de confusión del CNN
│   ├── rf/                    # Gráficas e importancia de features del Random Forest
│   ├── lr/                    # Gráficas y coeficientes de la Regresión Logística
│   ├── nb/                    # Gráficas y medias condicionales del Gaussian Naive Bayes
│   ├── svm_l/                 # Gráficas y coeficientes del SVM Lineal
│   ├── svm_rbf/               # Gráficas y distribución de SVs del SVM RBF
│   └── model_comparison.md    # Comparación cuantitativa entre modelos
├── models/
│   ├── cnn/                   # Artefactos del CNN (Keras, TFLite, parámetros del scaler)
│   ├── rf/                    # Artefactos del Random Forest (model_rf.h, predicciones)
│   ├── lr/                    # Artefactos de Reg. Logística (model_lr.h, scaler_params_lr.json)
│   ├── nb/                    # Artefactos de Naive Bayes (model_nb.h, predicciones)
│   ├── svm_l/                 # Artefactos de SVM Lineal (model_svm_linear.h, scaler_params_svm_linear.json)
│   └── svm_rbf/               # Artefactos de SVM RBF (model_svm_rbf.h, scaler_params_svm_rbf.json)
├── training/
│   ├── train_cnn.py           # Pipeline de entrenamiento CNN
│   ├── train_rf.py            # Entrenamiento Random Forest + exportación C++ (micromlgen)
│   ├── train_lr.py            # Entrenamiento Reg. Logística + exportación C++ (micromlgen)
│   ├── train_nb.py            # Entrenamiento Gaussian Naive Bayes + exportación C++ (micromlgen)
│   ├── train_svm_linear.py    # Entrenamiento SVM Lineal + exportación C++ (micromlgen via proxy)
│   ├── train_svm_rbf.py       # Entrenamiento SVM RBF + exportación C++ (micromlgen)
│   ├── eda.py                 # Análisis exploratorio de datos
│   ├── calc_scaler.py         # Cálculo global del scaler para el firmware CNN
│   ├── data/                  # Grabaciones IMU crudas (6 sujetos × ~60 sesiones)
│   └── requirements.txt
├── requirements.txt           # Dependencias completas del proyecto
└── LICENSE
```

---

## Inicio rápido

### Requisitos previos

- Python 3.10+
- Arduino IDE 2.x con las librerías `Arduino_BMI270_BMM150` y `ArduinoBLE` instaladas
- Un **Arduino Nano 33 BLE Sense Rev2**
- macOS / Linux (Windows requiere ajustar el backend BLE)

### 1. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

### 2. Cargar el firmware en el Arduino

1. Abrir `arduino/tinyml_ble_cnn/tinyml_ble_cnn.ino` en el Arduino IDE.
2. Verificar que `model_cnn.h` esté en la misma carpeta (ya está — se genera automáticamente y está en el repositorio).
3. Seleccionar **Arduino Nano 33 BLE** como placa y cargar el firmware.

### 3. Iniciar el puente BLE

```bash
python bridge/ble_bridge.py
```

El puente escanea por `SERVICE_UUID` de BLE, se conecta al Arduino y reenvía los mensajes a `ws://localhost:8765`.

### 4. Abrir el dashboard

Abrir `dashboard/index.html` en cualquier navegador moderno. El dashboard se conecta al servidor WebSocket local y muestra los resultados de inferencia en tiempo real.

---

## Pipeline de entrenamiento

El pipeline ingiere grabaciones IMU de múltiples sujetos, aplica segmentación por ventana deslizante, ajusta un `StandardScaler` y exporta un modelo TFLite cuantizado.

### Formato de los datos

Cada grabación CSV contiene las siguientes columnas:

```
timestamp_ms, ax, ay, az, gx, gy, gz
```

Las grabaciones están organizadas en `training/data/<Sujeto>/` a 100 Hz.

### Estrategia de ventaneo

| Parámetro | Valor | Justificación |
|---|---|---|
| Tamaño de ventana | 100 muestras | 1000 ms — suficiente para un ciclo completo de gesto (Banos et al., 2014) |
| Paso | 50 muestras | 50% de solapamiento para augmentación y continuidad |
| Frecuencia de muestreo | 100 Hz | Tasa de salida del FIFO del BMI270 |

### Ejecutar entrenamiento

```bash
# Análisis exploratorio de datos
python training/eda.py

# Entrenar CNN (recomendado — INT8 TFLite, 26.9 KB)
python training/train_cnn.py

# Entrenar Random Forest (sin TFLite — exporta model_rf.h para Arduino)
python training/train_rf.py

# Entrenar Regresión Logística (baseline lineal — exporta model_lr.h para Arduino)
python training/train_lr.py

# Entrenar Gaussian Naive Bayes (referencia de complejidad mínima — exporta model_nb.h)
python training/train_nb.py

# Entrenar SVM Lineal (margen máximo lineal — exporta model_svm_linear.h)
python training/train_svm_linear.py

# Entrenar SVM RBF (mejor modelo clásico esperado — exporta model_svm_rbf.h)
python training/train_svm_rbf.py

# Recalcular constantes del scaler para el firmware CNN
python training/calc_scaler.py
```

Los resultados se escriben en `models/cnn/`, `models/rf/` y `models/lr/` respectivamente.

### Arquitectura CNN

```
Entrada (100, 6)
  └── Conv1D(16, kernel=3, relu)
  └── Conv1D(32, kernel=3, relu)
  └── Conv1D(64, kernel=3, relu)
  └── GlobalAveragePooling1D
  └── Dense(32, relu)
  └── Dense(4, softmax)       ← 4 clases de gestos
```

El entrenamiento usa `Adam(lr=1e-3)` con `EarlyStopping(patience=15)` y pesos de clase balanceados para manejar la clase `DEFAULT`, que está subrepresentada.

---

## Recolección de datos

Usar la interfaz Tkinter para capturar nuevas grabaciones de gestos:

```bash
python data_collection/gui.py
```

O ejecutar el script de captura sin interfaz gráfica:

```bash
python data_collection/capture.py
```

Ambas herramientas se conectan a un Arduino cargado con `data_collection/recoleccion.ino`, que transmite tramas FIFO crudas del IMU por BLE y las guarda como CSV.

---

## Evaluación

Los scripts de evaluación regeneran todas las gráficas a partir de las predicciones almacenadas:

```bash
# Evaluar un único modelo
python3 eval/generate_eval.py --model cnn
python3 eval/generate_eval.py --model rf
python3 eval/generate_eval.py --model lr
python3 eval/generate_eval.py --model nb
python3 eval/generate_eval.py --model svm_l
python3 eval/generate_eval.py --model svm_rbf

# Evaluar todos y generar comparativa completa
python3 eval/generate_eval.py --model cnn rf lr nb svm_l svm_rbf
```

Cada ejecución produce:
- Matrices de confusión (Float32 e INT8 para CNN; única para RF/LR)
- Curvas de pérdida y exactitud durante el entrenamiento (solo CNN)
- Comparación del impacto de cuantización por clase (solo CNN)
- Heatmap de coeficientes por clase (solo LR)
- Heatmap de medias condicionales por clase (solo NB)
- `eval/<modelo>/report.md` — reporte de clasificación completo

Ver [`eval/model_comparison.md`](eval/model_comparison.md) para la comparación cuantitativa completa.

---

## Decisiones de ingeniería clave

Ver [`docs/technical_report.md`](docs/technical_report.md) para un relato completo de la arquitectura y los bugs encontrados. Resumen:

| Problema | Causa | Solución |
|---|---|---|
| Fallo silencioso de inferencia | El firmware usaba normalización por rango; el entrenamiento usaba Z-score | Unificado a `StandardScaler` — constantes integradas en el firmware |
| Mapeo de ejes incorrecto en el firmware | `readAcceleration()` intercambia X/Y y niega; el FIFO crudo no lo hace | Preprocesamiento de entrenamiento igualado a la lectura cruda del FIFO |
| Caché de nombres BLE en macOS | `CoreBluetooth` cachea nombres de dispositivos desactualizados | Descubrimiento del puente cambiado a escaneo por `SERVICE_UUID` |
| Riesgo de discrepancia en features del RF | La extracción de features en Python debe ser bit a bit idéntica al firmware C++ | 32 features definidos tanto en `train_rf.py` como en `tinyml_ble_rf.ino` |

---

## Licencia

Publicado bajo la [Licencia MIT](./LICENSE).
