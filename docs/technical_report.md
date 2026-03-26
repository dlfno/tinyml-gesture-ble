# Reporte Final — Sistema de Clasificación de Movimiento TinyML

## 1. Visión General del Proyecto
Este documento detalla el desarrollo, optimización y despliegue de un sistema de clasificación de gestos en tiempo real utilizando TinyML. El sistema captura datos inerciales (IMU), ejecuta inferencia en el borde (Edge AI) y transmite los resultados de forma inalámbrica a una interfaz de usuario.

La arquitectura del sistema se compone de tres bloques principales:
1. **Hardware / Borde (Arduino Nano 33 BLE Sense Rev2):** Captura datos del IMU (BMI270) a 100 Hz mediante lectura directa de la FIFO por I2C. Preprocesa ventanas de **1000 ms (100 muestras)** y ejecuta un modelo de Deep Learning en formato TFLite Micro para clasificar el movimiento en 4 clases: `CÍRCULO`, `LADO`, `QUIETO` y `DEFAULT`.
2. **Puente de Comunicación (Bridge Python en macOS):** Se conecta al Arduino vía Bluetooth Low Energy (BLE GATT), recibe las predicciones en un formato JSON muy compacto y las retransmite localmente mediante WebSockets.
3. **Interfaz de Usuario (Dashboard Web):** Un frontend en HTML/JS que se conecta al WebSocket y visualiza las probabilidades en tiempo real usando un suavizado EMA (Exponential Moving Average) para asegurar una experiencia fluida y evitar parpadeos abruptos.

---

## 2. Evolución del Modelo: De LSTM a CNN 1D
El proyecto comenzó utilizando una arquitectura de red neuronal recurrente (LSTM). Sin embargo, pronto me topé con varios bloqueos estructurales críticos que me obligaron a replantear el diseño del modelo base.

### Los problemas del LSTM
- **Falta de soporte nativo en TFLite Micro:** Las capas LSTM de TensorFlow generan internamente operaciones `TensorListReserve` para manejar los estados temporales a lo largo del tiempo. Estas operaciones no están presentes en el conjunto de operaciones nativas de microcontroladores (`TFLITE_BUILTINS`).
- **Overflow de Memoria Flash:** Para lograr que el LSTM funcionara, tuve que habilitar `SELECT_TF_OPS`, lo cual incrusta un subconjunto sustancial del runtime de TensorFlow en el binario final. Esto disparó el peso del modelo a ~574 KB que, al sumarlo con las librerías de BLE y mbed OS, excedía la capacidad de la memoria Flash del Arduino (1 MB).
- **Sensibilidad Extrema a la Cuantización:** Al intentar cuantizar las entradas y salidas del modelo LSTM a enteros de 8 bits (INT8) para ahorrar espacio, la precisión de clasificación colapsó dramáticamente de un 96.77% a un 74.45%. Los estados ocultos del LSTM son valores continuos altamente precisos y el error acumulado al redondearlos durante múltiples timesteps era inaceptable.

### La solución definitiva: Conv1D
Decidí abandonar el LSTM en favor de una Red Neuronal Convolucional 1D (Conv1D seguida de GlobalAveragePooling). Las convoluciones 1D son excepcionales encontrando patrones temporales locales en las series de tiempo del acelerómetro y giroscopio sin necesidad de mantener memoria de estado.
- **Reducción de Tamaño Masiva:** El modelo TFLite pasó de pesar ~574 KB a tan solo **~27 KB**, y ya no requería importar `SELECT_TF_OPS`. Esto resolvió por completo los problemas de overflow en el hardware.
- **Preservación de Precisión:** La precisión de la CNN 1D alcanzó ~97.93% en float32 y ~97.59% tras cuantización INT8, con una degradación de solo 0.34 pp.

---

## 3. Justificación del Tamaño de Ventana (Window Size)

### El problema con 300 ms
El proyecto inició con una ventana de **30 muestras a 100 Hz = 300 ms**. Esta duración estaba por debajo del mínimo recomendado por la literatura especializada en reconocimiento de actividad humana (HAR).

### Evidencia bibliográfica
La selección del tamaño de ventana en sistemas de clasificación de movimiento tiene un impacto directo en la calidad de las características extraídas. Los movimientos humanos ocurren a baja frecuencia (tipicamente 0.5–5 Hz), por lo que ventanas muy cortas no capturan un ciclo completo de movimiento.

**Banos et al. (2014)** — *"Window Size Impact in Human Activity Recognition"* — es la referencia más citada en este tema. Sus hallazgos centrales son:
- Ventanas menores a **500 ms** son insuficientes para capturar la dinámica temporal de los movimientos humanos; la precisión cae significativamente.
- El rango **1–2 s** es óptimo para sistemas en tiempo real, ya que equilibra latencia de respuesta con suficiencia de información temporal.
- Ventanas muy largas (>3 s) introducen latencia inaceptable y pueden englobar múltiples actividades distintas.

**Wang et al. (2018)** y múltiples trabajos de benchmark sobre el dataset UCI-HAR (estándar de facto en HAR) utilizan ventanas de **2.56 s a 50 Hz = 128 muestras** con 50% de overlap como configuración de referencia.

**Fida et al. (2015)** reportaron que para tareas de reconocimiento de gestos con IMU, el rango de **1–1.5 s** es suficiente para capturar un gesto completo (p.ej., un movimiento lateral o circular del brazo) sin sacrificar latencia.

### Decisión adoptada
Se estableció `WINDOW_SIZE = 100` muestras a 100 Hz = **1000 ms**, con `STEP_SIZE = 50` (50% de overlap durante entrenamiento). Esta configuración:
- Cumple el mínimo recomendado de 500 ms y cae en el rango óptimo para tiempo real (1–2 s).
- Captura al menos un ciclo completo de cada gesto (circulo, lado, quieto).
- Mantiene la tasa de inferencia a ~1 inferencia/segundo, aceptable para interacción humana en tiempo real.
- Produce 14,747 ventanas de entrenamiento sobre el dataset existente, suficientes para el modelo.

El impacto fue inmediato: la precisión del modelo pasó de **92.69%** (con 300 ms) a **97.59%** (con 1000 ms), una mejora de casi 5 puntos porcentuales sin cambiar la arquitectura ni los datos.

---

## 4. Bug Crítico: Normalización Z-score
Uno de los descubrimientos más importantes de este ciclo de desarrollo fue entender en la práctica cómo una discrepancia entre la etapa de entrenamiento y la inferencia puede romper silenciosamente un sistema de Machine Learning.

### El Bug de Normalización
Durante el entrenamiento del modelo en Python, utilicé `StandardScaler` (Z-score: `(x - mean) / std`) para normalizar los datos de entrenamiento. Sin embargo, en el firmware inicial en C++ del Arduino, estaba aplicando una técnica de "Range Normalization" simple (dividiendo los datos del acelerómetro entre 4.0 y los del giroscopio entre 2000.0).

Esto provocaba que el modelo desplegado estuviera recibiendo datos en un dominio estadístico completamente distinto al que aprendió. Por ejemplo: en reposo sobre una mesa, el eje X de la aceleración lee aproximadamente -3.87 (en las unidades del dataset). Con la normalización por rango estaba inyectando un valor de `-0.96`. Pero con Z-score, el modelo esperaba recibir un valor cercano a `0`. Debido a este desajuste masivo, las predicciones del modelo en tiempo real eran esencialmente ruido aleatorio, a pesar de que las pruebas teóricas daban sobre el 92% de precisión.

**Solución:** Se extrajeron de los datos de entrenamiento los arreglos reales de medias y desviaciones estándar (`SCALER_MEAN` y `SCALER_STD`) y se implementó exactamente la misma fórmula Z-score en el firmware C++.

---

## 5. Bug Crítico: Discrepancia en el Pipeline de Datos IMU

Este fue el bug más profundo y difícil de diagnosticar del proyecto. El síntoma era claro — el modelo siempre predecía la misma clase independientemente del movimiento — pero la causa era sutil e invisible en el código.

### Descripción del problema

El sistema tiene dos firmwares distintos para el Arduino BMI270:

| Firmware | Propósito | Método de lectura IMU |
|---|---|---|
| `recoleccion.ino` | Captura de datos de entrenamiento | FIFO raw por I2C directo |
| `tinyml_ble_cnn.ino` | Inferencia en tiempo real | `IMU.readAcceleration()` / `IMU.readGyroscope()` |

La discrepancia estaba en que estos dos métodos **no devuelven los mismos valores**.

### La causa: swap de ejes en la librería Arduino

La librería `Arduino_BMI270_BMM150` aplica una transformación de ejes específica para el board Nano 33 BLE. En `BMI270.cpp` (líneas 160–209):

```cpp
// Para ARDUINO_NANO33BLE:
x = -sensor_data.acc.y / 8192.0f;  // eje X = -Y físico del sensor
y = -sensor_data.acc.x / 8192.0f;  // eje Y = -X físico del sensor
z =  sensor_data.acc.z / 8192.0f;  // eje Z = Z físico (sin cambio)

// Lo mismo para el giroscopio:
x = -sensor_data.gyr.y / 16.384f;
y = -sensor_data.gyr.x / 16.384f;
z =  sensor_data.gyr.z / 16.384f;
```

El firmware de captura (`recoleccion.ino`), en cambio, lee la FIFO directamente por I2C y envía los bytes crudos del sensor. El script `capture.py` los interpreta en el orden nativo del hardware BMI270 (headerless FIFO: GYR_X, GYR_Y, GYR_Z, ACC_X, ACC_Y, ACC_Z) y los divide por las mismas constantes de escala (`ACC_SCALE = 8192`, `GYRO_SCALE = 16.384`).

El resultado: el dataset de entrenamiento contiene los ejes en el **sistema de coordenadas nativo del sensor**, mientras que `readAcceleration()` devuelve los ejes en el **sistema de coordenadas del board**, con X e Y intercambiados y negados. El modelo nunca vio los datos en ese orden durante el entrenamiento.

### Intentos de solución incorrectos

Durante el diagnóstico se intentaron dos correcciones que no resolvieron el problema:

1. **Multiplicar ax, ay, az por 9.80665** (conversión g → m/s²): El modelo pasó de predecir siempre `QUIETO` a predecir siempre `DEFAULT`, indicando que la escala del acelerómetro mejoró parcialmente pero el problema de ejes persistía.

2. **Invertir el swap manualmente** (`raw[] = { -ay, -ax, az, -gy, -gx, gz }`): Corregía los ejes pero mantenía la lectura a través de la librería, lo que introducía otras sutilezas (sincronización, timing, posibles diferencias en el filtrado interno).

### Solución definitiva: leer la FIFO directamente en inferencia

La solución limpia fue hacer que `tinyml_ble_cnn.ino` lea la FIFO del BMI270 exactamente igual que `recoleccion.ino`, eliminando por completo la dependencia de `readAcceleration()` y `readGyroscope()` para la captura de muestras:

```cpp
// Frame FIFO headerless: GYR_X GYR_Y GYR_Z ACC_X ACC_Y ACC_Z (int16 LE)
int16_t gx_raw = (int16_t)(f[0]  | (f[1]  << 8));
int16_t gy_raw = (int16_t)(f[2]  | (f[3]  << 8));
int16_t gz_raw = (int16_t)(f[4]  | (f[5]  << 8));
int16_t ax_raw = (int16_t)(f[6]  | (f[7]  << 8));
int16_t ay_raw = (int16_t)(f[8]  | (f[9]  << 8));
int16_t az_raw = (int16_t)(f[10] | (f[11] << 8));

// Mismas escalas que capture.py
float vals[] = {
    ax_raw / 8192.0f,   ay_raw / 8192.0f,   az_raw / 8192.0f,
    gx_raw / 16.384f,   gy_raw / 16.384f,   gz_raw / 16.384f
};
```

Con esto, el pipeline training ↔ inferencia es **bit a bit idéntico**: mismas unidades, mismo orden de ejes, mismas constantes de escala. El modelo funcionó correctamente en hardware en cuanto se aplicó este cambio.

### Lección aprendida

Este bug ilustra un principio fundamental en MLOps y sistemas embebidos: **la consistencia del pipeline de datos es tan importante como la arquitectura del modelo**. Un modelo con 97.6% de precisión en evaluación offline puede fallar completamente en producción si cualquier paso del preprocesamiento difiere entre entrenamiento e inferencia, aunque sea en el orden de los ejes.

---

## 6. Otros Bugs Importantes Resueltos

### 1. Out-of-Bounds Memory en el Tensor de Salida
En una versión temprana del firmware, el código intentaba iterar y leer 4 valores probabilísticos del tensor de salida de un modelo que en ese momento solo contaba con 3 clases de salida. Leer fuera de los límites de un tensor en TFLite Micro accede al byte inmediato siguiente en la memoria "arena". En la práctica, este valor era basura residual de memoria, lo que provocaba que la inferencia ocasionalmente determinara que el movimiento más probable era uno inexistente. Se corrigió empatando estrictamente las lecturas a la longitud de clases reales y validando las dimensiones del tensor en tiempo de ejecución.

### 2. Bloqueo de BLE Advertising por inicialización de TFLite
**Síntoma:** El Arduino se flasheaba exitosamente y encendía sus LEDs, pero era invisible en los escaneos Bluetooth.
**Causa:** Originalmente la función `setup()` inicializaba primero TFLite (`AllocateTensors()`) y luego el stack de Bluetooth. Cuando la asignación de memoria fallaba por falta de espacio en el Tensor Arena, la rutina abortaba antes de llegar a `BLE.advertise()`.
**Solución:** Se invirtió el flujo de inicialización. BLE se enciende primero. Si TFLite o el IMU fallan posteriormente, el dispositivo sigue siendo detectable para diagnóstico.

### 3. Caché agresivo de Nombres BLE en macOS
**Síntoma:** Incluso después de actualizar el nombre del dispositivo en el firmware, la librería `bleak` en Python no lograba encontrarlo al buscar por nombre.
**Causa:** CoreBluetooth en macOS tiene un caché muy agresivo para los nombres de dispositivos asociados a una dirección MAC específica.
**Solución:** Se refactorizó el Bridge de Python para escanear por **Service UUID** en lugar de nombre de dispositivo. El UUID forma parte de la carga útil del paquete BLE y no está afectado por el caché del sistema operativo.

---

## 7. Random Forest: Segunda Opción sin TFLite

### Motivación

Además de la CNN, se exploró un clasificador Random Forest como alternativa para el Arduino. La ventaja principal es que **no requiere TFLite Micro**, lo que ahorra ~200 KB de flash y elimina toda la complejidad del intérprete.

### Pipeline de features

A diferencia de la CNN, que consume la ventana cruda (100×6), el RF opera sobre **32 features estadísticas** extraídas por ventana:

- 5 estadísticos por eje (mean, std, min, max, RMS) × 6 ejes = 30 features
- Signal Magnitude Area del acelerómetro y giroscopio = 2 features

Este contrato de features debe implementarse de forma idéntica en Python (entrenamiento) y en C++ (firmware Arduino). Cualquier discrepancia produce el mismo tipo de fallo silencioso descrito en la Sección 4.

### Resultados

| Característica | CNN 1D | Random Forest |
|---|---|---|
| Parámetros / Árboles | ~13,000 params | 20 árboles, depth 8 |
| Test Accuracy | **97.59%** | 94.03% |
| Formato despliegue | INT8 TFLite | C++ header |
| Tamaño compilado (est.) | ~27 KB | ~80–120 KB |
| Dependencia TFLite | Sí (~200 KB) | **No** |
| Latencia inferencia | ~14 ms | **<1 ms** |
| Feature más importante | — | `ay_std` (18.6% Gini) |

### Exportación con micromlgen

El modelo sklearn se convierte a C++ puro con `micromlgen`, generando `model_rf.h` con if/else anidados:

```cpp
int predict(float *x) {
    uint8_t votes[4] = { 0 };
    // Árbol #1
    if (x[1] <= 0.312) { votes[0] += 1; } else { votes[2] += 1; }
    // ... 19 árboles más ...
    return argmax(votes, 4);
}
```

El firmware Arduino extrae los 32 features del buffer FIFO y llama a `rf.predict(features)` sin ninguna dependencia de ML externa.

### Cuándo usar Random Forest

El RF es preferible cuando: (a) el flash está al límite y no cabe TFLite Micro, (b) se necesita latencia de inferencia en microsegundos, o (c) se priorizan modelos interpretables (importancia de features). La CNN sigue siendo la opción recomendada para producción por su mayor accuracy y menor huella compilada total.

---

## 8. Regresión Logística — Baseline Lineal

### Motivación

Siguiendo las recomendaciones de `docs/model_viability_analysis.md`, se implementó Regresión Logística como **baseline lineal de referencia**. Su propósito es establecer el piso de rendimiento: si RF supera a Logística en más de 5 pp, eso cuantifica concretamente cuánto valor aporta el modelado no-lineal del ensemble.

### El problema del escalado

El hallazgo más importante de esta implementación fue la necesidad crítica de `StandardScaler`. Sin escalado, el accuracy colapsa a **~52%** — prácticamente equivalente a clasificación aleatoria en un problema de 4 clases. Con escalado, converge a **90.85%** en solo 123 iteraciones.

La causa es directa: los features estadísticos tienen escalas radicalmente distintas. La media del acelerómetro (`ax_mean`) opera en el rango [-4, 4] (unidades g), mientras que la media del giroscopio (`gx_mean`) puede alcanzar [-500, 500] (°/s). `lbfgs` no puede converger cuando el gradiente está dominado por las dimensiones de mayor escala.

```
Sin scaler:  52.27% test accuracy, no converge en 1000 iteraciones
Con scaler:  90.85% test accuracy, converge en 123 iteraciones
```

### Consideración de despliegue: doble scaler

El proyecto ya tiene un `StandardScaler` para el CNN: `scaler_params.json` normaliza la **ventana cruda** (100 × 6 valores de IMU) antes de la inferencia. El scaler de LR es diferente: normaliza los **32 features estadísticos** extraídos de esa ventana. Son dos etapas distintas del pipeline:

```
Ventana cruda (100×6)
  │
  ├── [CNN path] → scaler_params.json (Z-score de valores IMU crudos) → CNN
  │
  └── [RF/LR path] → extract_features() → 32 features estadísticos
                         │
                         └── [LR only] → scaler_params_lr.json (Z-score de features) → LRModel::predict()
```

RF es invariante a escala (árboles de decisión usan umbrales, no distancias), por eso no necesita scaler. LR no lo es.

El firmware Arduino para LR necesita:
1. Extraer los 32 features estadísticos de la ventana FIFO.
2. Aplicar Z-score con los valores de `scaler_params_lr.json`: `feat_scaled[i] = (feat[i] - mean[i]) / std[i]`.
3. Llamar a `LRModel::predict(feat_scaled)`.

### Resultados

| Conjunto | Accuracy |
|----------|----------|
| Train    | 91.43%   |
| Val      | 90.73%   |
| Test     | **90.85%** |
| CV 5-fold | 91.12% ± 0.61% |

```
              precision    recall  f1-score   support

     CIRCULO       0.92      0.89      0.90       934
     DEFAULT       0.53      0.71      0.61       150
        LADO       0.89      0.88      0.89       933
      QUIETO       0.99      0.99      0.99       933
```

La clase `DEFAULT` sigue siendo la más difícil (F1=0.61), igual que en RF. Esto no es una limitación del modelo lineal: es una propiedad del dataset, donde `DEFAULT` agrupa movimientos heterogéneos con pocas muestras (40 archivos vs 120 de las demás clases).

### Comparativa de los tres modelos

| Característica | CNN 1D | Random Forest | Reg. Logística |
|---|---|---|---|
| Test Accuracy | **97.59%** | 94.03% | 90.85% |
| Flash total | ~227 KB | ~80–120 KB | **~5 KB** |
| Dependencia TFLite | Sí | No | No |
| Scaler en firmware | Raw IMU (6 vals) | Ninguno | Features estadísticos (32 vals) |
| Latencia | ~14 ms | <1 ms | **<0.1 ms** |
| Interpretabilidad | Baja | Alta (Gini) | Alta (coeficientes) |

### Interpretación del heatmap de coeficientes

La Regresión Logística produce un coeficiente por cada (clase, feature), lo que permite entender directamente qué features empuja el modelo para discriminar cada clase. `QUIETO` muestra coeficientes fuertemente negativos en `*_std` y positivos en `*_mean` — coherente con que en reposo las desviaciones estándar son mínimas. `CIRCULO` muestra correlaciones positivas en varios features de aceleración que `LADO` no tiene, indicando que hay separabilidad lineal parcial entre estos dos gestos pero no completa (lo que explica la brecha de ~3 pp respecto a RF).

### Exportación con micromlgen

`micromlgen` no soporta `Pipeline` de sklearn, por lo que el `StandardScaler` y el `LogisticRegression` se exportan separadamente: el scaler como JSON para el firmware, y el clasificador como C++ puro:

```python
from micromlgen import port
c_code = port(clf, classname='LRModel', classmap=classmap)
# → model_lr.h, 4.81 KB — multiplicación de matrices + softmax en C++
```

El resultado es un modelo de **4.81 KB** — el más pequeño de todos, sin runtime externo.

---

## 9. Gaussian Naive Bayes — Referencia de Complejidad Mínima

### Motivación

Gaussian Naive Bayes (GNB) representa el extremo opuesto de complejidad respecto a la CNN: 256 parámetros (media y varianza por cada feature por clase), sin runtime externo, sin scaler. Su valor en el proyecto es como referencia diagnóstica: si GNB alcanzara accuracy cercano al RF, eso confirmaría que las 32 features estadísticas capturan casi toda la información necesaria de forma independiente.

### Resultado: 63.93% de accuracy

El modelo convergió sin problemas, pero el accuracy real (63.93%) queda 17-24 puntos porcentuales por debajo del rango estimado en el análisis de viabilidad (80-88%). El classification report revela el origen del fallo:

```
              precision    recall  f1-score   support

     CIRCULO       0.50      0.91      0.65       934
     DEFAULT       0.31      0.46      0.37       150
        LADO       0.80      0.31      0.45       933
      QUIETO       1.00      0.73      0.84       933
```

El 58.1% de las muestras `LADO` se predicen como `CIRCULO`. Este es el síntoma exacto de la violación del supuesto de independencia condicional: CIRCULO y LADO tienen distribuciones marginales similares para la mayoría de los features (ambos gestos generan aceleraciones de amplitud comparable), pero se distinguen por la **correlación entre ejes** — un movimiento circular genera una dependencia conjunta entre `ax` y `ay` que un movimiento lateral puro no tiene. GNB factoriza la probabilidad conjunta como producto de marginales y es ciego a esta estructura.

### Por qué el estimado de 80-88% era optimista

El análisis de viabilidad notaba que "GNB funciona mejor de lo que la teoría sugiere porque las distribuciones gaussianas por clase capturan suficiente señal aunque la independencia no se cumpla estrictamente". Esto es cierto para muchos datasets, pero este caso presenta una condición adversa específica: dos clases (CIRCULO y LADO) con distribuciones marginales altamente solapadas que solo se separan en el espacio conjunto. En presencia de este tipo de confusión, GNB fracasa sistemáticamente.

### Workaround técnico: `sigma_` vs `var_`

micromlgen espera el atributo `sigma_` de la API antigua de sklearn. sklearn ≥ 1.0 renombró este atributo a `var_`. El script añade un shim de compatibilidad antes de la exportación:

```python
if not hasattr(clf, 'sigma_') and hasattr(clf, 'var_'):
    clf.sigma_ = clf.var_
```

### Comparativa de los cuatro modelos

| Modelo | Accuracy | Flash total | Latencia | Scaler en firmware |
|---|---|---|---|---|
| CNN 1D | **97.59%** | ~227 KB | ~14 ms | Raw IMU (6 vals) |
| Random Forest | 94.03% | ~80–120 KB | <1 ms | Ninguno |
| Reg. Logística | 90.85% | ~5 KB | <0.1 ms | Features estadísticos (32 vals) |
| Gaussian NB | 63.93% | ~9 KB | <0.1 ms | Ninguno |

### Conclusión diagnóstica

La brecha CNN→RF→LR→NB (97.6% → 94.0% → 90.9% → 63.9%) no es monótona solo en términos de complejidad del modelo: también refleja cuánto depende el problema de capturar interacciones entre features. GNB asume cero interacciones; LR captura interacciones lineales; RF captura interacciones no-lineales mediante particiones del espacio de features; CNN captura patrones temporales locales en la señal cruda. La caída brusca entre LR y NB (27 pp) indica que incluso las interacciones lineales entre features son fundamentales para distinguir LADO de CIRCULO en este dataset.

---

## 10. SVM Lineal — Clasificador de Margen Máximo

### Motivación

El SVM Lineal (`LinearSVC`) busca el hiperplano que maximiza el margen entre clases en lugar de minimizar log-loss como Regresión Logística. Teóricamente ofrece mejor generalización en fronteras ajustadas, pero al operar en el mismo espacio lineal de 32 features debería producir resultados similares a LR.

### Resultado: 91.36% de accuracy

```
              precision    recall  f1-score   support

     CIRCULO       0.90      0.91      0.91       934
     DEFAULT       0.65      0.63      0.64       150
        LADO       0.90      0.88      0.89       933
      QUIETO       0.97      1.00      0.98       933
```

El modelo converge en solo **79 iteraciones** (frente a las 123 de LR) y alcanza 91.36% — prácticamente idéntico a LR (90.85%, +0.51 pp). La diferencia es estadísticamente insignificante dado el CV de 91.12% ± 0.49% vs 91.12% ± 0.61% de LR. Esto confirma lo predicho en el análisis de viabilidad: en este espacio de 32 features, el criterio de margen máximo no aporta ventaja medible frente a log-loss.

### Workaround de exportación: proxy LogisticRegression

`micromlgen` no soporta `LinearSVC`. Sin embargo, la función de decisión de ambos modelos es idéntica: `argmax(coef_ @ x + intercept_)`. El script copia los pesos entrenados del SVM a un objeto `LogisticRegression` vacío y lo exporta:

```python
lr_proxy = LogisticRegression()
lr_proxy.coef_      = svm.coef_       # (n_classes, n_features)
lr_proxy.intercept_ = svm.intercept_  # (n_classes,)
lr_proxy.classes_   = svm.classes_
c_code = port(lr_proxy, classname='SVMLinearModel', classmap=classmap)
```

Las predicciones del proxy son bit a bit idénticas al `LinearSVC` original (verificado con `np.all(svm.predict(X) == proxy.predict(X))`).

### Comparativa de modelos lineales

| Modelo | Loss | Accuracy | CV | Flash | Convergencia |
|---|---|---|---|---|---|
| Reg. Logística | Log-loss | 90.85% | 91.12% ± 0.61% | ~5 KB | 123 iter |
| SVM Lineal | Hinge loss | 91.36% | 91.12% ± 0.49% | ~5 KB | 79 iter |

La diferencia de 0.51 pp y la varianza idéntica en CV confirman que ambos modelos están capturando esencialmente la misma frontera de decisión lineal. El SVM converge más rápido porque `liblinear` (optimizador de LinearSVC) es más eficiente que `lbfgs` para este tamaño de dataset.

### Comparativa global de modelos evaluados

| Modelo | Accuracy | Flash total | Latencia |
|---|---|---|---|
| CNN 1D | **97.59%** | ~227 KB | ~14 ms |
| Random Forest | 94.03% | ~80–120 KB | <1 ms |
| SVM RBF | pendiente | ~40–70 KB | <2 ms |
| SVM Lineal | 91.36% | ~5 KB | <0.1 ms |
| Reg. Logística | 90.85% | ~5 KB | <0.1 ms |
| Gaussian NB | 63.93% | ~9 KB | <0.1 ms |

---

## 11. SVM RBF — Clasificador No-Lineal de Máximo Margen

### Motivación

SVM con kernel RBF (`SVC(kernel='rbf')`) es el modelo clásico identificado como Prioridad 1 en `docs/model_viability_analysis.md`. El kernel RBF proyecta los 32 features estadísticos a un espacio de alta dimensión donde la separación lineal es más factible — especialmente para las clases `CIRCULO` y `LADO`, cuyas distribuciones marginales se solapan pero que la brecha de 27 pp entre LR y GNB (Sección 9) confirma que se distinguen por correlaciones entre ejes. LR captura parte de esta estructura (90.85%), RF captura más vía particiones no-lineales (94.03%); SVM RBF debería capturar las mismas fronteras con el argumento teórico del máximo margen.

### Implementación

El pipeline es idéntico al de RF, LR, NB y SVM Lineal:

- 32 features estadísticas (mismo contrato que todos los modelos clásicos)
- `StandardScaler` ajustado en train, aplicado en val/test y firmware
- `SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')`

Los hiperparámetros `C=10, gamma='scale'` fueron seleccionados siguiendo las recomendaciones del análisis de viabilidad, que estima 200–600 support vectors con este rango — un footprint completamente manejable para el flash disponible.

### Exportación: micromlgen directo (sin proxy)

A diferencia de `LinearSVC` (que requiere un proxy `LogisticRegression` porque micromlgen no lo soporta), `SVC(kernel='rbf')` es soportado de forma nativa:

```python
from micromlgen import port
c_code = port(clf, classname='SVMRBFModel', classmap=classmap)
# → model_svm_rbf.h con loop de distancias kernel sobre todos los SVs
```

El tamaño del modelo escala linealmente con el número de support vectors: ~128 bytes por SV (32 features × 4 bytes). Con 200–600 SVs estimados, el modelo ocupa ~25–75 KB de datos más el código del loop.

### Consideraciones de firmware

El firmware Arduino para SVM RBF requiere el mismo pipeline que LR y SVM Lineal:

1. Extraer los 32 features estadísticos de la ventana FIFO.
2. Aplicar Z-score con `scaler_params_svm_rbf.json`: `feat_scaled[i] = (feat[i] - mean[i]) / std[i]`.
3. Llamar a `SVMRBFModel::predict(feat_scaled)`.

Sin este escalado, la función kernel RBF recibiría distancias incorrectas — los features del giroscopio en °/s dominarían sobre los del acelerómetro por diferencia de escala. El bug sería análogo al de normalización descrito en la Sección 4, pero a nivel de features estadísticos.

### Resultados

> **Pendiente de entrenamiento.** Ejecutar `python training/train_svm_rbf.py` para obtener los resultados reales.
> Accuracy estimado según análisis de viabilidad: **94–97%** — potencialmente competitivo con CNN 1D.

---

## 12. Pipeline de Despliegue (CNN, RF, LR, NB, SVM Lineal y SVM RBF)

- **Header Autogenerado (CNN):** El modelo `.tflite` se convierte en un array estático en C mediante `xxd`, resultando en `model_cnn.h` que se compila directamente junto con el firmware.
- **Header Autogenerado (RF, LR, NB y SVM):** `micromlgen` convierte los modelos sklearn a sus respectivos `.h` con lógica C++ pura. No requieren `xxd` ni pasos de conversión binaria. Para `LinearSVC` se usa un proxy `LogisticRegression` (mismos pesos); `SVC(rbf)` es exportado directamente.
- **Scaler para modelos lineales y SVM en firmware:** `scaler_params_lr.json`, `scaler_params_svm_linear.json` y `scaler_params_svm_rbf.json` contienen los 32 valores de media y std del `StandardScaler`. El firmware aplica Z-score sobre los features extraídos antes de llamar a `predict()`. RF y NB no requieren scaler.
- **Bridge Robusto:** El puente de Python captura el JSON compacto de BLE y lo publica con reconexión automática en caso de pérdida de señal.
- **Dashboard con Suavizado (EMA):** La inferencia opera a ~1 Hz. Para evitar una interfaz nerviosa, se aplica un filtro EMA con `alpha = 0.3` a los arrays probabilísticos en JavaScript: `smoothed[i] = (0.3 * raw[i]) + (0.7 * smoothed[i])`.
- **BLE UUIDs independientes:** CNN y RF usan UUIDs distintos para poder distinguirlos desde el bridge sin modificar el código Python.

---

## 13. Conclusiones Finales

1. **Tamaño de ventana basado en literatura:** La decisión de usar 1000 ms en lugar de 300 ms no fue arbitraria. Está respaldada por la bibliografía de HAR (Banos et al. 2014) y produjo una mejora de ~5 pp en precisión. Los movimientos humanos tienen una frecuencia baja intrínseca que requiere ventanas de al menos 500 ms para ser capturados correctamente.

2. **Sincronía total del pipeline:** Cualquier modelo, sin importar qué tan perfecto sea teóricamente, fracasará si el pipeline de datos entre entrenamiento e inferencia difiere. En este proyecto, la discrepancia no era en la fórmula de normalización sino en el sistema de coordenadas del IMU — algo que no es visible mirando el código Python de entrenamiento, solo emerge al comparar ambos firmwares de Arduino.

3. **Adecuación de Arquitectura al Hardware:** En el mundo de Edge AI y TinyML, la elección de arquitectura debe considerar tanto la precisión como las restricciones de hardware. La CNN 1D es la opción más eficiente en esta plataforma: menor latencia, menor arena de tensores y robustez probada con cuantización INT8.

4. **El desafío de la clase "Default":** Introducir una clase `DEFAULT` entrenada activamente con diversos movimientos aleatorios redujo enormemente los falsos positivos que surgían cada vez que se realizaba una acción no mapeada, otorgando robustez a toda la cadena de detección.

5. **Cinco modelos clásicos evaluados, uno pendiente:** CNN (97.59%, 26.9 KB INT8, recomendado), Random Forest (94.03%, sin TFLite), SVM Lineal (91.36%, ~5 KB), Regresión Logística (90.85%, 4.81 KB, baseline lineal) y Gaussian Naive Bayes (63.93%, 9 KB, referencia de complejidad mínima). SVM RBF está implementado y pendiente de entrenamiento (accuracy estimado 94–97%). La brecha CNN→RF→SVM_L≈LR→NB (97.6% → 94.0% → ~91% → 63.9%) refleja cuánto depende el problema de capturar interacciones: GNB asume cero; LR/SVM_L capturan lineales; RF captura no-lineales por partición; CNN captura patrones temporales locales. La caída brusca entre LR y NB (27 pp) confirma que las interacciones lineales son fundamentales para distinguir LADO de CIRCULO.

6. **El escalado de features es tan crítico como la normalización de datos crudos.** El bug de normalización Z-score (Sección 4) afecta al pipeline CNN. Un bug análogo existe para LR, SVM Lineal y SVM RBF: sin `StandardScaler` sobre los 32 features estadísticos, el accuracy de LR cae de 90.85% a 52%. GNB y RF no requieren scaler (GNB modela distribuciones por clase; RF usa umbrales) — lo que demuestra que el problema de NB no es de escala sino de supuestos estadísticos.

7. **GNB como herramienta diagnóstica, no como modelo de producción.** El resultado de 63.93% no es un fracaso del pipeline: es información valiosa. Confirma que la clasificación de estos gestos depende de correlaciones entre ejes que ningún modelo de independencia condicional puede capturar, y que el dataset tiene dos clases (CIRCULO y LADO) cuyas distribuciones marginales se solapan significativamente.

---

## 14. Referencias

- Banos, O., Galvez, J. M., Damas, M., Pomares, H., & Rojas, I. (2014). **Window Size Impact in Human Activity Recognition.** *Sensors*, 14(4), 6474–6499. https://doi.org/10.3390/s140406474
- Wang, J., Chen, Y., Hao, S., Peng, X., & Hu, L. (2018). **Deep Learning for Sensor-based Activity Recognition: A Survey.** *Pattern Recognition Letters*, 119, 3–11.
- Fida, B., Bernabucci, I., Bibbo, D., Conforto, S., & Schmid, M. (2015). **Varying Behavior of Different Window Sizes on the Classification of Static and Dynamic Physical Activities from a Single Accelerometer.** *Medical Engineering & Physics*, 37(7), 705–711.
- Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). **A Public Domain Dataset for Human Activity Recognition Using Smartphones.** *ESANN*.
