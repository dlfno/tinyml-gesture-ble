# Análisis de Viabilidad de Modelos de ML
## Contexto: TinyML Gesture Classifier — Arduino Nano 33 BLE Sense Rev2

---

## 1. Introducción

Este documento evalúa la viabilidad técnica de nueve modelos de Machine Learning en el contexto de este proyecto: los tres modelos implementados o evaluados durante el desarrollo (CNN 1D, LSTM, Random Forest) y seis algoritmos clásicos como posibles alternativas o complementos. El análisis cubre tres dimensiones:

1. **Adecuación al problema** — ¿tiene sentido teórico para clasificación de gestos con IMU?
2. **Deployabilidad en Arduino** — ¿puede correr en un microcontrolador con 1 MB de flash y 256 KB de RAM?
3. **Ruta de despliegue** — ¿TFLite Micro o `micromlgen`?

### Restricciones de hardware

| Recurso | Valor |
|---|---|
| MCU | ARM Cortex-M4F @ 64 MHz |
| Flash total | 1 MB |
| Flash disponible (post-BLE + mbed OS) | ~700 KB |
| RAM | 256 KB |
| Overhead TFLite Micro | ~200 KB de flash |
| Overhead micromlgen | 0 KB (C++ puro, sin runtime) |

### Representaciones de entrada disponibles

| Representación | Dimensión | Uso actual |
|---|---|---|
| Ventana cruda | 100 muestras × 6 ejes = **600 features** | CNN 1D |
| Features estadísticas | 5 estadísticos × 6 ejes + 2 SMA = **32 features** | Random Forest |

Salvo que se indique lo contrario, los modelos clásicos operan sobre los **32 features estadísticos**, ya que la mayoría no escala bien a 600 dimensiones sin overfitting y extracción de features es trivial dado que ya existe la infraestructura.

---

## 2. Modelos del proyecto

### 2.1 CNN 1D — Modelo en producción ✅

#### Adecuación teórica
Las Redes Convolucionales 1D son la arquitectura de referencia para clasificación de series de tiempo en microcontroladores. Una convolución 1D con kernel de tamaño 3 detecta patrones locales (picos, transiciones, correlaciones entre ejes contiguos) en la ventana temporal sin necesidad de extraer features manualmente.

Sus ventajas en este problema:
- Opera sobre la **ventana cruda** (100 × 6 = 600 valores) sin feature engineering.
- Invariante a pequeñas traslaciones temporales del gesto.
- `GlobalAveragePooling` provee regularización implícita y reduce parámetros.
- Arquitectura feed-forward: sin estados internos, sin bucles — ideal para cuantización.

```
Input (100, 6)
  └── Conv1D(16, kernel=3, relu)
  └── Conv1D(32, kernel=3, relu)
  └── Conv1D(64, kernel=3, relu)
  └── GlobalAveragePooling1D
  └── Dense(32, relu)
  └── Dense(4, softmax)         ← ~13,000 parámetros
```

**Accuracy real:** 97.93% float32 / **97.59% INT8**

#### Vía TFLite Micro

✅ **Ruta natural y única.** Todos los ops usados (`Conv1D`, `GlobalAveragePooling`, `Dense`, `Softmax`) son `TFLITE_BUILTINS` nativos. La cuantización INT8 post-entrenamiento produce una degradación de solo **0.34 pp**.

| Métrica | Valor |
|---|---|
| Tamaño modelo float32 | 52.1 KB |
| Tamaño modelo INT8 | **26.9 KB** |
| Compresión | 48.4% |
| Overhead TFLite Micro | ~200 KB |
| Flash total | **~227 KB** |
| Tensor Arena | 100 KB RAM |
| Latencia | ~14 ms |

#### Vía micromlgen

❌ **No aplicable.** `micromlgen` exporta modelos sklearn a C++. Las redes neuronales (Keras/TF) no son exportables por esta vía; requieren el runtime de TFLite Micro para ejecutarse.

#### Veredicto
**Modelo de producción recomendado.** Mayor accuracy de todos los modelos evaluados. Su único inconveniente es el overhead del runtime TFLite Micro (~200 KB de flash) y la latencia de ~14 ms, ambos completamente aceptables en este hardware.

---

### 2.2 LSTM — Evaluado y descartado ❌

#### Adecuación teórica
Las redes LSTM (Long Short-Term Memory) son arquitecturas recurrentes diseñadas para capturar dependencias temporales de largo alcance. En teoría son adecuadas para series de tiempo, pero presentan varios problemas estructurales en el contexto de TinyML:

- Mantienen **estados ocultos** que se propagan a través de los 100 timesteps — lo que genera memoria dinámica y ops de listas internas.
- Para el dataset actual (gestos simples de 1 segundo), el patrón temporal es capturado igualmente bien por la CNN con menor costo computacional.
- Requieren más RAM para el tensor arena (150 KB vs 100 KB de la CNN).

**Accuracy real:** 94.17% float32 / **94.14% dynamic-range**

#### Vía TFLite Micro

⚠️ **Viable con restricciones técnicas importantes.**

**Problema 1 — `LSTM` vs `RNN(LSTMCell)`:**
`tf.keras.layers.LSTM` genera internamente la operación `TensorListReserve`, que no es un `TFLITE_BUILTIN`. Solución: usar `RNN(LSTMCell(64))`, que genera el op nativo `UnidirectionalSequenceLSTM`.

```python
# ❌ Genera TensorListReserve — no compatible con TFLITE_BUILTINS:
layers.LSTM(64)

# ✅ Genera UnidirectionalSequenceLSTM — BUILTIN nativo:
layers.RNN(layers.LSTMCell(64))
```

**Problema 2 — Cuantización INT8 bloqueada:**
`RNN(LSTMCell)` introduce un op `WHILE` con `TensorListReserve` internos cuando el batch es dinámico. El pipeline de calibración INT8 no puede trazar el grafo congelado, fallando silenciosamente y guardando el modelo en float32 (86 KB) etiquetado como "INT8". La solución aplicada fue **dynamic-range quantization** desde el grafo congelado (pesos INT8, activaciones float32):

| Métrica | Valor |
|---|---|
| Tamaño modelo float32 | 86.0 KB |
| Tamaño modelo dynamic-range | **36.8 KB** |
| Compresión | 57.2% |
| Overhead TFLite Micro | ~200 KB |
| Flash total | **~237 KB** |
| Tensor Arena | 150 KB RAM |
| Degradación accuracy | −0.03 pp |

Para habilitar INT8 completo en un futuro re-entrenamiento se necesita `unroll=True` en la arquitectura, que desenrolla estáticamente los 100 timesteps eliminando el op `WHILE`.

#### Vía micromlgen

❌ **No aplicable.** Es una red neuronal; no exportable via micromlgen.

#### Por qué fue descartado

| Criterio | CNN 1D | LSTM |
|---|---|---|
| Accuracy (cuantizado) | **97.59%** | 94.14% |
| Flash total | **~227 KB** | ~237 KB |
| RAM (tensor arena) | **100 KB** | 150 KB |
| Complejidad de despliegue | INT8 estándar | Dynamic-range + workarounds |
| Tipo de cuantización | ✅ INT8 completo | ⚠️ Solo dynamic-range |

La CNN supera al LSTM en accuracy, ocupa menos flash, requiere menos RAM y tiene un pipeline de cuantización más limpio. Para el dataset actual (gestos simples de 1 segundo sin dependencias temporales complejas), no hay justificación para la complejidad adicional del LSTM.

#### Veredicto
**Descartado del proyecto.** El análisis completo de su evaluación, los bugs encontrados y las soluciones intentadas está documentado en [`technical_report.md`](technical_report.md), Sección 2.

---

### 2.3 Random Forest — Modelo en producción ✅

#### Adecuación teórica
Random Forest es un ensemble de 20 árboles de decisión entrenados sobre **32 features estadísticas** extraídas de la ventana IMU. Cada árbol vota y se toma la clase mayoritaria.

Sus ventajas en este proyecto:
- **Sin TFLite Micro:** el modelo exportado como C++ puro no tiene dependencias de runtime, ahorrando ~200 KB de flash.
- **Latencia sub-milisegundo:** el recorrido de 20 árboles es computacionalmente trivial en ARM Cortex-M4.
- **Interpretable:** la importancia de features revela que `ay_std` (18.6% Gini) y `az_std` son los discriminadores más potentes.

**Accuracy real:** **94.03%**

#### Vía TFLite Micro

❌ **No viable.** TensorFlow Decision Forests (TF-DF) puede exportar RF a TFLite, pero los ops requeridos (`SimpleMLInferenceOp`) no están implementados en TFLite **Micro** para ARM Cortex-M.

#### Vía micromlgen

✅ **Ruta natural y ya implementada.**

```python
from sklearn.ensemble import RandomForestClassifier
from micromlgen import port
clf = RandomForestClassifier(n_estimators=20, max_depth=8)
# → model_rf.h con 20 funciones predict_tree_N() encadenadas
```

| Métrica | Valor |
|---|---|
| Árboles | 20 (depth=8) |
| Tamaño `model_rf.h` (fuente C++) | ~576 KB texto |
| Tamaño compilado ARM Thumb2 | **~80–120 KB** |
| Flash total | **~80–120 KB** |
| RAM | < 1 KB (solo variables locales) |
| Latencia | **< 1 ms** |

> El archivo fuente `.h` de 576 KB parece grande, pero el compilador ARM (`arm-none-eabi-gcc`) lo optimiza agresivamente — el binario resultante es ~80–120 KB.

#### Veredicto
**Modelo de producción secundario.** Excelente trade-off cuando se prioriza latencia ultra-baja o cuando TFLite Micro no puede ser incluido por restricciones de flash. La brecha de accuracy respecto a CNN (3.56 pp) es el único punto en contra.

---

## 3. Modelos en evaluación (clásicos)

### 3.1 Regresión Lineal

#### Adecuación teórica
La Regresión Lineal es un modelo de **regresión** (salida continua), no de clasificación. Aplicarla a este problema requeriría codificar las 4 clases como enteros (0, 1, 2, 3) y usar `argmax` sobre las predicciones, lo cual:

- No produce probabilidades calibradas.
- Minimiza el error cuadrático medio (MSE), una función de pérdida incorrecta para clasificación.
- Genera sesgo sistemático hacia clases con mayor masa de datos.
- Asume linealidad entre las features y la *clase numérica*, una suposición sin sustento semántico.

En la práctica, este enfoque produce modelos con peor accuracy que Regresión Logística sin ninguna ventaja de interpretabilidad o footprint. La Regresión Logística resuelve exactamente el mismo problema de forma correcta.

#### Vías de despliegue

| Vía | Estado | Motivo |
|---|---|---|
| TFLite Micro | ❌ | Sin sentido para clasificación |
| micromlgen | ❌ | Soporta `LinearRegression` pero para tareas de regresión |

#### Veredicto
**Excluido.** No aplica para clasificación multiclase. Usar Regresión Logística en su lugar.

---

### 3.2 Regresión Logística

#### Adecuación teórica
A pesar del nombre, es un **clasificador**. Con función softmax extiende de forma natural a multiclase. Modela fronteras de decisión **lineales** en el espacio de features, lo que implica:

- Es el modelo más simple que puede funcionar si las clases son linealmente separables.
- Los gestos `CIRCULO` y `LADO` probablemente no son linealmente separables en el espacio de 32 features (el movimiento circular implica correlaciones no lineales entre ejes), lo que limitará el accuracy.
- Es un excelente **baseline lineal** para establecer el piso de rendimiento antes de modelos más complejos.

**Accuracy esperada:** 85–92%

#### Vía TFLite Micro

Regresión Logística es algebraicamente idéntica a una red neuronal de una sola capa densa:

```python
# Keras equivalent exacto:
model = Sequential([
    Input(shape=(32,)),
    Dense(4, activation='softmax')
])
```

Este modelo convierte limpiamente a TFLite con `TFLITE_BUILTINS`, cuantiza bien a INT8 y produce el modelo TFLite más pequeño posible (~1–2 KB). Sin embargo, el overhead de TFLite Micro (~200 KB) es completamente desproporcionado para un modelo de 132 parámetros.

| Vía TFLite | Estado |
|---|---|
| Conversión TFLite | ✅ Trivial |
| Quantización INT8 | ✅ Sin degradación notable |
| Tamaño del modelo | ~1–2 KB |
| Overhead runtime | ~200 KB (desproporcionado) |
| **Recomendación** | ❌ Tecnicamente factible, pero antieconómico |

#### Vía micromlgen

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000, C=1.0)
```

`micromlgen` exporta el modelo como una simple multiplicación de matrices en C++. El resultado es un archivo de ~1–2 KB con 0 dependencias de runtime.

| Vía micromlgen | Estado |
|---|---|
| Soporte `sklearn` | ✅ `LogisticRegression` soportado |
| Tamaño C++ | ~1–2 KB |
| Latencia inferencia | < 0.1 ms (128 multiplicaciones + softmax) |
| Flash total | ~2 KB (sin overhead de runtime) |
| **Recomendación** | ✅ Ruta óptima para este modelo |

#### Veredicto
**Viable y recomendado como baseline lineal.** Implementar via `micromlgen`. Su valor principal es establecer un piso de comparación: si SVM o RF superan a Logística en más de 5 pp, eso justifica su complejidad adicional.

---

### 3.3 K-Nearest Neighbors (KNN)

#### Adecuación teórica
KNN es un clasificador **no paramétrico** — no tiene fase de entrenamiento; guarda todos los ejemplos y clasifica por votación de los `k` vecinos más cercanos en el espacio de features. Sus implicaciones para este proyecto:

- Con 32 features, la distancia euclidiana es semánticamente razonable (features normalizadas).
- No asume ninguna distribución de datos — puede capturar fronteras de decisión arbitrariamente complejas.
- Su debilidad es el **costo de memoria y cómputo en inferencia**: escala con el tamaño del conjunto de entrenamiento.

**Accuracy esperada:** 88–93% (con k=5, 32 features, sin subsampling)

#### Problema crítico de memoria

El conjunto de entrenamiento tiene 14,747 ventanas. Almacenarlas todas en el Arduino requiere:

```
14,747 ventanas × 32 features × 4 bytes (float32) = 1,887,616 bytes ≈ 1.8 MB
```

Esto excede la capacidad total de flash del Arduino (1 MB) y es absolutamente inviable.

**Mitigación posible — Subsampling:**
Reducir a ~400–600 muestras representativas (ej. mediante K-Means sobre el dataset de entrenamiento para obtener prototipos por clase):

```
500 prototipos × 32 features × 4 bytes = 64 KB
```

Esto es técnicamente viable en flash, pero la accuracy caería significativamente al perder cobertura del espacio de features.

#### Vía TFLite Micro

❌ **No viable.** KNN no puede expresarse como un grafo de cómputo TensorFlow diferenciable. No existe ningún op de TFLite Micro que implemente búsqueda de vecinos más cercanos.

#### Vía micromlgen

`micromlgen` soporta `sklearn.neighbors.KNeighborsClassifier`, pero la limitación de memoria persiste independientemente de la herramienta de exportación.

| Vía micromlgen | Estado |
|---|---|
| Soporte `sklearn` | ✅ Técnicamente soportado |
| Con dataset completo | ❌ ~1.8 MB — no cabe |
| Con 500 prototipos | ⚠️ ~64 KB — cabe, pero accuracy cae |
| Latencia (500 pts) | ~1–2 ms |
| **Recomendación** | ❌ No recomendado para producción |

#### Veredicto
**Viable en Python, no recomendado en Arduino.** Útil para análisis offline (validar que las features estadísticas tienen estructura geométrica apropiada), pero la restricción de memoria lo hace impráctcio en hardware sin comprometer significativamente la precisión.

---

### 3.4 Naive Bayes (Gaussian)

#### Adecuación teórica
Gaussian Naive Bayes (GNB) modela la probabilidad condicional de cada feature dado la clase como una distribución gaussiana, asumiendo **independencia condicional** entre features:

```
P(clase | features) ∝ P(clase) × ∏ P(feature_i | clase)
```

La asunción de independencia es la debilidad central en este proyecto: los ejes del IMU están altamente correlacionados durante un gesto. Por ejemplo, durante un movimiento circular, `ax_mean`, `ay_mean` y `az_std` varían de forma conjunta y correlacionada. GNB ignorará estas correlaciones.

En la práctica, GNB funciona mejor de lo que la teoría sugiere porque las distribuciones gausianas por clase capturan suficiente señal incluso cuando la independencia no se cumple estrictamente.

**Accuracy esperada:** 80–88%

#### Vía TFLite Micro

❌ **No viable como implementación nativa.** GNB no es una red neuronal y no tiene representación directa como grafo TF diferenciable. Técnicamente podría implementarse como operaciones matriciales (log-likelihood computation), pero:
- Requeriría ops personalizados o `SELECT_TF_OPS`.
- El overhead de TFLite Micro (~200 KB) es completamente desproporcionado.
- No hay ventaja alguna sobre micromlgen.

#### Vía micromlgen

`micromlgen` soporta `sklearn.naive_bayes.GaussianNB`. El modelo exportado solo necesita almacenar media y varianza por feature por clase:

```
4 clases × 32 features × 2 parámetros (μ, σ²) × 4 bytes = 1,024 bytes ≈ 1 KB
```

El modelo C++ más pequeño posible después de Regresión Logística.

| Vía micromlgen | Estado |
|---|---|
| Soporte `sklearn` | ✅ `GaussianNB` soportado |
| Tamaño C++ | ~1 KB |
| Latencia inferencia | < 0.1 ms |
| Flash total | ~1 KB (sin overhead de runtime) |
| **Recomendación** | ✅ Viable, mínimo footprint |

#### Veredicto
**Viable con limitaciones de accuracy.** Tiene el footprint más pequeño de todos los modelos analizados. Su valor principal es como referencia de complejidad mínima: si el accuracy de GNB está cerca del de RF (94%), eso indicaría que las features estadísticas capturan casi toda la información necesaria para la tarea. Si hay una brecha grande, confirma que se necesita modelado de interacciones (non-lineal).

---

### 3.5 Árbol de Decisión

#### Adecuación teórica
Un Árbol de Decisión aprende umbrales sobre features individuales que particionan el espacio de forma recursiva. Sus ventajas para este proyecto:

- **Interpretable:** cada predicción tiene una ruta explicable ("si `ay_std > 0.31` y `az_mean < -0.2`, entonces `CIRCULO`").
- **No asume linealidad:** puede capturar fronteras no lineales.
- **Maneja correlaciones implícitamente:** selecciona las features más discriminativas en cada nodo.
- Con `max_depth` controlada, resiste el overfitting razonablemente bien.

El Árbol de Decisión es exactamente un árbol del Random Forest actual. Su accuracy será menor que RF (sin beneficio del ensemble) pero superior a los modelos lineales.

**Accuracy esperada:** 90–94%

#### Vía TFLite Micro

❌ **No viable.** `TensorFlow Decision Forests` (TF-DF) existe y puede exportar árboles a TFLite, pero los ops necesarios (`CategoricalFeatureSelector`, `SimpleMLInferenceOp`) **no están implementados en TFLite Micro** para microcontroladores.

> Nota: TF-DF funciona en TFLite para mobile (Android/iOS), pero no en el subset de ops disponibles para ARM Cortex-M.

#### Vía micromlgen

`micromlgen` soporta `sklearn.tree.DecisionTreeClassifier`. El tamaño del modelo depende de la profundidad:

| `max_depth` | Nodos (aprox.) | Tamaño C++ |
|---|---|---|
| 8 | ~255 | ~8–12 KB |
| 10 | ~1,023 | ~25–40 KB |
| 15 | Saturado por datos | ~80–150 KB |
| Sin límite | Overfitting completo | > 500 KB |

Se recomienda `max_depth=10` para equilibrar accuracy y tamaño.

| Vía micromlgen | Estado |
|---|---|
| Soporte `sklearn` | ✅ `DecisionTreeClassifier` soportado |
| Tamaño C++ (depth=10) | ~25–40 KB |
| Latencia inferencia | < 0.05 ms (solo traversal del árbol) |
| Flash total | ~30–45 KB |
| **Recomendación** | ✅ Viable, útil como ablación del RF |

#### Veredicto
**Viable.** Dado que Random Forest ya está implementado, añadir un único árbol de decisión tiene valor principalmente como **comparativa de ablación**: cuantifica exactamente cuánto accuracy se gana por usar un ensemble vs. un árbol solo. También útil si el flash está tan al límite que un solo árbol (30 KB) es preferible al RF completo (~150 KB).

---

### 3.6 SVM (Support Vector Machine)

#### Adecuación teórica
SVM es, teóricamente, el clasificador clásico más adecuado para este problema. Busca el **hiperplano de máximo margen** que separa las clases, y con el kernel RBF puede capturar fronteras de decisión altamente no lineales.

Sus ventajas específicas para clasificación de gestos:
- El kernel RBF proyecta los 32 features estadísticos a un espacio de alta dimensión donde la separación lineal es más fácil — ideal para gestos que no son linealmente separables en el espacio original.
- Es robusto a overfitting con el parámetro `C` (regularización) correctamente ajustado.
- Bien estudiado teóricamente: el margen máximo minimiza el error de generalización.

**Accuracy esperada:**
- `LinearSVC`: 88–93% (similar a Regresión Logística, pero con margen máximo).
- `SVC(kernel='rbf')`: **94–97%** — potencialmente competitivo con la CNN 1D.

#### Vía TFLite Micro

❌ **No viable de forma práctica.**

- **LinearSVC:** Su función de decisión (`w · x + b`) es algebraicamente una capa `Dense(4)`, equivalente a Regresión Logística. Tecnicamente exportable a TF/TFLite, pero sin ninguna ventaja sobre usar Regresión Logística directamente en Keras.
- **SVC(RBF):** Requiere computar el kernel entre el input y todos los support vectors. En TF esto implicaría almacenar todos los SVs en constantes del grafo y hacer `n_sv` operaciones de distancia — equivalente funcional a lo que hace micromlgen, pero con el overhead innecesario del runtime TFLite (~200 KB).

#### Vía micromlgen

`micromlgen` soporta tanto `LinearSVC` como `SVC` con kernels rbf, poly y sigmoid.

El tamaño del modelo para `SVC(RBF)` depende del número de support vectors resultantes del entrenamiento:

| Support Vectors | Tamaño C++ (32 features) |
|---|---|
| 100 SVs | ~13 KB |
| 300 SVs | ~38 KB |
| 500 SVs | ~64 KB |
| 1,000 SVs | ~128 KB |

Con parámetros `C=10, gamma='scale'`, es típico obtener 200–600 SVs en este dataset — un rango completamente manejable para el flash disponible.

| Vía micromlgen | Estado |
|---|---|
| Soporte `LinearSVC` | ✅ |
| Soporte `SVC(rbf)` | ✅ |
| Tamaño C++ típico | 38–64 KB (300–500 SVs) |
| Latencia inferencia | 0.5–2 ms |
| Flash total | ~40–70 KB |
| **Recomendación** | ✅ Alta prioridad — mejor modelo clásico esperado |

#### Veredicto
**Altamente recomendado. ✅ Implementado** — `training/train_svm_rbf.py` disponible. `SVC(kernel='rbf')` es el modelo clásico con mayor potencial de accuracy en este proyecto. Si alcanza 95%+, ofrece una alternativa competitiva a la CNN con menos complejidad de pipeline (sin TFLite, sin cuantización, sin `xxd`), latencia sub-milisegundo y un footprint manejable via `micromlgen`. A diferencia de `LinearSVC`, micromlgen exporta `SVC(rbf)` de forma nativa sin necesidad de proxy.

---

## 4. Tabla comparativa resumen

### Modelos del proyecto

| Modelo | Estado | TFLite Micro | micromlgen | Acc. real | Flash total | Latencia |
|---|---|---|---|---|---|---|
| CNN 1D | ✅ Producción | ✅ (INT8) | ❌ | **97.59%** | ~227 KB | ~14 ms |
| LSTM | ❌ Descartado | ⚠️ (dynamic-range) | ❌ | 94.14% | ~237 KB | > 14 ms |
| Random Forest | ✅ Producción | ❌ | ✅ | 94.03% | ~80–120 KB | < 1 ms |

### Modelos clásicos candidatos

| Modelo | Arduino viable | TFLite Micro | micromlgen | Acc. estimada | Flash total | Latencia |
|---|---|---|---|---|---|---|
| Regresión Lineal | ❌ | ❌ | ❌ | — | — | — |
| Regresión Logística | ✅ | ⚠️ antieconómico | ✅ | 85–92% | ~2 KB | < 0.1 ms |
| KNN | ⚠️ subsampling | ❌ | ⚠️ memoria | 88–93% | ~64 KB+ | ~1–2 ms |
| Naive Bayes | ✅ | ❌ | ✅ | 80–88% | ~1 KB | < 0.1 ms |
| Árbol de Decisión | ✅ | ❌ | ✅ | 90–94% | ~30–45 KB | < 0.05 ms |
| SVM (Linear) | ✅ | ⚠️ antieconómico | ✅ | 88–93% | ~2 KB | < 0.1 ms |
| SVM (RBF) | ✅ | ❌ | ✅ | **94–97%** | ~40–70 KB | < 2 ms |

---

## 5. Rutas de despliegue

### Por qué TFLite no es la ruta para modelos clásicos

TFLite Micro es un runtime diseñado exclusivamente para **redes neuronales**. Sus operaciones nativas (Conv, Dense, LSTM, Softmax, etc.) no tienen equivalentes para los algoritmos clásicos. El único modelo que mapea naturalmente es Regresión Logística (= Dense sin capas ocultas), pero incluso en ese caso el overhead del runtime (~200 KB) hace que micromlgen sea una elección claramente superior.

```
┌─────────────────────────────────────────────────────────┐
│                    Ruta de Despliegue                   │
├─────────────────┬───────────────────────────────────────┤
│  Modelo         │  Ruta óptima                          │
├─────────────────┼───────────────────────────────────────┤
│  CNN 1D         │  TFLite Micro (única opción)          │
│  LSTM           │  TFLite Micro (con workarounds)       │
│  Random Forest  │  micromlgen (única opción)            │
│  Reg. Logística │  micromlgen >> TFLite                 │
│  Naive Bayes    │  micromlgen (única opción viable)     │
│  Árbol Decisión │  micromlgen (única opción viable)     │
│  SVM Linear     │  micromlgen >> TFLite                 │
│  SVM RBF        │  micromlgen (única opción viable)     │
│  KNN            │  micromlgen (solo con subsampling)    │
└─────────────────┴───────────────────────────────────────┘
```

### Flujo de implementación (micromlgen)

Para cualquier modelo sklearn nuevo, el pipeline es idéntico al RF ya implementado:

```python
# 1. Extraer features estadísticas (mismo código que train_rf.py)
X_train, y_train = extract_features(windows_train)

# 2. Entrenar modelo
clf = SVC(kernel='rbf', C=10, gamma='scale')
clf.fit(X_train, y_train)

# 3. Exportar a C++
from micromlgen import port
with open('models/<model>/model_<name>.h', 'w') as f:
    f.write(port(clf, classmap={0: 'CIRCULO', 1: 'DEFAULT', 2: 'LADO', 3: 'QUIETO'}))
```

El archivo `model_<name>.h` se incluye directamente en el firmware Arduino sin ninguna dependencia adicional.

---

## 6. Prioridad de implementación recomendada

Dado el estado actual del proyecto (CNN 97.59%, RF 94.03%), los modelos que aportarían mayor valor incremental son:

### Prioridad 1 — SVM (RBF) ⭐ ✅ Implementado
- **Potencial de competir con CNN** (94–97%).
- Sin dependencia de TFLite: pipeline más simple.
- Único modelo clásico que podría reducir la brecha con la CNN.
- `training/train_svm_rbf.py` disponible — micromlgen exporta `SVC(rbf)` directamente, sin proxy.
- Resultados pendientes de entrenamiento.

### Prioridad 2 — Regresión Logística
- **Baseline lineal de referencia**: establece el piso de rendimiento.
- Footprint mínimo (~2 KB), latencia sub-ms.
- Implementación trivial; confirma si el problema es linealmente separable.

### Prioridad 3 — Árbol de Decisión (ablación)
- **Comparativa directa contra RF**: cuantifica el beneficio del ensemble.
- Si un solo árbol alcanza 92%+, el ensemble de 20 que usa RF tiene overhead justificable.
- Útil para documentación y análisis del proyecto.

### No recomendados para implementación
- **KNN**: restricción de memoria insalvable en Arduino.
- **Naive Bayes**: accuracy inferior con violación de asunciones; valor analítico limitado.
- **Regresión Lineal**: no aplica para clasificación.

---

## 7. Conclusiones

1. **TFLite Micro es exclusivo para redes neuronales.** Es el runtime correcto para CNN y LSTM, pero inadecuado para modelos clásicos. De los modelos clásicos analizados, solo Regresión Logística tiene un equivalente natural en TFLite (una capa Dense), y aún en ese caso micromlgen es preferible por no cargar los ~200 KB del runtime.

2. **LSTM fue descartado por razones de accuracy y complejidad, no solo de tamaño.** Aunque técnicamente desplegable via TFLite Micro (con `RNN(LSTMCell)` y dynamic-range quantization), su accuracy (94.14%) es inferior a la CNN (97.59%) con mayor footprint y pipeline más frágil. El caso LSTM ilustra que la adecuación de una arquitectura a TinyML no es solo sobre operaciones compatibles, sino sobre el ecosistema completo de cuantización.

3. **micromlgen democratiza el despliegue de ML clásico en Arduino.** Cualquier modelo sklearn compatible puede exportarse a C++ puro sin dependencias, con footprints de 1–150 KB. Esto lo convierte en la ruta natural para RF, SVM, Logística y Árboles de Decisión.

4. **SVM con kernel RBF es el candidato clásico más prometedor.** Es el único que, teóricamente, podría alcanzar accuracies competitivos con la CNN (94–97%), con la ventaja de eliminar la dependencia de TFLite Micro y su overhead de ~200 KB.

5. **El verdadero trade-off no es accuracy sino complejidad de pipeline.** CNN: TFLite Micro + cuantización + `xxd` + tensor arena de 100 KB. Modelos via micromlgen: extracción de 32 features + un único `.h`. Para aplicaciones donde 94–96% es suficiente, los modelos clásicos ofrecen una cadena de despliegue significativamente más simple y auditable.

6. **KNN es la excepción que confirma la regla.** A pesar de ser técnicamente soportado por micromlgen, la restricción de memoria del Arduino lo hace inviable sin comprometer gravemente la precisión mediante subsampling agresivo.

---

## Referencias

- `micromlgen` documentation: https://github.com/eloquentarduino/micromlgen
- Scikit-learn: Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
- TensorFlow Lite Micro supported ops: https://www.tensorflow.org/lite/microcontrollers/build_convert
- TF Decision Forests: https://www.tensorflow.org/decision_forests (mobile TFLite only, not Micro)
- Banos et al. (2014). *Window Size Impact in Human Activity Recognition*. Sensors, 14(4), 6474–6499.
