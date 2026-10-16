# Autoencoder desde cero con NumPy

Implementación de un **autoencoder (perceptrón multicapa)** construido completamente desde cero con NumPy, sin usar frameworks de deep learning. Se entrena sobre el dataset **MNIST** para aprender a comprimir y reconstruir imágenes de dígitos escritos a mano.

## Contenido

- `autoencoder_desde_cero.ipynb`: notebook con el código completo (modelo, entrenamiento y evaluación).

## ¿Qué hace?

El autoencoder recibe una imagen de 28×28 píxeles (aplanada a 784 valores), la comprime a una representación de menor dimensión y luego intenta reconstruirla lo más parecida posible a la original.

```
784 → 128 → 64 → 128 → 784
entrada   (codificador)  cuello de botella  (decodificador)  salida
```

## Detalles de la implementación

| Componente | Descripción |
|---|---|
| Arquitectura | `[784, 128, 64, 128, 784]` |
| Activación capas ocultas | `tanh` |
| Activación capa de salida | `sigmoide` (salida en el rango [0, 1]) |
| Inicialización de pesos | Xavier/Glorot uniforme |
| Función de pérdida | Error cuadrático medio (MSE) |
| Optimización | Gradiente descendente con mini-batches |
| Backpropagation | Implementado manualmente |

La clase `autoEncoder` incluye los métodos:

- `feed_forward`: propagación hacia adelante, guardando netas y activaciones.
- `back_propagation`: cálculo de gradientes de pesos y sesgos.
- `gradiente_descendente`: actualización de parámetros.
- `fit`: ciclo de entrenamiento con mezcla aleatoria de datos y mini-batches; devuelve el historial de pérdida.
- `predict`: reconstrucción de nuevas imágenes.

## Datos

- **Dataset:** MNIST (`mnist_784`), descargado con `fetch_openml`.
- **Preprocesamiento:** píxeles normalizados a [0, 1] dividiendo entre 255.
- **Partición:** 60.000 imágenes para entrenamiento y 10.000 para prueba (estratificada, `random_state=42`).

## Hiperparámetros de entrenamiento

| Parámetro | Valor |
|---|---|
| Épocas | 150 |
| Tasa de aprendizaje | 1 |
| Tamaño de batch | 128 |

## Resultados

Se compara el autoencoder contra un *baseline* que siempre predice la imagen promedio del conjunto de entrenamiento:

| Modelo | MSE en test |
|---|---|
| Baseline (imagen promedio) | ≈ 0.0672 |
| Autoencoder | ≈ 0.0238 |

El autoencoder reduce el error de reconstrucción en aproximadamente un 65 % respecto al baseline. Al final del notebook se muestra una comparación visual entre una imagen original y su reconstrucción.
