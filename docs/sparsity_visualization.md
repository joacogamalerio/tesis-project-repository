# Procedimiento corto para reproducir la figura de sparsity

1. Entrenar un autoencoder de una sola capa oculta sobre MNIST con arquitectura `784 -> 256 -> 784` y activaciones **ReLU** en la capa latente. En la versión con Dropout, usar la probabilidad definida para la comparación correspondiente.
2. Tomar **un minibatch aleatorio del conjunto de test**.
3. Para ese único minibatch, computar las activaciones ocultas `h` de tamaño `[batch_size, 256]`.
4. Graficar dos histogramas:
   - **Izquierda**: la distribución de la **media de activación por unidad oculta** a través del minibatch, o sea `mean(h, dim=0)`.
   - **Derecha**: la distribución de **todas las activaciones** del mismo minibatch, o sea `h.flatten()`.
5. Para la variante con Dropout, no aplicar un escalado adicional de pesos al momento de construir la figura. La comparación debe reflejar las activaciones obtenidas bajo el procedimiento definido para evaluación.

Este procedimiento es el que implementa `plot_representational_sparsity_from_model_path(...)` en `autoencoder_protocol.ipynb`.
