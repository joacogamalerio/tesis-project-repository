# Procedimiento corto para reproducir la figura de sparsity

1. Entrenar un autoencoder de una sola capa oculta sobre MNIST con arquitectura `784 -> 256 -> 784` y activaciones **ReLU** en la hidden layer. En la versión con dropout, usar `p = 0.5` en la capa oculta. citeturn663853search0turn993831view2
2. Tomar **un minibatch aleatorio del test set**. El paper no usa todo el test set para esa figura, sino un minibatch aleatorio. citeturn993831view2turn993831view3
3. Para ese único minibatch, computar las activaciones ocultas `h` de tamaño `[batch_size, 256]`. citeturn993831view2turn993831view3
4. Graficar dos histogramas:
   - **Izquierda**: la distribución de la **media de activación por unidad oculta** a través del minibatch, o sea `mean(h, dim=0)`.
   - **Derecha**: la distribución de **todas las activaciones** del mismo minibatch, o sea `h.flatten()`. citeturn993831view2turn993831view3
5. Para el modelo con dropout, **no escalar los pesos hacia abajo** al momento de hacer la figura. El paper lo aclara explícitamente para que la comparación sea justa. citeturn993831view2turn993831view3

Eso es exactamente lo que implementa `plot_representational_sparsity_from_model_path(...)` en la notebook.
