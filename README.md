# Efecto de la dilucion aleatoria en aprendizaje profundo

Repositorio de apoyo para la tesis de Licenciatura en Fisica de Joaquin Ignacio Gamalerio, presentada en FAMAF, Universidad Nacional de Cordoba.

El trabajo estudia mecanismos de regularizacion estocastica en redes neuronales feed-forward densas. En particular, compara una arquitectura base sin regularizacion estocastica con variantes basadas en Dropout, DropConnect y BoostDropout, un mecanismo propuesto que reemplaza la supresion binaria por una modulacion multiplicativa no binaria de las activaciones.

## Contenido

- `main_protocol.ipynb`: protocolo experimental principal de clasificacion sobre MNIST. Incluye preparacion reproducible, definicion de arquitecturas, particion de datos, busqueda de hiperparametros y evaluacion final multisemilla.
- `autoencoder_protocol.ipynb`: protocolo complementario con autoencoders densos para analizar cualitativamente representaciones internas, esparsidad representacional, coadaptacion y reconstrucciones.
- `reconstruct_gridsearch_comparison_plots.ipynb`: reconstruccion de figuras comparativas de la busqueda de hiperparametros a partir de artefactos guardados.
- `reconstruct_final_evaluation_plots.ipynb`: reconstruccion de figuras de evaluacion final a partir de corridas persistidas.
- `mnist_training_local_jupyter.py`: script historico con logica de entrenamiento local.
- `requirements.txt`: dependencias Python auxiliares. PyTorch y torchvision deben instalarse segun la plataforma disponible.
- `docs/`: documentacion de soporte y copia del manuscrito de tesis.

## Protocolos experimentales

El protocolo principal compara redes densas de topologia comun `784 -> 256 -> 256 -> 256 -> 10` sobre un subconjunto controlado de MNIST. La comparacion se organiza en dos etapas: una busqueda de hiperparametros con multiples semillas por configuracion y una evaluacion final con nuevas semillas compartidas entre arquitecturas. Los resultados se separan por criterio de seleccion de modelo, incluyendo mejor modelo de validacion y early stopping.

El protocolo complementario usa autoencoders de arquitectura `784 -> 256 -> 784`. Su objetivo no es optimizar exhaustivamente la reconstruccion, sino inspeccionar como los mecanismos de regularizacion modifican las caracteristicas aprendidas, la esparsidad de activaciones y las reconstrucciones sobre MNIST.

## Instalacion

Crear un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Instalar PyTorch y torchvision segun el entorno. Para CPU:

```bash
pip install torch torchvision torchaudio
```

Para GPU con CUDA, usar el indice recomendado por PyTorch para la version instalada del driver. Luego instalar el resto de dependencias:

```bash
pip install -r requirements.txt
```

## Ejecucion

Abrir Jupyter desde la raiz del repositorio:

```bash
jupyter lab
```

Ejecutar primero celdas de preparacion y definicion. Las celdas de entrenamiento final y busqueda de hiperparametros pueden tener costo computacional alto; conviene lanzar solo el bloque correspondiente al experimento que se quiera reproducir.

## Artefactos

Las corridas guardan resultados bajo `runs/`, incluyendo pesos de modelos, historiales, figuras, tablas y metadata. Esa carpeta puede crecer rapidamente y no es necesaria para entender el codigo base, pero permite reconstruir las figuras y tablas generadas por los notebooks de reconstruccion.

## Documentacion

- `docs/Tesis_Gamalerio_Joaquín.pdf`: manuscrito completo de la tesis.
- `docs/setup_jupyter.md`: notas de instalacion y ejecucion local.
- `docs/sparsity_visualization.md`: procedimiento usado para las figuras de esparsidad representacional del autoencoder.
