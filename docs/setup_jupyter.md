# Ejecucion local en Jupyter

Notas para ejecutar los notebooks del repositorio en un entorno local de Jupyter.

## Requisitos

- Linux o un entorno compatible con Jupyter
- Python 3.10 o 3.11
- JupyterLab o Notebook
- GPU NVIDIA con drivers instalados, si se desea acelerar el entrenamiento con CUDA

## Crear entorno

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## Instalar PyTorch

Antes de instalar PyTorch, verifica tu instalación NVIDIA:

```bash
nvidia-smi
```

Instala PyTorch con soporte CUDA siguiendo la variante recomendada por PyTorch para tu sistema. Ejemplo:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Para CPU:

```bash
pip install torch torchvision torchaudio
```

## Instalar dependencias auxiliares

```bash
pip install -r requirements.txt
```

## Levantar Jupyter
```bash
jupyter lab
```

o
```bash
jupyter notebook
```

## Verificar que la GPU se vea desde PyTorch

Dentro del notebook:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Sugerencia de ejecución

Primero probá un entrenamiento corto:
```python
history_overfit, model_overfit = train_overfitnet(
    num_epochs=3,
    batch_size=128,
    normalize=False,
    plot_curves=True
)
```

Después corré el grid search.

## Notas

- Los datos de MNIST se descargan en `./data`.
- Los resultados se guardan por defecto en `./runs`.
- `gTTS` es una dependencia opcional usada solo para avisos de audio.
