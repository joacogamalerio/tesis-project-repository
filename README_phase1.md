# Fase 1 - Colab portable a Jupyter local

Esta versión deja el flujo **monolítico** en un solo notebook/script y elimina la dependencia obligatoria de `google.colab`.

## Qué cambió

- Se eliminó el import obligatorio de `google.colab.drive`.
- Se agregó `setup_save_dir(...)` para que funcione:
  - en Colab
  - en Jupyter local
- Se mantiene autodetección de GPU con `torch.cuda.is_available()`.
- El dataset MNIST ahora usa `./data`.
- Los resultados se guardan por defecto en `./runs`.
- `gTTS` quedó como dependencia **opcional**. Si no está o falla, el flujo no se rompe.

## Requisitos

- Linux
- Python 3.10 o 3.11
- JupyterLab o Notebook
- GPU NVIDIA con drivers instalados si quieres acelerar con CUDA

## Crear environment

### Opción con venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Instalar PyTorch

#### Con GPU CUDA
Antes de instalar PyTorch, verifica tu instalación NVIDIA:
```bash
nvidia-smi
```

Luego instala PyTorch con soporte CUDA. Ejemplo:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> Si tu máquina usa otra variante de CUDA soportada por PyTorch, cambia el índice por el correspondiente.

#### Solo CPU
```bash
pip install torch torchvision torchaudio
```

### Instalar el resto
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

## Archivos

- `mnist_training_phase1_local_jupyter.ipynb`: notebook listo para Jupyter
- `mnist_training_phase1_local_jupyter.py`: misma lógica en script
- `requirements.txt`
- `.gitignore`

## Crear repo y subir a una rama

```bash
mkdir mnist-regularization-experiments
cd mnist-regularization-experiments
git init
git checkout -b phase1-local-jupyter
```

Copiá estos archivos dentro del repo y luego:

```bash
git add .
git commit -m "Phase 1: make Colab notebook runnable in local Jupyter"
git remote add origin <URL_DE_TU_REPO>
git push -u origin phase1-local-jupyter
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

- `gTTS` necesita Internet para generar audio.
- Si no quieres audio, usa:
```python
play_training_finished_audio(enable_audio=False)
```
- Si en Colab quieres volver a guardar en Drive:
```python
SAVE_DIR = setup_save_dir(
    local_save_dir="./runs",
    colab_drive_path="/content/drive/MyDrive/Tesis/2026-experimentacion oficial",
    mount_drive_if_needed=True,
)
```
