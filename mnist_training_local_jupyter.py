"""
Script de entrenamiento compatible con Google Colab y Jupyter local.

- Evita dependencias obligatorias de Colab.
- Usa GPU local automáticamente si PyTorch la detecta.
- Guarda resultados en un directorio configurable.
"""

# =========================
# INSTALLS (solo referencia)
# =========================
# En notebook local, instalar primero:
# pip install -r requirements.txt

# =========================
# IMPORTS
# =========================
import os
import json
import random
import datetime
import shutil
import hashlib
import uuid
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from IPython.display import Audio, display

# gTTS es opcional fuera de Colab
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    gTTS = None
    GTTS_AVAILABLE = False


# =========================
# PLATFORM / PATH SETUP
# =========================
def is_running_in_colab():
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def setup_save_dir(
    local_save_dir="./runs",
    colab_drive_path="/content/drive/MyDrive/Tesis/2026-experimentacion oficial",
    mount_drive_if_needed=False,
):
    """
    Devuelve un Path donde se guardarán resultados.

    - En Colab:
        * Si mount_drive_if_needed=True, monta Drive y usa colab_drive_path
        * Si no, usa el filesystem local del runtime
    - En local:
        * Usa local_save_dir
    """
    if is_running_in_colab():
        if mount_drive_if_needed:
            try:
                from google.colab import drive  # type: ignore
                drive.mount("/content/drive")
                save_dir = Path(colab_drive_path)
            except Exception as e:
                print(f"⚠️ No se pudo montar Google Drive: {e}")
                save_dir = Path("./runs")
        else:
            save_dir = Path("./runs")
    else:
        save_dir = Path(local_save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir.resolve()


# Cambia esto si quieres otra ruta local
SAVE_DIR = setup_save_dir(
    local_save_dir="./runs",
    colab_drive_path="/content/drive/MyDrive/Tesis/2026-experimentacion oficial",
    mount_drive_if_needed=False,  # en Colab, poner True si quieres guardar en Drive
)

DATA_ROOT = Path("./data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

print(f"📁 SAVE_DIR: {SAVE_DIR}")
print(f"📁 DATA_ROOT: {DATA_ROOT.resolve()}")


# =========================
# REPRODUCIBILITY
# =========================
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Asegura RNG determinista también en workers del DataLoader.
    """
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================
# DEVICE CHECK
# =========================
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    if device.type == "cuda":
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️ No se detectó GPU CUDA. Se usará CPU.")
    return device


device = get_device()


# =========================
# TRAINING UTILS
# =========================
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item() * y.size(0)
            correct += (pred.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def build_model(model_name, hidden_size=256, dropout_p=0.5, lambd=0.5, mask_normalization=False):
    """
    Factory para crear cualquiera de los 3 modelos.
    """
    model_name = model_name.lower()

    if model_name == "overfitnet":
        return OverfitNet(hidden_size=hidden_size)

    if model_name == "dropoutnetv2":
        return DropoutNetV2(hidden_size=hidden_size, dropout_p=dropout_p)

    if model_name == "boostdropoutnetv2":
        return BoostDropoutNetV2(
            hidden_size=hidden_size,
            dropout_p=dropout_p,
            lambd=lambd,
            mask_normalization=mask_normalization
        )

    raise ValueError(
        "model_name debe ser uno de: "
        "'OverfitNet', 'DropoutNetV2', 'BoostDropoutNetV2'"
    )


def get_optimizer(model, lr=1e-3, weight_decay=0.0):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def init_history():
    return {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }


def plot_history(history, model_name="Model"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curves - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_json(data, path):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_model_artifact(
    model,
    history,
    training_dir,
    artifact_type,
    epoch,
    val_loss,
    training_id,
    run_metadata,
    extra_info=None
):
    """
    artifact_type:
      - 'best_model'
      - 'last_model'
      - 'early_stopping'
    """
    training_dir = Path(training_dir)
    model_path = training_dir / f"{artifact_type}.pth"
    meta_path = training_dir / f"{artifact_type}.json"

    artifact_metadata = {
        "training_id": training_id,
        "artifact_type": artifact_type,
        "epoch": epoch,
        "val_loss": float(val_loss),
        "run_metadata": run_metadata,
        "history": history
    }

    if extra_info is not None:
        artifact_metadata["extra_info"] = extra_info

    torch.save(model.state_dict(), model_path)
    save_json(artifact_metadata, meta_path)

    print(f"💾 Guardado {artifact_type} en: {model_path}")


def save_history_csv(history, training_dir):
    training_dir = Path(training_dir)
    df = pd.DataFrame(history)
    csv_path = training_dir / "history.csv"
    df.to_csv(csv_path, index=False)
    print(f"📝 History CSV guardado en: {csv_path}")


def save_training_plot(history, training_dir, metric="loss"):
    training_dir = Path(training_dir)
    plt.figure(figsize=(8, 5))

    if metric == "loss":
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        file_name = "plot_loss.png"

    elif metric == "accuracy":
        plt.plot(history["train_acc"], label="Train Accuracy")
        plt.plot(history["val_acc"], label="Validation Accuracy", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves")
        file_name = "plot_accuracy.png"

    else:
        raise ValueError("metric debe ser 'loss' o 'accuracy'")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = training_dir / file_name
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"🖼️ Gráfico guardado en: {save_path}")


def generate_training_id(model_name):
    started_at = datetime.datetime.now()
    started_at_str = started_at.strftime("%Y-%m-%d-%H-%M-%S-%f")
    raw = f"{model_name}-{started_at_str}-{uuid.uuid4().hex}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    training_id = f"{started_at_str}-{short_hash}"
    return training_id, started_at_str


def make_training_dir(save_dir, training_id):
    training_dir = Path(save_dir) / training_id
    training_dir.mkdir(parents=True, exist_ok=True)
    return training_dir


def build_run_metadata(
    training_id,
    training_dir,
    started_at,
    requested_model_name,
    instantiated_model_name,
    instantiated_model_class,
    lr,
    hidden_size,
    batch_size,
    num_epochs,
    early_stopping_patience,
    train_fraction,
    val_fraction,
    dropout_p,
    lambd,
    mask_normalization,
    weight_decay,
    normalize,
    num_workers,
    mean,
    std
):
    return {
        "training_id": training_id,
        "training_dir": str(training_dir),
        "started_at": started_at,
        "requested_model_name": requested_model_name,
        "instantiated_model_name": instantiated_model_name,
        "instantiated_model_class": instantiated_model_class,
        "model_name": instantiated_model_name,
        "hyperparameters": {
            "lr": lr,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "early_stopping_patience": early_stopping_patience,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "dropout_p": dropout_p,
            "lambd": lambd,
            "mask_normalization": mask_normalization,
            "weight_decay": weight_decay,
            "normalize": normalize,
            "num_workers": num_workers,
        },
        "dataset_stats": {
            "mean": mean,
            "std": std
        }
    }


def save_run_metadata(training_dir, metadata):
    training_dir = Path(training_dir)
    path = training_dir / "run_metadata.json"
    save_json(metadata, path)
    print(f"🧾 Metadata general guardada en: {path}")


def generate_gridsearch_id():
    started_at = datetime.datetime.now()
    started_at_str = started_at.strftime("%Y-%m-%d-%H-%M-%S-%f")
    raw = f"{started_at_str}-gridsearch"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    gridsearch_id = f"{started_at_str}-{short_hash}"
    return gridsearch_id, started_at_str


def get_gridsearch_root_dir(save_dir):
    root = Path(save_dir) / "gridsearch"
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_gridsearch_dir(save_dir, gridsearch_id):
    gridsearch_root = get_gridsearch_root_dir(save_dir)
    gridsearch_dir = gridsearch_root / gridsearch_id
    gridsearch_dir.mkdir(parents=True, exist_ok=True)
    return gridsearch_dir


def make_model_gridsearch_dir(gridsearch_dir, model_name):
    model_dir = Path(gridsearch_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def expand_grid_config(param_grid):
    from itertools import product

    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values_product = product(*(param_grid[k] for k in keys))

    combinations = []
    for values in values_product:
        combinations.append(dict(zip(keys, values)))

    return combinations


def save_gridsearch_metadata(gridsearch_dir, metadata):
    gridsearch_dir = Path(gridsearch_dir)
    path = gridsearch_dir / "gridsearch_metadata.json"
    save_json(metadata, path)
    print(f"🧾 Metadata general del grid search guardada en: {path}")


def save_model_gridsearch_metadata(model_gridsearch_dir, metadata):
    model_gridsearch_dir = Path(model_gridsearch_dir)
    path = model_gridsearch_dir / "model_gridsearch_metadata.json"
    save_json(metadata, path)
    print(f"🧾 Metadata del grid search de modelo guardada en: {path}")


# =========================
# MODELS
# =========================
class OverfitNet(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.name = "OverfitNet"
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DropoutNetV2(nn.Module):
    def __init__(self, hidden_size=256, dropout_p=0.5):
        super().__init__()
        self.name = "DropoutNetV2"
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class BoostDropout(nn.Module):
    """
    En lugar de apagar activaciones como dropout clásico, algunas se amplifican
    por un factor (1 + lambda) con probabilidad (1 - p). Las demás quedan igual.
    """
    def __init__(self, p=0.8, lambd=0.2, mask_normalization=False):
        super().__init__()
        assert 0 <= p <= 1, "p debe estar entre 0 y 1"
        self.p = p
        self.lambd = lambd
        self.mask_normalization = mask_normalization

    def forward(self, x):
        if not self.training or self.p == 1:
            return x

        mask = torch.ones_like(x)
        boost_positions = torch.rand_like(x) > self.p
        mask[boost_positions] = 1 + self.lambd

        if self.mask_normalization:
            alpha = 1 + (1 - self.p) * self.lambd
            return x * (mask / alpha)
        else:
            return x * mask


class BoostDropoutNetV2(nn.Module):
    def __init__(self, hidden_size=256, dropout_p=0.5, lambd=0.5, mask_normalization=False):
        super().__init__()
        self.name = "BoostDropoutNetV2"
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

        self.boost_dropout = BoostDropout(
            p=dropout_p,
            lambd=lambd,
            mask_normalization=mask_normalization
        )

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.boost_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.boost_dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# =========================
# DATA
# =========================
def get_mnist_transform(normalize=True, mean=0.1307, std=0.3081):
    transform_list = [transforms.ToTensor()]

    if normalize:
        transform_list.append(transforms.Normalize((mean,), (std,)))

    return transforms.Compose(transform_list)


def count_classes_from_indices(targets, indices, num_classes=10):
    labels = targets[indices]
    return np.bincount(labels, minlength=num_classes)


def plot_class_distribution(train_counts, val_counts, title_suffix=""):
    plt.figure(figsize=(8, 4))
    x = np.arange(len(train_counts))
    plt.bar(x - 0.15, train_counts, width=0.3, label="Train")
    plt.bar(x + 0.15, val_counts, width=0.3, label="Val")
    plt.xticks(x)
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de muestras")
    plt.title(f"Distribución de clases {title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def load_small_data(
    train_fraction=0.10,
    val_fraction=0.25,
    normalize=True,
    mean=0.1307,
    std=0.3081,
    cache_prefix="mnist_stratified",
    cache_dir=None,
):
    """
    Devuelve un subconjunto estratificado y determinista de MNIST.
    Guarda los índices en disco para reutilizar exactamente el mismo split.
    """
    assert 0 < train_fraction <= 1, "train_fraction debe estar en (0, 1]"
    assert 0 < val_fraction < 1, "val_fraction debe estar en (0, 1)"

    if cache_dir is None:
        cache_dir = SAVE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    transform = get_mnist_transform(normalize=normalize, mean=mean, std=std)
    full_train = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transform)

    cache_name = (
        f"{cache_prefix}_frac-{train_fraction}_val-{val_fraction}"
        f"_norm-{normalize}_mean-{mean}_std-{std}_seed-{SEED}.npz"
    )
    cache_path = cache_dir / cache_name

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        print(f"📦 Split cargado desde cache: {cache_name}")
    else:
        rs = np.random.RandomState(SEED)

        targets = np.array(full_train.targets)
        all_indices = np.arange(len(full_train))

        train_idx_list = []
        val_idx_list = []

        for cls in range(10):
            cls_indices = all_indices[targets == cls].copy()
            rs.shuffle(cls_indices)

            n_selected_cls = int(round(len(cls_indices) * train_fraction))
            cls_selected = cls_indices[:n_selected_cls]

            n_val_cls = int(round(len(cls_selected) * val_fraction))
            n_train_cls = len(cls_selected) - n_val_cls

            cls_train_idx = cls_selected[:n_train_cls]
            cls_val_idx = cls_selected[n_train_cls:]

            train_idx_list.extend(cls_train_idx.tolist())
            val_idx_list.extend(cls_val_idx.tolist())

        train_idx = np.array(train_idx_list, dtype=np.int64)
        val_idx = np.array(val_idx_list, dtype=np.int64)

        rs.shuffle(train_idx)
        rs.shuffle(val_idx)

        np.savez_compressed(cache_path, train_idx=train_idx, val_idx=val_idx)
        print(f"💾 Split guardado en cache: {cache_name}")

    train_ds = Subset(full_train, train_idx.tolist())
    val_ds = Subset(full_train, val_idx.tolist())

    targets_np = np.array(full_train.targets)
    train_counts = count_classes_from_indices(targets_np, train_idx, num_classes=10)
    val_counts = count_classes_from_indices(targets_np, val_idx, num_classes=10)

    print("\n📊 Distribución de clases (Train vs Val) - estratificada, fija y cacheada")
    for i in range(10):
        print(f"Clase {i}: Train={train_counts[i]}  Val={val_counts[i]}")

    title_suffix = f"(frac={train_fraction}, val={val_fraction}, normalize={normalize})"
    plot_class_distribution(train_counts, val_counts, title_suffix=title_suffix)

    return train_ds, val_ds


def compute_dataset_mean_std(dataset, batch_size=512, num_workers=2):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    channel_sum = 0.0
    channel_squared_sum = 0.0
    num_pixels = 0

    for images, _ in loader:
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
        num_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_squared_sum / num_pixels - mean ** 2)

    return mean.item(), std.item()


def make_dataloaders(train_ds, val_ds, batch_size=64, num_workers=2):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    pin_memory = device.type == "cuda"

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator
    )

    return train_dl, val_dl


def train_model_on_mnist_subset(
    model_name="OverfitNet",
    lr=1e-3,
    hidden_size=256,
    batch_size=128,
    num_epochs=100,
    early_stopping_patience=10,
    train_fraction=0.10,
    val_fraction=0.25,
    dropout_p=0.5,
    lambd=0.5,
    mask_normalization=False,
    weight_decay=0.0,
    normalize=True,
    num_workers=2,
    plot_curves=True,
    save_final_csv=True,
    save_training_plots=True,
    base_save_dir=None
):
    print("\n" + "=" * 80)
    print(f"🚀 Entrenando modelo: {model_name}")
    print("=" * 80)
    print(f"lr={lr}")
    print(f"hidden_size={hidden_size}")
    print(f"batch_size={batch_size}")
    print(f"num_epochs={num_epochs}")
    print(f"early_stopping_patience={early_stopping_patience}")
    print(f"train_fraction={train_fraction}")
    print(f"val_fraction={val_fraction}")
    print(f"dropout_p={dropout_p}")
    print(f"lambd={lambd}")
    print(f"mask_normalization={mask_normalization}")
    print(f"weight_decay={weight_decay}")
    print(f"normalize={normalize}")
    print(f"num_workers={num_workers}")

    training_id, started_at = generate_training_id(model_name)
    effective_save_dir = SAVE_DIR if base_save_dir is None else Path(base_save_dir)
    training_dir = make_training_dir(effective_save_dir, training_id)

    print(f"🆔 training_id: {training_id}")
    print(f"📁 training_dir: {training_dir}")

    train_ds_raw, val_ds_raw = load_small_data(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        normalize=False,
        cache_dir=effective_save_dir,
    )

    if normalize:
        mean, std = compute_dataset_mean_std(
            train_ds_raw,
            batch_size=batch_size,
            num_workers=num_workers
        )
        print(f"📊 mean del subset train: {mean}")
        print(f"📊 std del subset train: {std}")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
    else:
        mean, std = None, None
        transform = transforms.ToTensor()

    full_train = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transform)

    train_ds = Subset(full_train, train_ds_raw.indices)
    val_ds = Subset(full_train, val_ds_raw.indices)

    train_dl, val_dl = make_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=batch_size,
        num_workers=num_workers
    )

    model = build_model(
        model_name=model_name,
        hidden_size=hidden_size,
        dropout_p=dropout_p,
        lambd=lambd,
        mask_normalization=mask_normalization
    ).to(device)

    requested_model_name = model_name
    instantiated_model_name = model.name
    instantiated_model_class = model.__class__.__name__

    requested_model_name_normalized = requested_model_name.lower()
    instantiated_model_name_normalized = instantiated_model_name.lower()

    if requested_model_name_normalized != instantiated_model_name_normalized:
        raise RuntimeError(
            f"Inconsistencia de modelo: se pidió '{requested_model_name}' "
            f"pero se construyó '{instantiated_model_name}'."
        )

    optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = init_history()

    history["mean"] = mean
    history["std"] = std
    history["training_id"] = training_id

    run_metadata = build_run_metadata(
        training_id=training_id,
        training_dir=training_dir,
        started_at=started_at,
        requested_model_name=requested_model_name,
        instantiated_model_name=instantiated_model_name,
        instantiated_model_class=instantiated_model_class,
        lr=lr,
        hidden_size=hidden_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        dropout_p=dropout_p,
        lambd=lambd,
        mask_normalization=mask_normalization,
        weight_decay=weight_decay,
        normalize=normalize,
        num_workers=num_workers,
        mean=mean,
        std=std
    )

    save_run_metadata(training_dir, run_metadata)

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    early_stopping_saved = False

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, loss_fn, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_dl, loss_fn, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{epoch}/{num_epochs}] "
            f"train_acc={tr_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"train_loss={tr_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        save_model_artifact(
            model=model,
            history=history,
            training_dir=training_dir,
            artifact_type="last_model",
            epoch=epoch,
            val_loss=val_loss,
            training_id=training_id,
            run_metadata=run_metadata,
            extra_info={
                "best_val_loss_so_far": best_val_loss if best_val_loss != float("inf") else None
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            save_model_artifact(
                model=model,
                history=history,
                training_dir=training_dir,
                artifact_type="best_model",
                epoch=epoch,
                val_loss=val_loss,
                training_id=training_id,
                run_metadata=run_metadata,
                extra_info={
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss
                }
            )
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and not early_stopping_saved
            and epochs_without_improvement >= early_stopping_patience
        ):
            best_model_path = Path(training_dir) / "best_model.pth"
            best_model_meta_path = Path(training_dir) / "best_model.json"

            early_model_path = Path(training_dir) / "early_stopping.pth"
            early_model_meta_path = Path(training_dir) / "early_stopping.json"

            shutil.copyfile(best_model_path, early_model_path)
            shutil.copyfile(best_model_meta_path, early_model_meta_path)

            with open(early_model_meta_path, "r", encoding="utf-8") as f:
                early_meta = json.load(f)

            early_meta["artifact_type"] = "early_stopping"
            early_meta["extra_info"] = {
                "patience": early_stopping_patience,
                "best_epoch_before_stop": best_epoch,
                "best_val_loss_before_stop": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "saved_from": "best_model"
            }

            save_json(early_meta, early_model_meta_path)

            early_stopping_saved = True
            print(f"⏹️ Early stopping cumplido en epoch {epoch}.")
            print(f"💾 early_stopping guardado en: {early_model_path}")

            # Si quieres cortar el entrenamiento al cumplirse:
            # break

    if not early_stopping_saved:
        print("ℹ️ No se cumplió el criterio de early stopping durante este entrenamiento.")

    if plot_curves:
        plot_history(history, model_name=model.name)

    if save_training_plots:
        save_training_plot(history, training_dir=training_dir, metric="loss")
        save_training_plot(history, training_dir=training_dir, metric="accuracy")

    if save_final_csv:
        save_history_csv(history, training_dir=training_dir)

    run_metadata["ended_at"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    run_metadata["results"] = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "early_stopping_saved": early_stopping_saved,
        "finished_epoch": len(history["train_loss"])
    }
    run_metadata["generated_files"] = [
        "best_model.pth",
        "best_model.json",
        "last_model.pth",
        "last_model.json",
        "history.csv",
        "plot_loss.png",
        "plot_accuracy.png",
        "run_metadata.json"
    ]
    if early_stopping_saved:
        run_metadata["generated_files"].extend([
            "early_stopping.pth",
            "early_stopping.json"
        ])

    save_run_metadata(training_dir, run_metadata)

    print("✅ Entrenamiento finalizado.")
    return history, model


# =========================
# GRID SEARCH UTILS
# =========================
def get_supported_model_params(model_name):
    common_params = {
        "model_name",
        "lr",
        "hidden_size",
        "batch_size",
        "num_epochs",
        "early_stopping_patience",
        "train_fraction",
        "val_fraction",
        "weight_decay",
        "normalize",
        "num_workers",
        "plot_curves",
        "save_final_csv",
        "save_training_plots",
    }

    model_name_lower = model_name.lower()

    if model_name_lower == "overfitnet":
        return common_params

    if model_name_lower == "dropoutnetv2":
        return common_params | {"dropout_p"}

    if model_name_lower == "boostdropoutnetv2":
        return common_params | {"dropout_p", "lambd", "mask_normalization"}

    raise ValueError(
        "model_name debe ser uno de: "
        "'OverfitNet', 'DropoutNetV2', 'BoostDropoutNetV2'"
    )


def filter_params_for_model(model_name, params):
    supported = get_supported_model_params(model_name)
    return {k: v for k, v in params.items() if k in supported}


def run_gridsearch(
    model_param_grids,
    common_params=None,
    save_dir=None
):
    if save_dir is None:
        save_dir = SAVE_DIR

    if common_params is None:
        common_params = {}

    gridsearch_id, gridsearch_started_at = generate_gridsearch_id()
    gridsearch_dir = make_gridsearch_dir(save_dir, gridsearch_id)

    print("\n" + "=" * 100)
    print("🔎 INICIANDO GRID SEARCH")
    print("=" * 100)
    print(f"🆔 gridsearch_id: {gridsearch_id}")
    print(f"📁 gridsearch_dir: {gridsearch_dir}")

    gridsearch_metadata = {
        "gridsearch_id": gridsearch_id,
        "gridsearch_dir": str(gridsearch_dir),
        "started_at": gridsearch_started_at,
        "type": "gridsearch",
        "common_params": common_params,
        "models": {},
        "generated_files": [
            "gridsearch_metadata.json"
        ]
    }

    save_gridsearch_metadata(gridsearch_dir, gridsearch_metadata)

    global_run_counter = 0

    for model_name, param_grid in model_param_grids.items():
        print("\n" + "-" * 100)
        print(f"📦 Grid search para modelo: {model_name}")
        print("-" * 100)

        model_gridsearch_dir = make_model_gridsearch_dir(gridsearch_dir, model_name)
        combinations = expand_grid_config(param_grid)

        model_grid_metadata = {
            "gridsearch_id": gridsearch_id,
            "model_name": model_name,
            "model_gridsearch_dir": str(model_gridsearch_dir),
            "started_at": datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
            "num_combinations": len(combinations),
            "param_grid": param_grid,
            "runs": [],
            "generated_files": [
                "model_gridsearch_metadata.json"
            ]
        }

        save_model_gridsearch_metadata(model_gridsearch_dir, model_grid_metadata)

        best_model_run_summary = None
        best_model_val_loss = float("inf")

        for combo_idx, combo_params in enumerate(combinations, start=1):
            global_run_counter += 1

            run_params = {}
            run_params.update(common_params)
            run_params.update(combo_params)
            run_params = filter_params_for_model(model_name, run_params)
            run_params["model_name"] = model_name

            print("\n" + "~" * 100)
            print(f"▶️ Run {combo_idx}/{len(combinations)} para {model_name}")
            print(f"🌐 Run global {global_run_counter}")
            print(f"⚙️ Params: {run_params}")
            print("~" * 100)

            history, model = train_model_on_mnist_subset(
                **run_params,
                base_save_dir=model_gridsearch_dir
            )

            training_id = history["training_id"]
            training_dir = Path(model_gridsearch_dir) / training_id
            run_metadata_path = training_dir / "run_metadata.json"

            with open(run_metadata_path, "r", encoding="utf-8") as f:
                run_metadata = json.load(f)

            requested_model_name = run_metadata.get("requested_model_name")
            instantiated_model_name = run_metadata.get("instantiated_model_name")
            instantiated_model_class = run_metadata.get("instantiated_model_class")

            if instantiated_model_name is None:
                instantiated_model_name = run_metadata.get("model_name")

            if instantiated_model_name is None:
                raise RuntimeError(
                    f"No se pudo determinar el modelo entrenado en {training_dir}"
                )

            if requested_model_name is not None and requested_model_name.lower() != model_name.lower():
                raise RuntimeError(
                    f"Grid search inconsistente en {training_dir}: "
                    f"el run fue solicitado como '{requested_model_name}', "
                    f"pero estaba bajo el modelo '{model_name}'."
                )

            run_summary = {
                "training_id": training_id,
                "training_dir": str(training_dir),
                "requested_model_name": requested_model_name,
                "instantiated_model_name": instantiated_model_name,
                "instantiated_model_class": instantiated_model_class,
                "params": run_params,
                "best_epoch": run_metadata.get("results", {}).get("best_epoch"),
                "best_val_loss": run_metadata.get("results", {}).get("best_val_loss"),
                "early_stopping_saved": run_metadata.get("results", {}).get("early_stopping_saved"),
                "generated_files": run_metadata.get("generated_files", [])
            }

            model_grid_metadata["runs"].append(run_summary)

            current_best_val_loss = run_summary["best_val_loss"]
            if current_best_val_loss is not None and current_best_val_loss < best_model_val_loss:
                best_model_val_loss = current_best_val_loss
                best_model_run_summary = run_summary

            save_model_gridsearch_metadata(model_gridsearch_dir, model_grid_metadata)

        model_grid_metadata["ended_at"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        model_grid_metadata["best_run"] = best_model_run_summary
        model_grid_metadata["best_val_loss"] = (
            None if best_model_val_loss == float("inf") else best_model_val_loss
        )
        model_grid_metadata["runs_summary_table"] = [
            {
                "training_id": run["training_id"],
                "best_val_loss": run["best_val_loss"],
                "best_epoch": run["best_epoch"]
            }
            for run in sorted(
                model_grid_metadata["runs"],
                key=lambda x: float("inf") if x["best_val_loss"] is None else x["best_val_loss"]
            )
        ]

        save_model_gridsearch_metadata(model_gridsearch_dir, model_grid_metadata)

        gridsearch_metadata["models"][model_name] = {
            "model_gridsearch_dir": str(model_gridsearch_dir),
            "num_combinations": len(combinations),
            "best_run": best_model_run_summary,
            "best_val_loss": None if best_model_val_loss == float("inf") else best_model_val_loss
        }
        gridsearch_metadata["generated_files"].append(
            f"{model_name}/model_gridsearch_metadata.json"
        )

        save_gridsearch_metadata(gridsearch_dir, gridsearch_metadata)

    gridsearch_metadata["ended_at"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    best_global_model_name = None
    best_global_run = None
    best_global_val_loss = float("inf")

    for model_name, model_info in gridsearch_metadata["models"].items():
        model_best = model_info.get("best_val_loss")
        if model_best is not None and model_best < best_global_val_loss:
            best_global_val_loss = model_best
            best_global_model_name = model_name
            best_global_run = model_info.get("best_run")

    gridsearch_metadata["best_global_result"] = {
        "model_name": best_global_model_name,
        "best_run": best_global_run,
        "best_val_loss": None if best_global_val_loss == float("inf") else best_global_val_loss
    }

    save_gridsearch_metadata(gridsearch_dir, gridsearch_metadata)

    print("\n" + "=" * 100)
    print("✅ GRID SEARCH FINALIZADO")
    print("=" * 100)
    print(f"📁 Resultados en: {gridsearch_dir}")

    return gridsearch_metadata


# =========================
# WRAPPERS LISTOS PARA LOS 3 MODELOS
# =========================
def train_overfitnet(
    lr=1e-3,
    hidden_size=256,
    batch_size=128,
    num_epochs=100,
    early_stopping_patience=10,
    train_fraction=0.10,
    val_fraction=0.25,
    weight_decay=0.0,
    normalize=True,
    num_workers=2,
    plot_curves=True,
    save_final_csv=True,
    save_training_plots=True
):
    return train_model_on_mnist_subset(
        model_name="OverfitNet",
        lr=lr,
        hidden_size=hidden_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        weight_decay=weight_decay,
        normalize=normalize,
        num_workers=num_workers,
        plot_curves=plot_curves,
        save_final_csv=save_final_csv,
        save_training_plots=save_training_plots
    )


def train_dropoutnetv2(
    lr=1e-3,
    hidden_size=256,
    batch_size=128,
    num_epochs=100,
    early_stopping_patience=10,
    train_fraction=0.10,
    val_fraction=0.25,
    dropout_p=0.5,
    weight_decay=0.0,
    normalize=True,
    num_workers=2,
    plot_curves=True,
    save_final_csv=True,
    save_training_plots=True
):
    return train_model_on_mnist_subset(
        model_name="DropoutNetV2",
        lr=lr,
        hidden_size=hidden_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        dropout_p=dropout_p,
        weight_decay=weight_decay,
        normalize=normalize,
        num_workers=num_workers,
        plot_curves=plot_curves,
        save_final_csv=save_final_csv,
        save_training_plots=save_training_plots
    )


def train_boostdropoutnetv2(
    lr=1e-3,
    hidden_size=256,
    batch_size=128,
    num_epochs=100,
    early_stopping_patience=10,
    train_fraction=0.10,
    val_fraction=0.25,
    dropout_p=0.5,
    lambd=0.5,
    mask_normalization=False,
    weight_decay=0.0,
    normalize=True,
    num_workers=2,
    plot_curves=True,
    save_final_csv=True,
    save_training_plots=True
):
    return train_model_on_mnist_subset(
        model_name="BoostDropoutNetV2",
        lr=lr,
        hidden_size=hidden_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        dropout_p=dropout_p,
        lambd=lambd,
        mask_normalization=mask_normalization,
        weight_decay=weight_decay,
        normalize=normalize,
        num_workers=num_workers,
        plot_curves=plot_curves,
        save_final_csv=save_final_csv,
        save_training_plots=save_training_plots
    )


# =========================
# AUDIO ALERT
# =========================
def play_training_finished_audio(enable_audio=True, filename="voz_final.mp3"):
    """
    En local funciona si:
    - tienes gTTS instalado
    - tienes conexión a Internet (gTTS usa un servicio remoto)
    Si falla, se degrada con un mensaje de texto.
    """
    if not enable_audio:
        print("🔕 Audio deshabilitado.")
        return

    if not GTTS_AVAILABLE:
        print("⚠️ gTTS no está disponible. Omitiendo audio.")
        return

    texto = (
        "Entrenamiento finalizado correctamente maestro. "
        "Vení a mirar los resultados antes de que se cierre el entorno de ejecución. "
        "Los modelos han completado todas las épocas sin interrupciones críticas papá. "
        "Las métricas de validación se encuentran disponibles en la carpeta de resultados. "
        "Por favor máquina, revisá las curvas de pérdida y exactitud para verificar si el modelo está sobreajustando. "
        "Recordá que el tiempo de sesión de Google Colab es limitado. Venii!! "
        "Te recomiendo guardar los pesos entrenados y exportar los gráficos cuanto antes monstro. "
        "Este mensaje se autodestruirá en tres, dos, uno... mentira, pero apurate igual. "
        "Fin de la transmisión."
    )

    audio_path = Path(SAVE_DIR) / filename
    try:
        tts = gTTS(texto, lang="es")
        tts.save(str(audio_path))
        display(Audio(str(audio_path), autoplay=True))
        print(f"🔊 Audio guardado en: {audio_path}")
    except Exception as e:
        print(f"⚠️ No se pudo generar/reproducir el audio: {e}")


# =========================
# EXAMPLES / RUN
# =========================
# Ejecuta solo los bloques que quieras. No es necesario correr todo junto.

# -------- Ejemplo 1: OverfitNet --------
# history_overfit, model_overfit = train_overfitnet(
#     lr=1e-3,
#     hidden_size=256,
#     batch_size=128,
#     num_epochs=30,
#     early_stopping_patience=10,
#     train_fraction=0.10,
#     val_fraction=0.25,
#     weight_decay=0.0,
#     normalize=False,
#     num_workers=2,
#     plot_curves=True,
#     save_final_csv=True
# )
# play_training_finished_audio()

# -------- Ejemplo 2: DropoutNetV2 --------
# history_dropout, model_dropout = train_dropoutnetv2(
#     lr=1e-3,
#     hidden_size=256,
#     batch_size=128,
#     num_epochs=30,
#     early_stopping_patience=5,
#     train_fraction=0.1,
#     val_fraction=0.25,
#     dropout_p=0.5,
#     weight_decay=0.0,
#     normalize=False,
#     num_workers=2,
#     plot_curves=True,
#     save_final_csv=True
# )
# play_training_finished_audio()

# -------- Ejemplo 3: BoostDropoutNetV2 --------
# history_boost, model_boost = train_boostdropoutnetv2(
#     lr=1e-3,
#     hidden_size=256,
#     batch_size=128,
#     num_epochs=30,
#     early_stopping_patience=5,
#     train_fraction=0.1,
#     val_fraction=0.25,
#     dropout_p=0.5,
#     lambd=0.6,
#     mask_normalization=False,
#     weight_decay=0.0,
#     normalize=False,
#     num_workers=2,
#     plot_curves=True,
#     save_final_csv=True
# )
# play_training_finished_audio()

# -------- Grid Search --------
# gridsearch_results = run_gridsearch(
#     model_param_grids={
#         "OverfitNet": {
#             "lr": [1e-3],
#             "hidden_size": [256],
#             "batch_size": [128]
#         },
#         "DropoutNetV2": {
#             "lr": [1e-3],
#             "hidden_size": [256],
#             "batch_size": [128],
#             "dropout_p": [0.3]
#         },
#         "BoostDropoutNetV2": {
#             "lr": [1e-3],
#             "hidden_size": [256],
#             "batch_size": [128],
#             "dropout_p": [0.5],
#             "lambd": [0.6],
#             "mask_normalization": [False]
#         }
#     },
#     common_params={
#         "num_epochs": 35,
#         "early_stopping_patience": 7,
#         "train_fraction": 0.10,
#         "val_fraction": 0.25,
#         "weight_decay": 0.0,
#         "normalize": False,
#         "num_workers": 2,
#         "plot_curves": False,
#         "save_final_csv": True,
#         "save_training_plots": True
#     }
# )
# play_training_finished_audio()
