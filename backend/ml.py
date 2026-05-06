from __future__ import annotations

import io
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR.parent
AI_DIR = PROJECT_DIR / "AI"
STORAGE_DIR = ROOT_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
ORIGINAL_MODEL_DIR = MODELS_DIR / "original"
CUSTOM_MODEL_DIR = MODELS_DIR / "custom"
STATIC_DIR = ROOT_DIR / "static"
REGISTRY_PATH = STORAGE_DIR / "model_registry.json"

MNIST_DATA_DIR = AI_DIR / "data" / "mnist"
NORMALIZE_MEAN = (0.1307,)
NORMALIZE_STD = (0.3081,)


def ensure_storage_layout() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    ORIGINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_training_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def generate_model_id(prefix: str = "temp") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}-{stamp}-{suffix}"


def generate_job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    weight_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_t = 127

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            best_t = t

    return best_t


def _foreground_mask(gray: np.ndarray) -> Tuple[np.ndarray, str]:
    threshold = _otsu_threshold(gray)
    mask_dark = gray < threshold
    mask_light = gray > threshold

    dark_ratio = float(mask_dark.mean())
    light_ratio = float(mask_light.mean())

    def score(ratio: float) -> float:
        if ratio < 0.005 or ratio > 0.8:
            return 10.0
        return abs(ratio - 0.15)

    if score(dark_ratio) <= score(light_ratio):
        return mask_dark, "dark"
    return mask_light, "light"


def _center_digit(gray: np.ndarray, mask: np.ndarray, polarity: str) -> Image.Image:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        canvas = np.zeros((28, 28), dtype=np.uint8)
        return Image.fromarray(canvas, mode="L")

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    crop = gray[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1].astype(np.uint8)

    if polarity == "dark":
        foreground = (255 - crop) * crop_mask
    else:
        foreground = crop * crop_mask

    fg_ys, fg_xs = np.where(foreground > 0)
    if fg_xs.size == 0 or fg_ys.size == 0:
        foreground = crop_mask * 255
        fg_ys, fg_xs = np.where(foreground > 0)

    fg_y0, fg_y1 = int(fg_ys.min()), int(fg_ys.max()) + 1
    fg_x0, fg_x1 = int(fg_xs.min()), int(fg_xs.max()) + 1
    foreground = foreground[fg_y0:fg_y1, fg_x0:fg_x1]

    height, width = foreground.shape
    size = max(height, width) + 20
    canvas = np.zeros((size, size), dtype=np.uint8)
    top = (size - height) // 2
    left = (size - width) // 2
    canvas[top : top + height, left : left + width] = foreground

    image = Image.fromarray(canvas, mode="L")
    return image.resize((28, 28), Image.Resampling.BICUBIC)


def preprocess_digit_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    gray_image = background.convert("L")
    gray = np.array(gray_image)
    mask, polarity = _foreground_mask(gray)
    digit = _center_digit(gray, mask, polarity)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    return transform(digit)


def build_loaders(
    data_dir: Path,
    batch_size: int,
    test_batch_size: int,
    download: bool,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )

    train_dataset = datasets.MNIST(
        str(data_dir), train=True, download=download, transform=transform
    )
    test_dataset = datasets.MNIST(
        str(data_dir), train=False, download=download, transform=transform
    )

    use_cuda = bool(device is not None and device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0,
    )
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, 100.0 * correct / total


def save_metrics(output_dir: Path, history: Dict[str, List[float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def train_model(
    base_model_path: Optional[Path],
    output_path: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    test_batch_size: int,
    download: bool,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    active_device = device or resolve_training_device(prefer_cuda=True)
    train_loader, test_loader = build_loaders(
        MNIST_DATA_DIR,
        batch_size,
        test_batch_size,
        download,
        device=active_device,
    )

    model = MNISTCNN().to(active_device)
    if base_model_path is not None and base_model_path.exists():
        state = torch.load(base_model_path, map_location=active_device)
        model.load_state_dict(state)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, active_device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate_model(model, active_device, test_loader, criterion)

        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["test_loss"].append(float(test_loss))
        history["test_acc"].append(float(test_acc))

        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    save_metrics(output_path.parent, history)

    return {
        "history": history,
        "final_test_acc": history["test_acc"][-1],
        "final_test_loss": history["test_loss"][-1],
        "model_path": str(output_path),
    }


class ModelRegistry:
    def __init__(self, registry_path: Path = REGISTRY_PATH) -> None:
        self.registry_path = registry_path
        self._lock = threading.RLock()
        self._models: List[Dict[str, Any]] = []
        self._load()

    def _original_record(self) -> Dict[str, Any]:
        original_path = ORIGINAL_MODEL_DIR / "mnist_cnn.pt"
        return {
            "id": "original",
            "name": "Original model",
            "kind": "original",
            "base_model_id": None,
            "created_at": "2026-05-06T00:00:00Z",
            "status": "ready" if original_path.exists() else "missing",
            "available": original_path.exists(),
            "path": str(original_path),
            "metrics": {},
            "description": "Immutable base checkpoint.",
        }

    def _load(self) -> None:
        ensure_storage_layout()
        if self.registry_path.exists():
            try:
                payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
                self._models = list(payload.get("models", []))
            except json.JSONDecodeError:
                self._models = []
        if not any(model.get("id") == "original" for model in self._models):
            self._models.insert(0, self._original_record())
        else:
            for index, model in enumerate(self._models):
                if model.get("id") == "original":
                    self._models[index] = self._original_record()
                    break
        self.save()

    def save(self) -> None:
        with self._lock:
            payload = {"models": self._models}
            temp_path = self.registry_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(self.registry_path)

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        for model in self._models:
            item = dict(model)
            path = Path(item["path"])
            item["available"] = path.exists()
            models.append(item)
        return sorted(models, key=lambda model: model.get("created_at", ""), reverse=True)

    def get_model(self, model_id: str) -> Dict[str, Any]:
        for model in self._models:
            if model.get("id") == model_id:
                item = dict(model)
                item["available"] = Path(item["path"]).exists()
                return item
        raise KeyError(model_id)

    def register_model(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._models = [model for model in self._models if model.get("id") != record.get("id")]
            self._models.append(record)
            self.save()
            return dict(record)

    def update_model(self, model_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            for index, model in enumerate(self._models):
                if model.get("id") == model_id:
                    self._models[index] = {**model, **updates}
                    self.save()
                    return dict(self._models[index])
        raise KeyError(model_id)


def predict_with_model(model_path: Path, image_bytes: bytes, device: Optional[torch.device] = None) -> Dict[str, Any]:
    active_device = device or default_device()
    model = MNISTCNN().to(active_device)
    state = torch.load(model_path, map_location=active_device)
    model.load_state_dict(state)
    model.eval()

    tensor = preprocess_digit_image(image_bytes).unsqueeze(0).to(active_device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

    ranked = sorted(
        [{"digit": index, "probability": float(prob)} for index, prob in enumerate(probabilities)],
        key=lambda item: item["probability"],
        reverse=True,
    )

    top = ranked[0]
    return {
        "prediction": int(top["digit"]),
        "confidence": float(top["probability"]),
        "top_k": ranked[:3],
    }


def create_model_record(
    model_id: str,
    name: str,
    kind: str,
    base_model_id: Optional[str],
    path: Path,
    status: str = "training",
    metrics: Optional[Dict[str, Any]] = None,
    description: str = "",
) -> Dict[str, Any]:
    return {
        "id": model_id,
        "name": name,
        "kind": kind,
        "base_model_id": base_model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "available": path.exists(),
        "path": str(path),
        "metrics": metrics or {},
        "description": description,
    }
