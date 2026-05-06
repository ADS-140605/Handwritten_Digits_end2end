from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from backend.ml import predict_with_model


# Custom CSS for better styling and mobile optimization
CUSTOM_CSS = """
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    .canvas-container { display: flex; justify-content: center; }
    canvas { cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><circle cx="8" cy="8" r="4" fill="black"/></svg>') 8 8, auto !important; }
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] { padding: 0.5rem; }
        h1 { font-size: 1.5rem; }
        h3 { font-size: 1rem; }
    }
</style>
"""

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
STORAGE_DIR = BACKEND_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
REGISTRY_PATH = STORAGE_DIR / "model_registry.json"


def _load_registry() -> List[Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    models = payload.get("models", [])
    if not isinstance(models, list):
        return []
    return [model for model in models if isinstance(model, dict)]


def _resolve_model_path(record: Dict[str, Any]) -> Optional[Path]:
    raw_path = record.get("path")
    if isinstance(raw_path, str):
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate
        candidate = ROOT_DIR / raw_path
        if candidate.exists():
            return candidate

    model_id = record.get("id")
    if model_id == "original":
        candidate = MODELS_DIR / "original" / "mnist_cnn.pt"
        if candidate.exists():
            return candidate
    if isinstance(model_id, str) and model_id.startswith("temp-"):
        candidate = MODELS_DIR / "custom" / model_id / "mnist_cnn.pt"
        if candidate.exists():
            return candidate
    return None


def _load_models() -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    for record in _load_registry():
        model_path = _resolve_model_path(record)
        if model_path is None:
            continue
        models.append(
            {
                "id": record.get("id", model_path.stem),
                "name": record.get("name", model_path.stem),
                "path": model_path,
                "available": True,
                "status": record.get("status", "ready"),
            }
        )

    if not models:
        fallback = MODELS_DIR / "original" / "mnist_cnn.pt"
        if fallback.exists():
            models.append(
                {
                    "id": "original",
                    "name": "Original model",
                    "path": fallback,
                    "available": True,
                    "status": "ready",
                }
            )
    return models


def _image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _canvas_to_bytes(canvas_data: Optional[np.ndarray]) -> Optional[bytes]:
    if canvas_data is None:
        return None
    if not np.any(canvas_data):
        return None
    canvas_image = Image.fromarray(canvas_data.astype(np.uint8), mode="RGBA")
    return _image_to_bytes(canvas_image)


st.set_page_config(page_title="Handwritten Digit Studio", page_icon="9", layout="wide")
st.title("Handwritten Digit Studio")
st.caption("Draw a digit, upload an image, and test the MNIST checkpoint directly in Streamlit.")

models = _load_models()
if not models:
    st.error("No trained model checkpoint was found. Put a model at backend/storage/models/original/mnist_cnn.pt.")
    st.stop()

with st.sidebar:
    st.header("Model")
    selected_model = st.selectbox(
        "Choose a model",
        options=models,
        format_func=lambda item: f"{item['name']} ({item['id']})",
    )
    st.write(f"Path: {selected_model['path']}")
    st.write(f"Status: {selected_model.get('status', 'ready')}")
    st.divider()
    st.write("The FastAPI backend keeps retraining and model management. This Streamlit page focuses on inference preview.")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Draw a digit")
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0

    clear_pressed = st.button("Clear canvas", use_container_width=True)
    if clear_pressed:
        st.session_state.canvas_key += 1

    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#ffffff",
        background_image=None,
        update_streamlit=True,
        height=420,
        width=420,
        drawing_mode="freedraw",
        key=f"canvas-{st.session_state.canvas_key}",
    )

with right:
    st.subheader("Upload an image")
    uploaded = st.file_uploader("PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"])
    st.markdown("### Prediction")

    candidate_bytes: Optional[bytes] = None
    if uploaded is not None:
        candidate_bytes = uploaded.getvalue()
    else:
        candidate_bytes = _canvas_to_bytes(getattr(canvas, "image_data", None))

    if candidate_bytes is None:
        st.info("Draw a digit or upload an image to get a prediction.")
    else:
        try:
            prediction = predict_with_model(Path(selected_model["path"]), candidate_bytes)
            st.metric("Predicted digit", prediction["prediction"])
            st.write(f"Confidence: {prediction['confidence'] * 100:.1f}%")
            for item in prediction["top_k"]:
                st.write(f"Digit {item['digit']}")
                st.progress(float(item["probability"]))
                st.caption(f"{item['probability'] * 100:.1f}%")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
