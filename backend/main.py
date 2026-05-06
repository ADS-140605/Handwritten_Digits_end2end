from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import torch

from .ml import (
    CUSTOM_MODEL_DIR,
    STATIC_DIR,
    ModelRegistry,
    create_model_record,
    default_device,
    ensure_storage_layout,
    generate_job_id,
    generate_model_id,
    predict_with_model,
    resolve_training_device,
    train_model,
)

from pathlib import Path


ensure_storage_layout()

app = FastAPI(title="Handwritten Digit Studio", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

registry = ModelRegistry()
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


class RetrainRequest(BaseModel):
    base_model_id: str = Field(default="original")
    epochs: int = Field(default=2, ge=1, le=30)
    lr: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=64, ge=1, le=512)
    test_batch_size: int = Field(default=1000, ge=1, le=5000)
    download: bool = False
    prefer_cuda: bool = True
    require_cuda: bool = False
    name: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(default_device()),
        "models": len(registry.list_models()),
    }


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    return {"models": registry.list_models()}


@app.get("/api/models/{model_id}")
def get_model(model_id: str) -> Dict[str, Any]:
    try:
        return registry.get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}") from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.post("/api/predict")
async def predict_digit(file: UploadFile = File(...), model_id: str = Form("original")) -> Dict[str, Any]:
    try:
        model = registry.get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}") from exc

    model_path = Path(model["path"])
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Model checkpoint is missing for {model_id}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    prediction = predict_with_model(model_path, image_bytes)
    return {
        "model": {
            "id": model["id"],
            "name": model["name"],
            "kind": model["kind"],
        },
        **prediction,
    }


@app.post("/api/retrain")
def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    try:
        registry.get_model(request.base_model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown base model: {request.base_model_id}") from exc

    model_id = generate_model_id()
    job_id = generate_job_id()
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "model_id": model_id,
            "base_model_id": request.base_model_id,
            "progress": {"epoch": 0, "epochs": request.epochs},
        }

    def run_job() -> None:
        base_model = registry.get_model(request.base_model_id)
        base_path = Path(base_model["path"])
        if not base_path.exists():
            base_path = None

        output_dir = CUSTOM_MODEL_DIR / model_id
        checkpoint_path = output_dir / "mnist_cnn.pt"

        if request.require_cuda and not torch.cuda.is_available():
            with jobs_lock:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "CUDA is not available in the current PyTorch install"
            return

        try:
            with jobs_lock:
                jobs[job_id]["status"] = "running"
            registry.register_model(
                create_model_record(
                    model_id=model_id,
                    name=request.name or f"Temporary model {model_id[-6:]}",
                    kind="temporary",
                    base_model_id=request.base_model_id,
                    path=checkpoint_path,
                    status="training",
                    description=f"Retrained from {request.base_model_id}.",
                )
            )

            def progress_callback(progress: Dict[str, Any]) -> None:
                with jobs_lock:
                    jobs[job_id]["progress"] = progress
                    jobs[job_id]["status"] = "running"

            training_device = resolve_training_device(prefer_cuda=request.prefer_cuda)
            if request.require_cuda and training_device.type != "cuda":
                raise RuntimeError("CUDA is not available in the current PyTorch install")

            result = train_model(
                base_model_path=base_path,
                output_path=checkpoint_path,
                epochs=request.epochs,
                lr=request.lr,
                batch_size=request.batch_size,
                test_batch_size=request.test_batch_size,
                download=request.download,
                device=training_device,
                progress_callback=progress_callback,
            )

            registry.update_model(
                model_id,
                {
                    "status": "ready",
                    "available": True,
                    "metrics": {
                        "final_test_accuracy": result["final_test_acc"],
                        "final_test_loss": result["final_test_loss"],
                    },
                    "path": str(checkpoint_path),
                },
            )
            with jobs_lock:
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["result"] = {"model_id": model_id, **result}
        except Exception as exc:
            registry.update_model(
                model_id,
                {
                    "status": "failed",
                    "available": checkpoint_path.exists(),
                    "error": str(exc),
                },
            )
            with jobs_lock:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(exc)

    background_tasks.add_task(run_job)
    return {
        "job_id": job_id,
        "model_id": model_id,
        "status": "queued",
        "base_model_id": request.base_model_id,
    }
