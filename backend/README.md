# Backend

FastAPI service for handwritten digit prediction and temporary retraining.

## Run

Install the dependencies from [backend/requirements.txt](backend/requirements.txt), then start the app from the project root:

```bash
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/` to use the canvas UI.

## API

- `GET /api/models` lists the original model plus any temporary retrained models.
- `POST /api/predict` accepts a single image file and a `model_id`.
- `POST /api/retrain` starts server-side fine-tuning from a selected model without overwriting the original checkpoint.
- `GET /api/jobs/{job_id}` reports retraining progress.

## Storage

Saved checkpoints and metadata live under `backend/storage/` and are ignored by git.
