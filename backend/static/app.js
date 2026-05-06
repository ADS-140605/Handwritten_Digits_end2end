const canvas = document.getElementById("digitCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const modelSelect = document.getElementById("modelSelect");
const baseModelSelect = document.getElementById("baseModelSelect");
const healthPill = document.getElementById("healthPill");
const predictionValue = document.getElementById("predictionValue");
const predictionMeta = document.getElementById("predictionMeta");
const topKList = document.getElementById("topKList");
const jobState = document.getElementById("jobState");
const predictBtn = document.getElementById("predictBtn");
const retrainBtn = document.getElementById("retrainBtn");
const clearBtn = document.getElementById("clearBtn");
const fileInput = document.getElementById("fileInput");
const epochsInput = document.getElementById("epochsInput");
const lrInput = document.getElementById("lrInput");
const batchSizeInput = document.getElementById("batchSizeInput");
const downloadInput = document.getElementById("downloadInput");

const state = {
  drawing: false,
  lastPoint: null,
  activeJobId: null,
  models: [],
};

function clearCanvas() {
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function drawLine(from, to) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#111111";
  ctx.lineWidth = 28;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

function strokeStart(event) {
  state.drawing = true;
  state.lastPoint = pointerPosition(event);
}

function strokeMove(event) {
  if (!state.drawing || !state.lastPoint) {
    return;
  }
  const next = pointerPosition(event);
  drawLine(state.lastPoint, next);
  state.lastPoint = next;
}

function strokeEnd() {
  state.drawing = false;
  state.lastPoint = null;
}

function renderModelOptions() {
  const optionsHtml = state.models
    .map(
      (model) =>
        `<option value="${model.id}">${model.name}${model.available ? "" : " (missing)"}</option>`,
    )
    .join("");
  modelSelect.innerHTML = optionsHtml;
  baseModelSelect.innerHTML = optionsHtml;

  const availableModels = state.models.filter((model) => model.available);
  const preferred =
    availableModels.find((model) => model.id === "original") || availableModels[0] || state.models[0];
  if (preferred) {
    modelSelect.value = preferred.id;
    baseModelSelect.value = preferred.id;
  }
}

function renderTopK(topK) {
  if (!Array.isArray(topK) || topK.length === 0) {
    topKList.innerHTML = `<div class="job-state">Run a prediction to see probabilities.</div>`;
    return;
  }

  topKList.innerHTML = topK
    .map(
      (item) => `
        <div class="topk-row">
          <strong>${item.digit}</strong>
          <div class="bar"><span style="width: ${Math.max(2, item.probability * 100)}%"></span></div>
          <span>${(item.probability * 100).toFixed(1)}%</span>
        </div>
      `,
    )
    .join("");
}

function showStatus(message) {
  jobState.textContent = message;
}

async function loadModels() {
  const response = await fetch("/api/models");
  const data = await response.json();
  state.models = data.models || [];
  renderModelOptions();
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    healthPill.textContent = `Backend ready · ${data.device}`;
  } catch (error) {
    healthPill.textContent = "Backend offline";
  }
}

function drawImageOnCanvas(image) {
  clearCanvas();
  const size = canvas.width;
  const scale = Math.min(size / image.width, size / image.height) * 0.88;
  const width = image.width * scale;
  const height = image.height * scale;
  const offsetX = (size - width) / 2;
  const offsetY = (size - height) / 2;
  ctx.drawImage(image, offsetX, offsetY, width, height);
}

async function loadFileToCanvas(file) {
  if (!file) {
    return;
  }
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    drawImageOnCanvas(image);
    URL.revokeObjectURL(url);
  };
  image.src = url;
}

async function canvasBlob() {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

async function predictDigit() {
  const blob = await canvasBlob();
  if (!blob) {
    showStatus("Could not export the canvas.");
    return;
  }

  const formData = new FormData();
  formData.append("file", blob, "digit.png");
  formData.append("model_id", modelSelect.value);

  predictBtn.disabled = true;
  showStatus("Predicting...");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed");
    }

    predictionValue.textContent = String(data.prediction);
    predictionMeta.textContent = `Model ${data.model.name} · ${(data.confidence * 100).toFixed(1)}% confidence`;
    renderTopK(data.top_k);
    showStatus("Prediction complete.");
  } catch (error) {
    showStatus(error.message);
  } finally {
    predictBtn.disabled = false;
  }
}

async function pollJob(jobId, modelId) {
  const tick = async () => {
    const response = await fetch(`/api/jobs/${jobId}`);
    const data = await response.json();

    if (data.status === "queued") {
      showStatus("Training queued...");
      return window.setTimeout(tick, 1800);
    }

    if (data.status === "running") {
      const progress = data.progress || {};
      const epoch = progress.epoch || 0;
      const epochs = progress.epochs || "?";
      showStatus(`Training epoch ${epoch} of ${epochs}...`);
      return window.setTimeout(tick, 1800);
    }

    if (data.status === "completed") {
      showStatus("Temporary model ready.");
      await loadModels();
      modelSelect.value = modelId;
      baseModelSelect.value = modelId;
      return;
    }

    showStatus(data.error || "Training failed.");
  };

  tick().catch((error) => showStatus(error.message));
}

async function retrainModel() {
  retrainBtn.disabled = true;
  showStatus("Starting server-side training...");
  try {
    const response = await fetch("/api/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_model_id: baseModelSelect.value,
        epochs: Number(epochsInput.value),
        lr: Number(lrInput.value),
        batch_size: Number(batchSizeInput.value),
        download: downloadInput.checked,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Training request failed");
    }

    state.activeJobId = data.job_id;
    showStatus(`Training ${data.model_id}...`);
    await pollJob(data.job_id, data.model_id);
  } catch (error) {
    showStatus(error.message);
  } finally {
    retrainBtn.disabled = false;
  }
}

canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  strokeStart(event);
});
canvas.addEventListener("pointermove", strokeMove);
canvas.addEventListener("pointerup", strokeEnd);
canvas.addEventListener("pointerleave", strokeEnd);
canvas.addEventListener("pointercancel", strokeEnd);

clearBtn.addEventListener("click", () => {
  clearCanvas();
  predictionValue.textContent = "-";
  predictionMeta.textContent = "No prediction yet.";
  renderTopK([]);
  showStatus("Canvas cleared.");
});

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files || [];
  if (file) {
    loadFileToCanvas(file);
  }
});

document.addEventListener("paste", (event) => {
  const items = event.clipboardData?.items || [];
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) {
        loadFileToCanvas(file);
        showStatus("Pasted image onto the slate.");
        break;
      }
    }
  }
});

predictBtn.addEventListener("click", predictDigit);
retrainBtn.addEventListener("click", retrainModel);

clearCanvas();
renderTopK([]);

Promise.all([loadHealth(), loadModels()]).catch((error) => {
  showStatus(error.message);
});
