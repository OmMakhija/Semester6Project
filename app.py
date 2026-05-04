"""
server/app.py — FastAPI backend for Microplastics Detector
"""

import io
import sys
import time
import json
import datetime
import urllib.parse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model import load_model

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
WEIGHTS_PATH = BASE_DIR / "best_model.pth"
NUM_CLASSES  = 2
CONF_THRESH  = 0.45
INPUT_SIZE   = 320
MAX_IMAGE_MB = 10

# 3-tier threshold: 0 = clean, 1-7 = clean, 8-15 = moderate, 16+ = contaminated
def get_rating(count: int) -> dict:
    if count <= 7:
        return {
            "label":    "Clean",
            "color":    "#22c55e",
            "message":  f"✅ {'No microplastics' if count == 0 else str(count) + ' particle' + ('' if count == 1 else 's')} detected. Water is safe.",
            "score":    10.0,
            "tier":     "clean",
        }
    elif count <= 15:
        return {
            "label":    "Moderate",
            "color":    "#f0a020",
            "message":  f"⚠️ {count} particles detected. Water quality is borderline.",
            "score":    round(5.0 + (15 - count) / 8 * 3, 1),
            "tier":     "moderate",
        }
    else:
        return {
            "label":    "Contaminated",
            "color":    "#ef4444",
            "message":  f"🔴 {count} particles detected. Water is contaminated.",
            "score":    round(max(0.0, 5.0 - (count - 15) * 0.2), 1),
            "tier":     "contaminated",
        }

HISTORY_FILE = BASE_DIR / "history.json"

def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_to_history(entry):
    history = load_history()
    history.append(entry)
    history = history[-50:]   # keep last 50
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


app = FastAPI(title="Microplastics Detector", version="1.0.0")

# ─────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────
model  = None
device = None

@app.on_event("startup")
def load():
    global model, device
    if not WEIGHTS_PATH.exists():
        print(f"⚠️  WARNING: weights not found at {WEIGHTS_PATH}")
        print("    Run: python main.py --data_dir dataset --epochs 100")
        return
    model, device = load_model(str(WEIGHTS_PATH), num_classes=NUM_CLASSES)
    print(f"✅  Model loaded from {WEIGHTS_PATH} on {device}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "device":       str(device) if device else None,
    }


@app.get("/history")
def get_history_stats():
    history = load_history()
    total_tests = len(history)
    detections  = sum(1 for e in history if e["count"] > 0)

    since = "0d"
    if history:
        try:
            first_date = datetime.datetime.fromisoformat(history[0]["timestamp"])
            days  = (datetime.datetime.now() - first_date).days
            since = f"{max(1, days)}d"
        except Exception:
            since = "1d"

    return {
        "total_tests": total_tests,
        "detections":  detections,
        "since":       since,
        "items":       history[::-1],   # latest first
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf_thresh: float = CONF_THRESH):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run training first.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected an image, got: {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image too large. Max {MAX_IMAGE_MB} MB.")

    try:
        img_orig = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    orig_w, orig_h = img_orig.size
    transform  = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor()])
    img_tensor = transform(img_orig).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        preds = model(img_tensor)[0]
    elapsed_ms = int((time.time() - t0) * 1000)

    boxes  = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    sx = orig_w / INPUT_SIZE
    sy = orig_h / INPUT_SIZE
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    keep  = scores >= conf_thresh
    count = int(keep.sum())

    rating = get_rating(count)

    entry = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "filename":     file.filename,
        "count":        count,
        "conf_thresh":  conf_thresh,
        "rating_label": rating["label"],
        "rating_color": rating["color"],
        "rating_score": rating["score"],
        "tier":         rating["tier"],
        "inference_ms": elapsed_ms,
    }
    save_to_history(entry)

    buf = io.BytesIO()
    img_orig.save(buf, format="JPEG", quality=92)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/jpeg",
        headers={
            "X-Inference-Ms":   str(elapsed_ms),
            "X-Conf-Threshold": str(conf_thresh),
            "X-Particle-Count": str(count),
            "X-Rating-Tier":    rating["tier"],
            "X-Rating-Label":   rating["label"],
            "X-Rating-Color":   rating["color"],
            "X-Rating-Score":   str(rating["score"]),
            "X-Rating-Message": urllib.parse.quote(rating["message"]),
            "Access-Control-Expose-Headers": (
                "X-Inference-Ms, X-Conf-Threshold, X-Particle-Count, "
                "X-Rating-Tier, X-Rating-Label, X-Rating-Color, "
                "X-Rating-Score, X-Rating-Message"
            ),
        },
    )


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)