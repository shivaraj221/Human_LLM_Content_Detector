import math
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "ShantanuT01/gradient-ai-text-detector"

# The model has ONE output score. High score is assumed to mean AI-written.
# If your own tests show the opposite, change this to False.
HIGH_MEANS_AI = True

MIN_WORDS = 30
MAX_WORDS = 5000
BASE_DIR = Path(__file__).resolve().parent

state = {}
infer_lock = threading.Lock()  # one analysis at a time keeps RAM/CPU predictable


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...", flush=True)
    state["tok"] = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    state["model"] = model
    print("Model ready.", flush=True)
    yield
    state.clear()


app = FastAPI(title="AI Text Detector", lifespan=lifespan)


def run_model(input_ids, attention_mask):
    with torch.no_grad():
        out = state["model"](input_ids=input_ids, attention_mask=attention_mask).logits
    return out.reshape(-1)[0].item()


def score(text: str):
    tok = state["tok"]
    try:
        enc = tok(text, truncation=True, max_length=512, stride=64,
                  return_overflowing_tokens=True, padding=True,
                  return_tensors="pt")
        values = [
            run_model(enc["input_ids"][i:i + 1], enc["attention_mask"][i:i + 1])
            for i in range(enc["input_ids"].shape[0])
        ]
        return sum(values) / len(values), len(values)
    except Exception:
        enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
        return run_model(enc["input_ids"], enc["attention_mask"]), 1


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if "model" not in state:
        raise HTTPException(503, "Model is still loading, try again shortly.")

    text = req.text.strip()
    words = len(text.split())
    if words < MIN_WORDS:
        raise HTTPException(400, f"Too short for a reliable result ({words} words). Paste ~100+ words.")
    if words > MAX_WORDS:
        raise HTTPException(400, f"Too long. Limit is {MAX_WORDS} words.")

    with infer_lock:
        raw, chunks = score(text)

    high_pct = 100 / (1 + math.exp(-raw))
    ai_pct = high_pct if HIGH_MEANS_AI else 100 - high_pct
    human_pct = 100 - ai_pct

    if ai_pct >= 70:
        verdict = "Likely AI generated"
    elif ai_pct <= 30:
        verdict = "Likely human written"
    else:
        verdict = "Uncertain"

    return {
        "words": words,
        "chunks": chunks,
        "raw_score": round(raw, 3),
        "ai_pct": round(ai_pct, 1),
        "human_pct": round(human_pct, 1),
        "verdict": verdict,
    }


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


