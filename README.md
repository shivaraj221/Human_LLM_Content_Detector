# AI Text Detector (FastAPI)

## Run locally
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload
Open http://localhost:8000

## Deploy on Render
1. Push this folder to a GitHub repo.
2. Render dashboard -> New -> Blueprint -> pick the repo (it reads render.yaml).
   Or New -> Web Service with:
   - Build:  pip install -r requirements.txt && python download_model.py
   - Start:  uvicorn app.main:app --host 0.0.0.0 --port $PORT
   - Env:    PYTHON_VERSION=3.11.9, HF_HOME=/opt/render/project/src/.hf_cache
3. Health check path: /health
