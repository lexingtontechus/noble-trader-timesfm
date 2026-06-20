"""
TimesFM Forecasting API
Exposes google/timesfm-2.5-200m-pytorch as an HTTP service.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import timesfm
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("timesfm-server")

# --- Model configuration -----------------------------------------------------

MODEL_ID = os.getenv("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
MAX_CONTEXT = int(os.getenv("TIMESFM_MAX_CONTEXT", "1024"))
MAX_HORIZON = int(os.getenv("TIMESFM_MAX_HORIZON", "256"))
PORT = int(os.getenv("PORT", "8000"))

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

# --- App state (model loaded once at startup) --------------------------------

state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Loading TimesFM model: {MODEL_ID}")
    t0 = time.time()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    state["model"] = model
    log.info(f"Model loaded in {time.time() - t0:.1f}s")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="TimesFM Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — tighten this in production if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas -----------------------------------------------------------------

class ForecastRequest(BaseModel):
    horizon: int = Field(..., ge=1, le=MAX_HORIZON, description="Forecast horizon length")
    inputs: list[list[float]] = Field(
        ..., min_length=1,
        description="List of input time series. Each series is a list of floats (max length = max_context)."
    )

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, v):
        for i, series in enumerate(v):
            if len(series) == 0:
                raise ValueError(f"inputs[{i}] is empty")
            if len(series) > MAX_CONTEXT:
                raise ValueError(f"inputs[{i}] length {len(series)} exceeds max_context {MAX_CONTEXT}")
        return v


class ForecastResponse(BaseModel):
    horizon: int
    num_series: int
    point_forecast: list[list[float]]            # shape: (num_series, horizon)
    quantile_forecast: list[list[list[float]]]   # shape: (num_series, horizon, num_quantiles)
    elapsed_ms: float


# --- Routes ------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness/readiness probe target."""
    return {
        "status": "ok" if state["model"] is not None else "loading",
        "device": device,
        "model": MODEL_ID,
    }


@app.get("/info")
def info():
    return {
        "model": MODEL_ID,
        "device": device,
        "max_context": MAX_CONTEXT,
        "max_horizon": MAX_HORIZON,
        "gpu_memory_allocated_mb": (
            round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            if device == "cuda" else None
        ),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    try:
        inputs = [np.array(s, dtype=np.float32) for s in req.inputs]
        t0 = time.time()
        point, quantile = state["model"].forecast(
            horizon=req.horizon,
            inputs=inputs,
        )
        elapsed_ms = (time.time() - t0) * 1000

        return ForecastResponse(
            horizon=req.horizon,
            num_series=len(req.inputs),
            point_forecast=point.tolist(),
            quantile_forecast=quantile.tolist(),
            elapsed_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        log.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        # Single worker — TimesFM holds GPU state, multi-worker would OOM the T4
        workers=1,
    )