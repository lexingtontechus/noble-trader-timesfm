"""
TimesFM Forecasting API
Exposes google/timesfm-2.5-200m-pytorch as an HTTP service.

Hardening:
- Per-key rate limiting (sliding window, in-memory)
- Structured request logging (key ID, endpoint, status, latency)
- Security headers (HSTS, nosniff, noframe)
- Graceful error handling
"""
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import numpy as np
import torch
import timesfm
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
import uvicorn

# --- Logging -----------------------------------------------------------------

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

# Rate limiting config
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
RATE_LIMIT_WINDOW_S = 60

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

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
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — tighten this in production if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Security headers middleware (Step 6) ------------------------------------

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # HSTS — tells browsers to always use HTTPS for the next year
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # Prevent MIME-type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    # Don't cache API responses (forecasts are time-sensitive)
    response.headers["Cache-Control"] = "no-store"
    return response


# --- Request logging middleware (Step 4) -------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000

    # Extract key ID from header (first 8 chars only — never log full key)
    api_key = request.headers.get("X-API-Key", "")
    key_id = api_key[:8] + "…" if api_key else "-"

    log.info(
        f"req method={request.method} path={request.url.path} "
        f"status={response.status_code} latency_ms={elapsed_ms:.1f} "
        f"key={key_id} client={request.client.host if request.client else '-'}"
    )
    return response


# --- Auth + Rate limiting (Steps 3 + auth) -----------------------------------

VALID_KEYS = frozenset(
    k.strip() for k in os.getenv("TIMESFM_API_KEYS", "").split(",") if k.strip()
)

if not VALID_KEYS:
    dev_key = secrets.token_urlsafe(32)
    VALID_KEYS = frozenset({dev_key})
    log.warning(f"No TIMESFM_API_KEYS env var set. Generated dev key: {dev_key}")
    log.warning("Set TIMESFM_API_KEYS in Studio env vars for production.")
else:
    log.info(f"Loaded {len(VALID_KEYS)} API key(s)")
    log.info(f"Rate limit: {RATE_LIMIT_PER_MIN} requests/min per key")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Per-key sliding-window rate limiter (in-memory; fine for single-worker)
request_history: dict[str, deque] = defaultdict(lambda: deque())


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    # Constant-time key comparison
    key_valid = False
    for valid_key in VALID_KEYS:
        if secrets.compare_digest(api_key, valid_key):
            key_valid = True
            break

    if not key_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Rate limiting: sliding window
    now = time.time()
    history = request_history[api_key]
    # Drop entries older than the window
    while history and history[0] < now - RATE_LIMIT_WINDOW_S:
        history.popleft()

    if len(history) >= RATE_LIMIT_PER_MIN:
        # Compute seconds until oldest request exits the window
        retry_after = int(history[0] + RATE_LIMIT_WINDOW_S - now) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT_PER_MIN} requests per {RATE_LIMIT_WINDOW_S}s",
            headers={"Retry-After": str(retry_after)},
        )

    history.append(now)
    return api_key


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
    point_forecast: list[list[float]]
    quantile_forecast: list[list[list[float]]]
    elapsed_ms: float

# --- Routes ------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness/readiness probe target. No auth required."""
    return {
        "status": "ok" if state["model"] is not None else "loading",
        "device": device,
        "model": MODEL_ID,
    }


@app.get("/info", dependencies=[Depends(verify_api_key)])
def info():
    return {
        "model": MODEL_ID,
        "device": device,
        "max_context": MAX_CONTEXT,
        "max_horizon": MAX_HORIZON,
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "gpu_memory_allocated_mb": (
            round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            if device == "cuda" else None
        ),
    }


@app.post("/forecast", response_model=ForecastResponse, dependencies=[Depends(verify_api_key)])
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
        workers=1,
    )