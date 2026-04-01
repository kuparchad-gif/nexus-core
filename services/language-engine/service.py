"""
Dakar Language Engine Service — FastAPI wrapper for Railway deployment.

Endpoints:
    POST /generate       — Generate text from a 50D Dakar state vector
    POST /train          — Train on a batch of memory pairs
    POST /dream          — Run a dream cycle training pass
    GET  /status         — Engine status and vocabulary info
    GET  /health         — Health check
    POST /weights/save   — Get weight particles for holocell storage
    POST /weights/load   — Load weight particles from holocells
"""

import os
import time
import logging
from typing import List

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the engine — in the Docker image this is copied alongside service.py
from dakar_language_engine import (
    DakarTrainingLoop,
    create_language_engine,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dakar-language-service")

# ---------------------------------------------------------------------------
# Initialize engine
# ---------------------------------------------------------------------------

engine, particle_manager = create_language_engine(vocab_size=4096, embedding_dim=50)
trainer = DakarTrainingLoop(engine)

logger.info(
    "Dakar Language Engine initialized: vocab_size=%d, embedding_dim=%d",
    engine.vocab_size,
    engine.embedding_dim,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dakar Language Engine",
    description="Generative language from 50D weight particles",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request to generate text from a Dakar state vector."""

    dakar_state: List[float]
    max_length: int = 50
    temperature: float = 0.8
    top_k: int = 50


class GenerateResponse(BaseModel):
    """Generated text response."""

    tokens: List[str]
    text: str
    token_count: int
    generation_time_ms: float


class TrainRequest(BaseModel):
    """Request to train on a batch of memory pairs."""

    encodings: List[List[float]]  # [batch_size, 50]
    sequences: List[List[int]]  # [batch_size, seq_len]


class TrainResponse(BaseModel):
    """Training result."""

    loss: float
    batch_size: int


class DreamRequest(BaseModel):
    """Request to run a dream cycle."""

    memories: List[dict]  # [{encoding: [50], tokens: [int]}]


class WeightState(BaseModel):
    """Weight particle state for holocell persistence."""

    particles: dict  # name -> list of lists


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "alive", "engine": "dakar-language", "version": "1.0.0"}


@app.get("/status")
async def status() -> dict:
    """Engine status."""
    param_count = sum(p.numel() for p in engine.parameters())
    return {
        "vocab_size": engine.vocab_size,
        "embedding_dim": engine.embedding_dim,
        "parameter_count": param_count,
        "parameter_size_kb": param_count * 4 / 1024,
        "stop_idx": engine.stop_idx,
        "sample_tokens": list(engine.vocab.token_to_idx.keys())[:20],
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text from a 50D Dakar state vector."""
    start = time.time()

    # Convert input to tensor
    state_vec = req.dakar_state
    if len(state_vec) < engine.embedding_dim:
        state_vec = state_vec + [0.0] * (engine.embedding_dim - len(state_vec))
    elif len(state_vec) > engine.embedding_dim:
        state_vec = state_vec[: engine.embedding_dim]

    dakar_state = torch.tensor(state_vec, dtype=torch.float32)

    # Generate
    tokens = engine.generate(
        dakar_state,
        max_length=req.max_length,
        temperature=req.temperature,
        top_k=req.top_k,
    )

    elapsed_ms = (time.time() - start) * 1000

    return GenerateResponse(
        tokens=tokens,
        text=" ".join(tokens),
        token_count=len(tokens),
        generation_time_ms=round(elapsed_ms, 2),
    )


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest) -> TrainResponse:
    """Train on a batch of Dakar encoding + token sequence pairs."""
    encodings = torch.tensor(req.encodings, dtype=torch.float32)

    # Pad sequences to same length
    max_len = max(len(s) for s in req.sequences)
    padded = [s + [0] * (max_len - len(s)) for s in req.sequences]
    sequences = torch.tensor(padded, dtype=torch.long)

    loss = trainer.train_on_memory(encodings, sequences)

    return TrainResponse(loss=loss, batch_size=len(req.encodings))


@app.post("/dream")
async def dream(req: DreamRequest) -> dict:
    """Run a dream cycle training pass."""
    memories = []
    for m in req.memories:
        enc = m.get("encoding", [])
        toks = m.get("tokens", [])
        if enc and toks:
            memories.append((torch.tensor(enc, dtype=torch.float32), toks))

    if not memories:
        return {"loss": 0.0, "memories_processed": 0}

    loss = trainer.dream_cycle(memories)
    return {"loss": loss, "memories_processed": len(memories)}


@app.post("/weights/save")
async def weights_save() -> dict:
    """Get current weight particles for holocell storage."""
    state = particle_manager.get_particle_state()
    return {"particles": state, "vocab_size": engine.vocab_size}


@app.post("/weights/load")
async def weights_load(req: WeightState) -> dict:
    """Load weight particles from holocells."""
    particle_manager.load_particle_state(req.particles)
    return {"status": "loaded", "particles_loaded": len(req.particles)}


@app.get("/vocab")
async def vocab_info() -> dict:
    """Get vocabulary information."""
    return {
        "size": engine.vocab_size,
        "tokens": list(engine.vocab.token_to_idx.keys()),
    }


@app.on_event("startup")
async def startup() -> None:
    """Log startup."""
    logger.info("Dakar Language Engine service starting on Railway")
    logger.info(
        "Vocab: %d tokens, Params: %d",
        engine.vocab_size,
        sum(p.numel() for p in engine.parameters()),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
