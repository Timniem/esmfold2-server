#!/usr/bin/env python3
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from transformers import EsmForProteinFolding


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MODEL_NAME = os.getenv("ESMFOLD_MODEL", "facebook/esmfold_v1")
CHUNK_SIZE = int(os.getenv("ESMFOLD_CHUNK_SIZE", "128"))
USE_FP16 = os.getenv("ESMFOLD_FP16", "1") != "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("esmfold-service")


class FoldRequest(BaseModel):
    sequence: str = Field(..., description="Protein sequence using the 20 standard amino acids")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_sequence(sequence: str) -> str:
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("Sequence is empty.")

    bad = sorted({aa for aa in sequence if aa not in VALID_AMINO_ACIDS})
    if bad:
        raise ValueError(f"Invalid amino acid characters: {''.join(bad)}")

    return sequence


def load_model(device: torch.device) -> EsmForProteinFolding:
    t0 = time.time()

    model = EsmForProteinFolding.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
    )

    try:
        model.trunk.set_chunk_size(CHUNK_SIZE)
    except AttributeError:
        pass

    model = model.to(device)

    fp16_active = False
    if device.type == "cuda" and USE_FP16:
        model.esm = model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True
        fp16_active = True

    model.eval()

    dt = time.time() - t0
    log.info(
        "model loaded model=%s device=%s chunk_size=%s fp16=%s load_time_sec=%.2f",
        MODEL_NAME,
        device,
        CHUNK_SIZE,
        fp16_active,
        dt,
    )

    if device.type == "cuda":
        log.info("cuda name=%s", torch.cuda.get_device_name(0))

    return model


def infer_pdb(model: EsmForProteinFolding, sequence: str) -> str:
    t0 = time.time()
    log.info("inference start seq_len=%d", len(sequence))
    with torch.no_grad():
        pdb = model.infer_pdb(sequence)
    dt = time.time() - t0
    log.info("inference done seq_len=%d time_sec=%.2f", len(sequence), dt)
    return pdb


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = pick_device()
    model = load_model(device)
    app.state.device = device
    app.state.model = model
    app.state.model_lock = threading.Lock()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "device": str(app.state.device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "fp16_enabled": bool(app.state.device.type == "cuda" and USE_FP16),
    }


@app.post("/fold")
def fold(req: FoldRequest) -> Response:
    try:
        sequence = validate_sequence(req.sequence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        with app.state.model_lock:
            pdb = infer_pdb(app.state.model, sequence)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    return Response(
        content=pdb,
        media_type="chemical/x-pdb",
        headers={"Content-Disposition": 'attachment; filename="prediction.pdb"'},
    )
