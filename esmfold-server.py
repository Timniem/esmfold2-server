#!/usr/bin/env python3
import logging
import os
import threading
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, List, Optional, Union

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, validator

# ESMFold2 imports
from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    LigandInput,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
VALID_NUCLEOTIDES = set("ACGTU")
MODEL_NAME = os.getenv("ESMFOLD_MODEL", "biohub/ESMFold2")
USE_FP16 = os.getenv("ESMFOLD_FP16", "1") != "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("esmfold-service")

class InputType(str, Enum):
    PROTEIN = "protein"
    DNA = "dna"
    LIGAND = "ligand"

class SequenceInput(BaseModel):
    id: str = Field(..., description="Chain identifier (e.g., 'A', 'B')")
    sequence: str = Field(..., description="Biological sequence")
    input_type: InputType = Field(
        InputType.PROTEIN, description="Type of input: protein, dna, or ligand"
    )

    @validator("sequence")
    def validate_sequence(cls, v: str, values: dict[str, Any]) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("Sequence is empty.")

        input_type = values.get("input_type", InputType.PROTEIN)
        if input_type == InputType.PROTEIN:
            bad = {aa for aa in v if aa not in VALID_AMINO_ACIDS}
        elif input_type in (InputType.DNA, InputType.LIGAND):
            bad = {nt for nt in v if nt not in VALID_NUCLEOTIDES}
        else:
            bad = set()

        if bad:
            raise ValueError(f"Invalid characters for {input_type.value}: {''.join(sorted(bad))}")
        return v

class FoldRequest(BaseModel):
    sequences: List[SequenceInput] = Field(
        ...,
        description="List of sequences to fold. Can be proteins, DNA, or ligands.",
        min_items=1,
    )
    num_loops: Optional[int] = Field(
        20, ge=1, le=100, description="Number of diffusion loops"
    )
    num_sampling_steps: Optional[int] = Field(
        100, ge=1, description="Number of sampling steps per loop"
    )
    num_diffusion_samples: Optional[int] = Field(
        1, ge=1, description="Number of diffusion samples"
    )
    seed: Optional[int] = Field(0, description="Random seed for reproducibility")
    output_format: Optional[str] = Field(
        "pdb", description="Output format: 'pdb' or 'mmcif'"
    )

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(device: torch.device) -> ESMFold2Model:
    t0 = time.time()
    log.info("Loading ESMFold2 model: %s", MODEL_NAME)

    model = ESMFold2Model.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
    ).to(device)

    fp16_active = False
    if device.type == "cuda" and USE_FP16:
        model = model.half()
        torch.backends.cuda.matmul.allow_tf32 = True
        fp16_active = True

    model.eval()

    dt = time.time() - t0
    log.info(
        "model loaded model=%s device=%s fp16=%s load_time_sec=%.2f",
        MODEL_NAME, device, fp16_active, dt,
    )
    if device.type == "cuda":
        log.info("cuda name=%s", torch.cuda.get_device_name(0))

    return model

def build_structure_prediction_input(sequences: List[SequenceInput]) -> StructurePredictionInput:
    inputs = []
    for seq in sequences:
        if seq.input_type == InputType.PROTEIN:
            inputs.append(ProteinInput(id=seq.id, sequence=seq.sequence))
        elif seq.input_type == InputType.DNA:
            inputs.append(DNAInput(id=seq.id, sequence=seq.sequence))
        elif seq.input_type == InputType.LIGAND:
            inputs.append(LigandInput(id=seq.id, sequence=seq.sequence))
    return StructurePredictionInput(sequences=inputs)

def infer_structure(
    model: ESMFold2Model,
    spi: StructurePredictionInput,
    num_loops: int,
    num_sampling_steps: int,
    num_diffusion_samples: int,
    seed: int,
    output_format: str,
) -> tuple[str, dict[str, float]]:
    t0 = time.time()
    log.info(
        "inference start num_sequences=%d num_loops=%d num_steps=%d",
        len(spi.sequences), num_loops, num_sampling_steps,
    )

    with torch.no_grad():
        result = ESMFold2InputBuilder().fold(
            model,
            spi,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
        )

    dt = time.time() - t0
    metrics = {
        "plddt_mean": float(result.plddt.mean()),
        "ptm": float(result.ptm),
        "iptm": float(result.iptm),
        "inference_time_sec": dt,
    }
    log.info(
        "inference done pLDDT=%.3f pTM=%.3f ipTM=%.3f time_sec=%.2f",
        metrics["plddt_mean"], metrics["ptm"], metrics["iptm"], dt,
    )

    if output_format == "mmcif":
        output = result.complex.to_mmcif()
    else:
        output = result.complex.to_pdb()

    return output, metrics

import time

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
        "model": MODEL_NAME,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "fp16_enabled": bool(app.state.device.type == "cuda" and USE_FP16),
    }

@app.post("/fold")
def fold(req: FoldRequest) -> Response:
    try:
        spi = build_structure_prediction_input(req.sequences)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        with app.state.model_lock:
            output, metrics = infer_structure(
                app.state.model,
                spi,
                req.num_loops,
                req.num_sampling_steps,
                req.num_diffusion_samples,
                req.seed,
                req.output_format,
            )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    content_type = "chemical/x-pdb" if req.output_format == "pdb" else "chemical/x-mmcif"
    filename = f"prediction.{req.output_format}"

    # Add metrics to response headers
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        **{f"X-Metric-{k}": f"{v:.4f}" for k, v in metrics.items()},
    }

    return Response(content=output, media_type=content_type, headers=headers)