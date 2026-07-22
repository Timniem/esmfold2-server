# esmfold-server
A minimal HTTP service for folding amino acid sequences with ESMFold2 and returning a mmCIF file.

## Features
- Loads the ESMFold model once at startup
- Accepts a protein sequence via HTTP
- Returns the predicted (complex) structure as a `.mmcif` file
- Automatically selects the best available device:
  - CUDA
  - Apple Metal
  - CPU
- Uses fp16 on CUDA by default

## Requirements
- Python 3.11+
- `uv`
- PyTorch
- FastAPI
- Transformers
- ESMFold model weights available locally if running with `HF_HUB_OFFLINE=1` (pip install esm@git+https://github.com/Biohub/esm.git@main)

## Install (allow online mode once)
```
uv sync
HF_HUB_OFFLINE=0 uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

## Run (offline)
```
uv run uvicorn esmfold-server:app --host 0.0.0.0 --port 8000
```

## Health check
```
curl http://127.0.0.1:8000/health
```

Example response:
```
{
  "status": "ok",
  "device": "mps",
  "cuda_available": false,
  "mps_available": true,
  "fp16_enabled": false
}
```

Rules:
- Sequence is required
- Only the 20 standard amino acids are accepted
- Lowercase input is converted to uppercase

## Response
Returns the predicted structure as a mmcif file with content type `chemical/x-mmcif`.

## Environment variables
- ESMFOLD_MODEL: Model name, default: facebook/esmfold_v1
- ESMFOLD_CHUNK_SIZE: Trunk chunk size, default: 128
- ESMFOLD_FP16: Enable fp16 on CUDA, default: 1
- HF_HUB_OFFLINE: Put Hugging Face Hub access into offline mode, default: 1

## Notes
- FP16 is only enabled on CUDA
- On Apple Silicon, the service uses mps if available
- Requests are serialized with a lock so one fold runs at a time per service instance

## Example requests

#### Single protein (simple)
```
curl -X POST http://localhost:8000/fold \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [{"id": "A", "sequence": "MAKTPSDHLLSTLEELVPYDFEKFKFKLQNTSVQKEHSRIPRSQIQRARPVKMATLLVTY", "input_type": "protein"}],
  }'
```
#### Protein-protein complex
```
curl -X POST http://localhost:8000/fold \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      {"id": "A", "sequence": "MAKTPSDHLLSTLEELVPYDFEKFKFKLQNTSVQKEHSRIPRSQIQRARPVKMATLLVTY...", "input_type": "protein"},
      {"id": "B", "sequence": "MDDREDLVYQAKLAEQAERYDEMVESMKKVAGMDVELTVEERNLLSVAYKNVIGARRASW...", "input_type": "protein"}
    ],
    "num_loops": 20,
    "num_sampling_steps": 100,
  }'
```

