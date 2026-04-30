# Try-On FastAPI Server (CatVTON + OOTDiffusion)

This server runs try-on generation remotely and exposes a simple HTTP API.

## Install

Create/activate your server Python environment (ideally on the GPU machine), then:

```bash
pip install -r requirements.tryon_server.txt
```

## Run

Option A (recommended):

```bash
python scripts/run_tryon_server.py --host 0.0.0.0 --port 8000
```

Option B (uvicorn directly):

```bash
uvicorn postvton.tryon_server.app:app --host 0.0.0.0 --port 8000
```

Environment variables:

- `TRYON_SERVER_DEVICE` = `cuda` (default) or `cpu`
- `TRYON_SERVER_OUTPUT_DIR` = output root directory (default `outputs/`)

## API

### Health

`GET /health`

Response:

```json
{"status": "ok"}
```

### Try-on

`POST /tryon` (multipart/form-data)

Fields:
- `cloth_type`: `upper|lower|overall` (default: `upper`)
- `num_inference_steps`: int (default: 10)
- `guidance_scale`: float (default: 2.5)
- `seed`: int (default: -1)

Files:
- `person_image`
- `cloth_image`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/tryon" \
  -F "cloth_type=upper" \
  -F "person_image=@person.png" \
  -F "cloth_image=@cloth.png"
```

Response:

```json
{
  "success": true,
  "job_id": "...",
  "model_used": "catvton|ootdiffusion-hd|ootdiffusion-dc",
  "pose_score": 0.0,
  "inference_time": 0.0,
  "output_url": "/outputs/<job_id>.png",
  "candidates": [...]
}
```

### Download output

`GET /outputs/{name}`

Example:

```bash
curl -o out.png "http://127.0.0.1:8000/outputs/<job_id>.png"
```

## Use From PostVTON CLI (Client)

Once the server is running, point the PostVTON pipeline at it:

```bash
python -m postvton.pipeline \
  --model-image path/to/person.png \
  --garment-image path/to/cloth.png \
  --cloth-type upper \
  --tryon-server-url http://127.0.0.1:8000
```

Note: The client no longer runs CatVTON/OOTDiffusion locally; `--tryon-server-url`
or `TRYON_SERVER_URL` is required for try-on.
