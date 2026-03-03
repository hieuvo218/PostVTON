from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import io
import uuid
from pathlib import Path
import sys

# Add external CatVTON to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from postvton.tools.tryon.catvton import CatVTONInference

app = FastAPI(title="VTON Server", version="1.0.0")

# Global model instance
_catvton: CatVTONInference = None


class GenerateRequest(BaseModel):
    num_inference_steps: int = 50
    guidance_scale: float = 2.5
    seed: int | None = None


class GenerateResponse(BaseModel):
    success: bool
    output_path: str | None = None
    error: str | None = None


@app.on_event("startup")
async def startup():
    global _catvton
    _catvton = CatVTONInference()
    print("CatVTON model loaded successfully")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": _catvton is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate_tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    cloth_type: str = "upper",
    num_inference_steps: int = 50,
    guidance_scale: float = 2.5,
    seed: int | None = None,
    show_type: str = "result only"
):
    """Generate virtual try-on image"""
    try:
        # Load images from uploads
        person_img = Image.open(io.BytesIO(await person_image.read())).convert('RGB')
        cloth_img = Image.open(io.BytesIO(await cloth_image.read())).convert('RGB')
        
        # Generate try-on
        result = _catvton.generate(
            person_image=person_img,
            cloth_image=cloth_img,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            # return_intermediate=False  # Return only the output image
        )
        
        # Save output
        output_dir = Path("outputs/vton_server")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{uuid.uuid4()}.png"
        
        # Handle both dict and Image results
        if isinstance(result, dict):
            result['output'].save(output_path)
        else:
            result.save(output_path)
        
        return GenerateResponse(success=True, output_path=str(output_path))
        
    except Exception as e:
        return GenerateResponse(success=False, error=str(e))


@app.get("/output/{filename}")
async def get_output(filename: str):
    """Download generated image"""
    file_path = Path("outputs/vton_server") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)