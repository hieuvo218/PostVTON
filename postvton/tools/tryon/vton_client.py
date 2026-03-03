import httpx
from pathlib import Path
from PIL import Image
import io
from typing import Optional
from pydantic import BaseModel


class VTONClient:
    """Client for VTON REST API server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        # 10 min timeout for generation (inference can take 2-5 minutes)
        self.client = httpx.AsyncClient(timeout=600.0)
    
    async def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def generate(
        self,
        person_image_path: str,
        cloth_image_path: str,
        cloth_type: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: Optional[int] = None
    ) -> dict:
        """
        Generate virtual try-on via REST API
        
        Args:
            person_image_path: Path to person image
            cloth_image_path: Path to garment image
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            seed: Random seed
            
        Returns:
            Dict with success status and output path
        """
        try:
            # Prepare files
            with open(person_image_path, "rb") as f:
                person_data = f.read()
            with open(cloth_image_path, "rb") as f:
                cloth_data = f.read()
            
            # Build request
            files = {
                "person_image": ("person.png", person_data, "image/png"),
                "cloth_image": ("garment.png", cloth_data, "image/png"),
            }
            params = {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
            if seed is not None:
                params["seed"] = seed
            
            # Send request
            response = await self.client.post(
                f"{self.base_url}/generate",
                files=files,
                params=params
            )
            response.raise_for_status()
            
            return response.json()
            
        except httpx.TimeoutException as e:
            return {
                "success": False, 
                "error": f"Request timeout after {self.client.timeout.read}s. Inference may take 2-5 minutes for 50 steps."
            }
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP error {e.response.status_code}: {e.response.text}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def download_output(self, filename: str, save_path: str) -> bool:
        """Download generated image from server"""
        try:
            response = await self.client.get(f"{self.base_url}/output/{filename}")
            response.raise_for_status()
            
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()