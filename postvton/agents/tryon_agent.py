"""TryOn Agent for Virtual Try-On using CatVTON or OOTDiffusion"""

import asyncio
import sys
import uuid
import time
import traceback
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from postvton.tools.tryon.catvton import CatVTONInference
from postvton.tools.tryon.ootdiffusion import OOTDiffusionInference


class VTONModel(str, Enum):
    """Available virtual try-on models"""
    CATVTON = "catvton"
    OOTDIFFUSION = "ootdiffusion"


@dataclass
class TryOnResult:
    """Result from a try-on operation"""
    success: bool
    output_path: Optional[str] = None
    message: str = ""
    model_used: Optional[str] = None
    inference_time: float = 0.0


class TryOnAgent:
    """Agent for orchestrating virtual try-on operations.

    Supports CatVTON and OOTDiffusion, runs inference locally.
    Models are lazy-loaded on first call.
    """

    def __init__(
        self,
        model: VTONModel = VTONModel.OOTDIFFUSION,
        device: str = "cuda",
    ):
        """Initialize TryOn Agent.

        Args:
            model: Which model to use (VTONModel.CATVTON or VTONModel.OOTDIFFUSION)
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.model_name = model if isinstance(model, VTONModel) else VTONModel(model)
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the selected model."""
        if self._model is not None:
            return

        print(f"[TryOnAgent] Loading {self.model_name} model on {self.device}...")

        if self.model_name == VTONModel.CATVTON:
            self._model = CatVTONInference(device=self.device)
        elif self.model_name == VTONModel.OOTDIFFUSION:
            gpu_id = 0 if self.device.startswith("cuda") else -1
            self._model = OOTDiffusionInference(model_type="hd", gpu_id=gpu_id)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        print(f"[TryOnAgent] {self.model_name} model loaded successfully")

    async def generate(
        self,
        person_image: Union[Image.Image, str],
        cloth_image: Union[Image.Image, str],
        cloth_type: str = "upper",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> TryOnResult:
        """Generate a virtual try-on result.

        Args:
            person_image: PIL Image or path to the person image
            cloth_image: PIL Image or path to the cloth/garment image
            cloth_type: Type of clothing.
                CatVTON: "upper", "lower", "overall", "inner", "outer"
                OOTDiffusion: "upperbody", "lowerbody", "dress"
            output_path: Where to save the output PNG.
                If None, auto-saved to outputs/tryon_agent/<model>_<uuid>.png
            **kwargs: Extra parameters forwarded to the model
                (e.g. num_inference_steps, guidance_scale, seed)

        Returns:
            TryOnResult with success, output_path, inference_time, etc.
        """
        start_time = time.time()

        try:
            self._load_model()

            # Convert string paths to PIL Images
            if isinstance(person_image, str):
                person_image = Image.open(person_image).convert("RGB")
            if isinstance(cloth_image, str):
                cloth_image = Image.open(cloth_image).convert("RGB")

            # OOTDiffusion expects seed as int, never None
            if "seed" in kwargs and kwargs["seed"] is None:
                kwargs["seed"] = -1

            # Run inference
            if self.model_name == VTONModel.CATVTON:
                result = self._model.generate(
                    person_image=person_image,
                    cloth_image=cloth_image,
                    cloth_type=cloth_type,
                    **kwargs,
                )
            elif self.model_name == VTONModel.OOTDIFFUSION:
                result = self._model.generate(
                    person_image=person_image,
                    cloth_image=cloth_image,
                    category=cloth_type,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            # Normalise result to a single PIL Image
            if isinstance(result, list):
                result = result[0]
            elif hasattr(result, "images"):
                result = result.images[0]

            if not isinstance(result, Image.Image):
                raise RuntimeError(f"Model returned unexpected type: {type(result)}")

            # Resolve output path
            if output_path is None:
                out_dir = project_root / "outputs" / "tryon_agent"
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(out_dir / f"{self.model_name}_{uuid.uuid4().hex[:8]}.png")
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            result.save(output_path)
            elapsed = time.time() - start_time
            print(f"[TryOnAgent] Saved result -> {output_path} ({elapsed:.2f}s)")

            return TryOnResult(
                success=True,
                output_path=output_path,
                message=f"Generated with {self.model_name}",
                model_used=self.model_name,
                inference_time=elapsed,
            )

        except Exception as e:
            traceback.print_exc()
            return TryOnResult(
                success=False,
                message=f"Inference failed: {e}",
                model_used=self.model_name,
                inference_time=time.time() - start_time,
            )

    def unload(self):
        """Unload models to free GPU memory."""
        if self._model is not None:
            if hasattr(self._model, "unload"):
                self._model.unload()
            self._model = None
        print("[TryOnAgent] Models unloaded")


# ---------------------------------------------------------------------------
# Async entry-point
# ---------------------------------------------------------------------------

async def run_tryon_agent(
    person_image_path: str,
    cloth_image_path: str,
    cloth_type: str = "upper",
    output_path: Optional[str] = None,
    model: VTONModel = VTONModel.OOTDIFFUSION,
    device: str = "cuda",
    **kwargs,
) -> TryOnResult:
    """Run virtual try-on inference locally.

    Args:
        person_image_path: Path to person image
        cloth_image_path: Path to cloth/garment image
        cloth_type: Clothing type (model-specific, see TryOnAgent.generate)
        output_path: Where to save the result (auto-generated if None)
        model: VTONModel.CATVTON or VTONModel.OOTDIFFUSION
        device: "cuda" or "cpu"
        **kwargs: Extra model parameters (num_inference_steps, guidance_scale, seed)

    Returns:
        TryOnResult
    """
    agent = TryOnAgent(model=model, device=device)
    result = await agent.generate(
        person_image=person_image_path,
        cloth_image=cloth_image_path,
        cloth_type=cloth_type,
        output_path=output_path,
        **kwargs,
    )
    agent.unload()
    return result


# ---------------------------------------------------------------------------
# Synchronous wrapper
# ---------------------------------------------------------------------------

def run_tryon_agent_sync(
    person_image_path: str,
    cloth_image_path: str,
    cloth_type: str = "upper",
    output_path: Optional[str] = None,
    model: VTONModel = VTONModel.OOTDIFFUSION,
    device: str = "cuda",
    **kwargs,
) -> TryOnResult:
    """Synchronous wrapper around run_tryon_agent.

    Args:
        person_image_path: Path to person image
        cloth_image_path: Path to cloth/garment image
        cloth_type: Clothing type (model-specific, see TryOnAgent.generate)
        output_path: Where to save the result (auto-generated if None)
        model: VTONModel.CATVTON or VTONModel.OOTDIFFUSION
        device: "cuda" or "cpu"
        **kwargs: Extra model parameters

    Returns:
        TryOnResult
    """
    return asyncio.run(
        run_tryon_agent(
            person_image_path=person_image_path,
            cloth_image_path=cloth_image_path,
            cloth_type=cloth_type,
            output_path=output_path,
            model=model,
            device=device,
            **kwargs,
        )
    )
