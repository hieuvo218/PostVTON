from dataclasses import dataclass
from typing import Any, Optional
import logging

import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from diffusers import QwenImageEditPipeline
except ImportError as exc:
    QwenImageEditPipeline = None
    _QWEN_IMPORT_ERROR = exc
else:
    _QWEN_IMPORT_ERROR = None


DEFAULT_PROMPT = (
    "Edit ONLY the malformed hand(s) area to become normal and natural. "
    "Everything else besides the area of the hand(s) should remain unchanged."
)


@dataclass
class HandRefinementResult:
    """Result of a hand refinement operation."""

    success: bool
    output_image: Optional[Image.Image]
    image_size: Optional[tuple[int, int]] = None
    error: Optional[str] = None
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_image": self.output_image is not None,
            "image_size": list(self.image_size) if self.image_size else None,
            "error": self.error,
            "detail": dict(self.detail or {}),
        }


class HandRefiner:
    """Refine malformed hand regions in a generated image.

    The pipeline is loaded lazily on first use to avoid expensive import-time
    initialization.
    """

    def __init__(
        self,
        model_path: str = "ovedrive/qwen-image-edit-4bit",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        disable_progress_bar: bool = False,
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        elif self.device == "cuda":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
        self.disable_progress_bar = disable_progress_bar
        self._pipeline: Optional["QwenImageEditPipeline"] = None

    def refine(
        self,
        image: Image.Image,
        num_inference_steps: int = 15,
        true_cfg_scale: float = 3.0,
        seed: Optional[int] = 0,
        edit_prompt: Optional[str] = None,
        **kwargs,
    ) -> HandRefinementResult:
        """Run hand refinement using Qwen image editor on the full image.

        Args:
            image: Input PIL image.
            num_inference_steps: Diffusion inference steps.
            true_cfg_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility. If None, stochastic run.
            edit_prompt: Optional custom prompt from hand detection. Uses DEFAULT_PROMPT if None.
            **kwargs: Ignored; for backward compatibility with old callers.
        """
        try:
            if not isinstance(image, Image.Image):
                return HandRefinementResult(
                    success=False,
                    output_image=None,
                    error=f"Expected PIL.Image.Image, got {type(image).__name__}",
                )

            input_img = image.convert("RGB")
            width, height = input_img.size
            prompt = edit_prompt or DEFAULT_PROMPT

            refined = self._run_qwen_edit(
                image=input_img,
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                seed=seed,
            ).resize((width, height), Image.LANCZOS)

            return HandRefinementResult(
                success=True,
                output_image=refined,
                image_size=(width, height),
                detail={"mode": "qwen_full_image"},
            )
        except Exception as exc:
            logger.exception("Hand refinement failed")
            return HandRefinementResult(
                success=False,
                output_image=None,
                error=str(exc),
            )

    def _run_qwen_edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        true_cfg_scale: float,
        seed: Optional[int],
    ) -> Image.Image:
        width, height = image.size
        pipeline = self._get_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        payload = {
            "image": image.convert("RGB"),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "generator": generator,
        }

        with torch.inference_mode():
            output = pipeline(**payload)

        return output.images[0].resize((width, height), Image.LANCZOS)

    def _get_pipeline(self) -> "QwenImageEditPipeline":
        if QwenImageEditPipeline is None:
            raise ImportError(
                "QwenImageEditPipeline is unavailable. Install Qwen-capable "
                "Diffusers dependencies with: pip install -U -r requirements.system.txt"
            ) from _QWEN_IMPORT_ERROR

        if self._pipeline is None:
            logger.info("Loading QwenImageEditPipeline from %s", self.model_path)
            pipe = QwenImageEditPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
            )
            pipe.set_progress_bar_config(disable=self.disable_progress_bar)
            pipe.to(self.device)
            self._pipeline = pipe
        return self._pipeline


def refine_hands(
    image: Image.Image,
    model_path: str = "ovedrive/qwen-image-edit-4bit",
    device: Optional[str] = None,
    num_inference_steps: int = 15,
    true_cfg_scale: float = 3.0,
    seed: Optional[int] = 0,
    edit_prompt: Optional[str] = None,
    **kwargs,
) -> HandRefinementResult:
    """One-shot hand refinement API using Qwen image editor on full image.
    
    Args:
        image: Input PIL image.
        model_path: Path to Qwen model.
        device: Device to use (cuda/cpu).
        num_inference_steps: Diffusion steps.
        true_cfg_scale: Classifier-free guidance scale.
        seed: Random seed.
        edit_prompt: Optional custom prompt from hand detection.
        **kwargs: Additional arguments for backward compatibility.
    """
    refiner = HandRefiner(model_path=model_path, device=device)
    return refiner.refine(
        image=image,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        seed=seed,
        edit_prompt=edit_prompt,
        **kwargs,
    )

