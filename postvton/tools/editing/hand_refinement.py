from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from skimage.exposure import match_histograms

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_image": self.output_image is not None,
            "image_size": list(self.image_size) if self.image_size else None,
            "error": self.error,
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
        self._pipeline: Optional[QwenImageEditPipeline] = None

    def refine(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = "",
        num_inference_steps: int = 15,
        true_cfg_scale: float = 3.0,
        seed: Optional[int] = 0,
        apply_histogram_matching: bool = True,
    ) -> HandRefinementResult:
        """Run hand refinement and return a refined PIL image.

        Args:
            image: Input PIL image.
            prompt: Editing prompt passed to Qwen image editor.
            negative_prompt: Negative prompt for guidance.
            num_inference_steps: Diffusion inference steps.
            true_cfg_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility. If None, stochastic run.
            apply_histogram_matching: Match output color histogram to input.
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
            pipeline = self._get_pipeline()

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            payload = {
                "image": input_img,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "true_cfg_scale": true_cfg_scale,
                "generator": generator,
            }

            with torch.inference_mode():
                output = pipeline(**payload)

            refined = output.images[0].resize((width, height), Image.LANCZOS)

            if apply_histogram_matching:
                refined = self._match_histograms(refined, input_img)

            return HandRefinementResult(
                success=True,
                output_image=refined,
                image_size=(width, height),
            )
        except Exception as exc:
            logger.exception("Hand refinement failed")
            return HandRefinementResult(
                success=False,
                output_image=None,
                error=str(exc),
            )

    def _get_pipeline(self) -> QwenImageEditPipeline:
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

    @staticmethod
    def _match_histograms(output_image: Image.Image, reference_image: Image.Image) -> Image.Image:
        out_np = np.array(output_image)
        ref_np = np.array(reference_image)
        matched = match_histograms(out_np, ref_np, channel_axis=-1)
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        return Image.fromarray(matched)


def refine_hands(
    image: Image.Image,
    model_path: str = "ovedrive/qwen-image-edit-4bit",
    device: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = "",
    num_inference_steps: int = 15,
    true_cfg_scale: float = 3.0,
    seed: Optional[int] = 0,
    apply_histogram_matching: bool = True,
) -> HandRefinementResult:
    """One-shot hand refinement API."""
    refiner = HandRefiner(model_path=model_path, device=device)
    return refiner.refine(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        seed=seed,
        apply_histogram_matching=apply_histogram_matching,
    )