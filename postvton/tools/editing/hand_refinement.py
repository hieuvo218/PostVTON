from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import argparse
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
    output_path: Optional[str]
    image_size: Optional[tuple[int, int]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_path": self.output_path,
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
        image: Union[str, Path, Image.Image],
        output_path: Union[str, Path],
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = "",
        num_inference_steps: int = 15,
        true_cfg_scale: float = 3.0,
        seed: Optional[int] = 0,
        apply_histogram_matching: bool = True,
    ) -> HandRefinementResult:
        """Run hand refinement and save the result.

        Args:
            image: Input image path or PIL image.
            output_path: Where to save the refined output.
            prompt: Editing prompt passed to Qwen image editor.
            negative_prompt: Negative prompt for guidance.
            num_inference_steps: Diffusion inference steps.
            true_cfg_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility. If None, stochastic run.
            apply_histogram_matching: Match output color histogram to input.
        """
        try:
            input_img = self._load_image(image)
            if input_img is None:
                return HandRefinementResult(
                    success=False,
                    output_path=None,
                    error=f"Could not load image: {image}",
                )

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

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            refined.save(out)

            return HandRefinementResult(
                success=True,
                output_path=str(out),
                image_size=(width, height),
            )
        except Exception as exc:
            logger.exception("Hand refinement failed")
            return HandRefinementResult(
                success=False,
                output_path=None,
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
    def _load_image(image: Union[str, Path, Image.Image]) -> Optional[Image.Image]:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        path = Path(image)
        if not path.exists():
            return None
        return Image.open(path).convert("RGB")

    @staticmethod
    def _match_histograms(output_image: Image.Image, reference_image: Image.Image) -> Image.Image:
        out_np = np.array(output_image)
        ref_np = np.array(reference_image)
        matched = match_histograms(out_np, ref_np, channel_axis=-1)
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        return Image.fromarray(matched)


def refine_hands(
    image: Union[str, Path, Image.Image],
    output_path: Union[str, Path],
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
        output_path=output_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        seed=seed,
        apply_histogram_matching=apply_histogram_matching,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine malformed hands in an image")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--model-path", default="ovedrive/qwen-image-edit-4bit")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", None])
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--no-hist-match", action="store_true")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cli = _build_cli_parser().parse_args()

    result = refine_hands(
        image=cli.input,
        output_path=cli.output,
        model_path=cli.model_path,
        device=cli.device,
        prompt=cli.prompt,
        negative_prompt=cli.negative_prompt,
        num_inference_steps=cli.steps,
        true_cfg_scale=cli.cfg,
        seed=cli.seed,
        apply_histogram_matching=not cli.no_hist_match,
    )

    if result.success:
        print(f"Saved refined image to: {result.output_path}")
    else:
        print(f"Refinement failed: {result.error}")