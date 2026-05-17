from dataclasses import dataclass
from typing import Any, Optional
import logging

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from skimage.exposure import match_histograms

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
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = "",
        num_inference_steps: int = 15,
        true_cfg_scale: float = 3.0,
        seed: Optional[int] = 0,
        apply_histogram_matching: bool = True,
        localize_edit: bool = True,
        fallback_to_global_edit: bool = False,
        hand_padding_ratio: float = 0.45,
        min_crop_size: int = 320,
        composite_blur_radius: int = 18,
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
            localize_edit: Edit detected hand crops and composite them back.
            fallback_to_global_edit: If no hand crop is found, allow full-image edit.
                Keep this False to avoid global quality degradation.
            hand_padding_ratio: Padding added around detected hand boxes.
            min_crop_size: Minimum square crop size sent to the edit model.
            composite_blur_radius: Soft edge radius for crop compositing.
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
            if localize_edit:
                hand_boxes, detector_source = self._detect_hand_boxes(input_img)
                if hand_boxes:
                    refined = input_img.copy()
                    crop_boxes = []
                    for box in hand_boxes:
                        crop_box = self._expand_box(
                            box=box,
                            image_size=(width, height),
                            padding_ratio=hand_padding_ratio,
                            min_size=min_crop_size,
                        )
                        crop_boxes.append(crop_box)
                        crop = input_img.crop(crop_box)
                        edited_crop = self._run_qwen_edit(
                            image=crop,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            true_cfg_scale=true_cfg_scale,
                            seed=seed,
                        )
                        if apply_histogram_matching:
                            edited_crop = self._match_histograms(edited_crop, crop)
                        refined = self._paste_soft_crop(
                            base=refined,
                            edited_crop=edited_crop,
                            crop_box=crop_box,
                            blur_radius=composite_blur_radius,
                        )

                    return HandRefinementResult(
                        success=True,
                        output_image=refined,
                        image_size=(width, height),
                        detail={
                            "mode": "localized",
                            "detector": detector_source,
                            "hand_boxes": [list(box) for box in hand_boxes],
                            "crop_boxes": [list(box) for box in crop_boxes],
                            "crop_count": len(crop_boxes),
                        },
                    )

                if not fallback_to_global_edit:
                    logger.warning("No hand crop detected; returning original image without global edit")
                    return HandRefinementResult(
                        success=True,
                        output_image=input_img.copy(),
                        image_size=(width, height),
                        error="No hand crop detected; skipped global hand refinement to preserve image quality.",
                        detail={
                            "mode": "skipped",
                            "reason": "no_hand_crop_detected",
                            "detector": detector_source,
                            "crop_count": 0,
                        },
                    )

            refined = self._run_qwen_edit(
                image=input_img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                seed=seed,
            ).resize((width, height), Image.LANCZOS)

            if apply_histogram_matching:
                refined = self._match_histograms(refined, input_img)

            return HandRefinementResult(
                success=True,
                output_image=refined,
                image_size=(width, height),
                detail={
                    "mode": "global",
                    "crop_count": 0,
                },
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

    @staticmethod
    def _detect_hand_boxes(image: Image.Image) -> tuple[list[tuple[int, int, int, int]], str]:
        boxes = HandRefiner._detect_mediapipe_hand_boxes(image)
        if boxes:
            return boxes, "mediapipe_hands"
        boxes = HandRefiner._detect_pose_wrist_boxes(image)
        if boxes:
            return boxes, "mediapipe_pose"
        return [], "none"

    @staticmethod
    def _detect_mediapipe_hand_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
        try:
            import mediapipe as mp
        except Exception:
            return []

        arr = np.array(image.convert("RGB"))
        height, width = arr.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []

        try:
            with mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.25,
            ) as hands:
                results = hands.process(arr)
        except Exception:
            return []

        for landmarks in getattr(results, "multi_hand_landmarks", None) or []:
            xs = [lm.x * width for lm in landmarks.landmark]
            ys = [lm.y * height for lm in landmarks.landmark]
            boxes.append((
                max(0, int(min(xs))),
                max(0, int(min(ys))),
                min(width, int(max(xs))),
                min(height, int(max(ys))),
            ))
        return HandRefiner._merge_overlapping_boxes(boxes)

    @staticmethod
    def _detect_pose_wrist_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
        try:
            import mediapipe as mp
        except Exception:
            return []

        arr = np.array(image.convert("RGB"))
        height, width = arr.shape[:2]

        try:
            with mp.solutions.pose.Pose(static_image_mode=True) as pose:
                results = pose.process(arr)
        except Exception:
            return []

        landmarks = getattr(getattr(results, "pose_landmarks", None), "landmark", None)
        if not landmarks:
            return []

        pose_landmark = mp.solutions.pose.PoseLandmark
        wrist_elbow_pairs = [
            (pose_landmark.LEFT_WRIST, pose_landmark.LEFT_ELBOW),
            (pose_landmark.RIGHT_WRIST, pose_landmark.RIGHT_ELBOW),
        ]
        boxes: list[tuple[int, int, int, int]] = []
        for wrist_idx, elbow_idx in wrist_elbow_pairs:
            wrist = landmarks[int(wrist_idx)]
            elbow = landmarks[int(elbow_idx)]
            if wrist.visibility < 0.05:
                continue
            wx, wy = wrist.x * width, wrist.y * height
            ex, ey = elbow.x * width, elbow.y * height
            arm_len = max(((wx - ex) ** 2 + (wy - ey) ** 2) ** 0.5, min(width, height) * 0.08)
            radius = max(arm_len * 0.9, min(width, height) * 0.10)
            boxes.append((
                max(0, int(wx - radius)),
                max(0, int(wy - radius)),
                min(width, int(wx + radius)),
                min(height, int(wy + radius)),
            ))
        if boxes:
            return HandRefiner._merge_overlapping_boxes(boxes)

        return HandRefiner._estimate_lower_arm_hand_boxes(landmarks, width, height, pose_landmark)

    @staticmethod
    def _estimate_lower_arm_hand_boxes(landmarks, width: int, height: int, pose_landmark) -> list[tuple[int, int, int, int]]:
        wrist_elbow_pairs = [
            (pose_landmark.LEFT_WRIST, pose_landmark.LEFT_ELBOW),
            (pose_landmark.RIGHT_WRIST, pose_landmark.RIGHT_ELBOW),
        ]
        boxes: list[tuple[int, int, int, int]] = []
        for wrist_idx, elbow_idx in wrist_elbow_pairs:
            wrist = landmarks[int(wrist_idx)]
            elbow = landmarks[int(elbow_idx)]
            elbow_visible = elbow.visibility >= 0.05
            wrist_in_frame = 0.0 <= wrist.x <= 1.0 and 0.0 <= wrist.y <= 1.0

            if wrist.visibility >= 0.01 and wrist_in_frame:
                cx, cy = wrist.x * width, wrist.y * height
                if elbow_visible:
                    ex, ey = elbow.x * width, elbow.y * height
                    radius = max(((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5 * 1.0, min(width, height) * 0.12)
                else:
                    radius = min(width, height) * 0.16
            elif elbow_visible:
                ex, ey = elbow.x * width, elbow.y * height
                # If wrist is not usable, estimate the hand below the elbow.
                cx, cy = ex, ey + min(width, height) * 0.16
                radius = min(width, height) * 0.18
            else:
                continue

            boxes.append((
                max(0, int(cx - radius)),
                max(0, int(cy - radius)),
                min(width, int(cx + radius)),
                min(height, int(cy + radius)),
            ))
        return HandRefiner._merge_overlapping_boxes(boxes)

    @staticmethod
    def _expand_box(
        box: tuple[int, int, int, int],
        image_size: tuple[int, int],
        padding_ratio: float,
        min_size: int,
    ) -> tuple[int, int, int, int]:
        width, height = image_size
        x1, y1, x2, y2 = box
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        side = max(box_w, box_h, min_size)
        side = int(side * (1.0 + max(0.0, padding_ratio)))
        side = min(side, width, height)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        nx1 = int(round(cx - side / 2))
        ny1 = int(round(cy - side / 2))
        nx2 = nx1 + side
        ny2 = ny1 + side

        if nx1 < 0:
            nx2 -= nx1
            nx1 = 0
        if ny1 < 0:
            ny2 -= ny1
            ny1 = 0
        if nx2 > width:
            nx1 -= nx2 - width
            nx2 = width
        if ny2 > height:
            ny1 -= ny2 - height
            ny2 = height

        return max(0, nx1), max(0, ny1), min(width, nx2), min(height, ny2)

    @staticmethod
    def _paste_soft_crop(
        base: Image.Image,
        edited_crop: Image.Image,
        crop_box: tuple[int, int, int, int],
        blur_radius: int,
    ) -> Image.Image:
        x1, y1, x2, y2 = crop_box
        crop_w, crop_h = x2 - x1, y2 - y1
        edited_crop = edited_crop.resize((crop_w, crop_h), Image.LANCZOS)

        mask = Image.new("L", (crop_w, crop_h), 0)
        draw = ImageDraw.Draw(mask)
        inset = max(1, int(max(crop_w, crop_h) * 0.08))
        draw.rounded_rectangle(
            (inset, inset, crop_w - inset, crop_h - inset),
            radius=max(1, inset),
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(0, blur_radius)))

        result = base.copy()
        result.paste(edited_crop, (x1, y1), mask)
        return result

    @staticmethod
    def _merge_overlapping_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        merged: list[tuple[int, int, int, int]] = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            did_merge = False
            for idx, other in enumerate(merged):
                ox1, oy1, ox2, oy2 = other
                intersects = x1 <= ox2 and x2 >= ox1 and y1 <= oy2 and y2 >= oy1
                if intersects:
                    merged[idx] = (min(x1, ox1), min(y1, oy1), max(x2, ox2), max(y2, oy2))
                    did_merge = True
                    break
            if not did_merge:
                merged.append(box)
        return merged

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
    localize_edit: bool = True,
    fallback_to_global_edit: bool = False,
    hand_padding_ratio: float = 0.45,
    min_crop_size: int = 320,
    composite_blur_radius: int = 18,
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
        localize_edit=localize_edit,
        fallback_to_global_edit=fallback_to_global_edit,
        hand_padding_ratio=hand_padding_ratio,
        min_crop_size=min_crop_size,
        composite_blur_radius=composite_blur_radius,
    )
