"""Accessory restoration tool for virtual try-on post-processing.

After a try-on model removes accessories (bracelets, watches, etc.) from the
output, this tool detects them in the original source image and composites them
back onto the try-on result using precise segmentation masks.

Pipeline::

    source image (PIL) ──► MissingAccessoryDetector.detect_accessories() ──► masks + bboxes ──┐
    target image (PIL) ──────────────────────────────────────────────────────► composite ──► restored image

Usage::

    restorer = AccessoryRestorer()
    result = restorer.restore(source_image=person_pil, target_image=tryon_pil)
    if result.success:
        print(f"Restored {result.restored_count} accessory/ies")

    # One-shot functional API
    result = restore_accessories(person_pil, tryon_pil)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from postvton.tools.detection.missing_accessory_detector import (
    MissingAccessoryDetector,
    AccessoryDetectionResult,
    DetectedAccessory,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AccessoryRestorationResult:
    """Result of an accessory restoration operation."""

    success: bool
    output_image: Optional[Image.Image]
    restored_count: int
    labels_restored: List[str] = field(default_factory=list)
    detection_result: Optional[AccessoryDetectionResult] = None
    error: Optional[str] = None

    def __str__(self) -> str:
        if not self.success:
            return f"AccessoryRestorationResult(failed: {self.error})"
        return (
            f"AccessoryRestorationResult(restored={self.restored_count}, "
            f"labels={self.labels_restored})"
        )

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_image": self.output_image is not None,
            "restored_count": self.restored_count,
            "labels_restored": self.labels_restored,
            "detection": self.detection_result.to_dict() if self.detection_result else None,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Restorer
# ---------------------------------------------------------------------------

class AccessoryRestorer:
    """Restore accessories from a source image onto a target (try-on) image.

    Detection is performed on the source image (the original photo that has
    accessories). The detected accessories are then composited onto the target
    image (the try-on output that may be missing them).

    Args:
        detector:     Pre-built MissingAccessoryDetector. If None, a default one is created.
        model_path:   YOLOE weights path (used only when detector is None).
        class_names:  Accessory classes to detect (used only when detector is None).
        conf:         Detection confidence threshold.
    """

    def __init__(
        self,
        detector: Optional[MissingAccessoryDetector] = None,
        model_path: str = "yoloe-11l-seg.pt",
        class_names: Optional[List[str]] = None,
        conf: float = 0.25,
    ):
        self.detector = detector or MissingAccessoryDetector(
            model_path=model_path,
            class_names=class_names,
            conf=conf,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def restore(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
    ) -> AccessoryRestorationResult:
        """Detect accessories in source and paste them onto target.

        Args:
            source_image: Original image containing accessories (PIL).
            target_image: Try-on result image to restore accessories onto (PIL).

        Returns:
            AccessoryRestorationResult.
        """
        if not isinstance(source_image, Image.Image):
            return AccessoryRestorationResult(
                success=False,
                output_image=None,
                restored_count=0,
                error=f"Expected PIL.Image.Image for source_image, got {type(source_image).__name__}",
            )

        if not isinstance(target_image, Image.Image):
            return AccessoryRestorationResult(
                success=False,
                output_image=None,
                restored_count=0,
                error=f"Expected PIL.Image.Image for target_image, got {type(target_image).__name__}",
            )

        src = self._pil_to_bgr(source_image)
        dst = self._pil_to_bgr(target_image)

        # ---- detect accessories in source ----
        detection = self.detector.detect_accessories(source_image)
        if detection.error:
            return AccessoryRestorationResult(
                success=False,
                output_image=None,
                restored_count=0,
                detection_result=detection,
                error=f"Detection failed: {detection.error}",
            )

        if not detection.found:
            print("[AccessoryRestorer] No accessories detected in source image.")
            return AccessoryRestorationResult(
                success=True,
                output_image=target_image.copy(),
                restored_count=0,
                detection_result=detection,
            )

        # ---- resize source to match target dimensions ----
        tgt_h, tgt_w = dst.shape[:2]
        src_h, src_w = src.shape[:2]

        if (src_w, src_h) != (tgt_w, tgt_h):
            scale_x = tgt_w / src_w
            scale_y = tgt_h / src_h
            src_resized = cv2.resize(src, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            accessories = self._rescale_accessories(detection.accessories, scale_x, scale_y, (tgt_h, tgt_w))
        else:
            src_resized = src
            accessories = detection.accessories

        # ---- composite each accessory onto dst ----
        result_img = dst.copy()
        labels_restored: List[str] = []

        for acc in accessories:
            result_img = self._paste_accessory(src_resized, result_img, acc)
            labels_restored.append(acc.label)
            print(f"[AccessoryRestorer] Pasted '{acc.label}' (conf={acc.confidence:.2f}) onto target.")

        result_pil = self._bgr_to_pil(result_img)
        return AccessoryRestorationResult(
            success=True,
            output_image=result_pil,
            restored_count=len(labels_restored),
            labels_restored=labels_restored,
            detection_result=detection,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_bgr(image: Image.Image) -> np.ndarray:
        array = np.array(image.convert("RGB"))
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_pil(image: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _rescale_accessories(
        accessories: List[DetectedAccessory],
        scale_x: float,
        scale_y: float,
        new_shape: Tuple[int, int],   # (H, W)
    ) -> List[DetectedAccessory]:
        """Scale bboxes and masks to new image dimensions."""
        new_h, new_w = new_shape
        rescaled = []
        for acc in accessories:
            x1, y1, x2, y2 = acc.bbox
            nx1 = max(0, int(x1 * scale_x))
            ny1 = max(0, int(y1 * scale_y))
            nx2 = min(new_w, int(x2 * scale_x))
            ny2 = min(new_h, int(y2 * scale_y))

            new_mask = cv2.resize(acc.mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            new_mask = (new_mask > 127).astype(np.uint8) * 255

            rescaled.append(DetectedAccessory(
                label=acc.label,
                confidence=acc.confidence,
                bbox=(nx1, ny1, nx2, ny2),
                mask=new_mask,
            ))
        return rescaled

    @staticmethod
    def _paste_accessory(
        src: np.ndarray,
        dst: np.ndarray,
        acc: DetectedAccessory,
    ) -> np.ndarray:
        """Composite one accessory crop from src onto dst using its mask."""
        x1, y1, x2, y2 = acc.bbox
        if x2 <= x1 or y2 <= y1:
            return dst

        obj = cv2.bitwise_and(src, src, mask=acc.mask)

        obj_crop = obj[y1:y2, x1:x2]
        mask_crop = acc.mask[y1:y2, x1:x2]
        roi = dst[y1:y2, x1:x2]

        mask_3ch = cv2.merge([mask_crop, mask_crop, mask_crop])
        mask_bin = (mask_3ch > 0).astype(np.uint8)

        if roi.shape != obj_crop.shape:
            obj_crop = cv2.resize(obj_crop, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_bin = cv2.resize(mask_bin, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

        blended = roi * (1 - mask_bin) + obj_crop * mask_bin
        dst = dst.copy()
        dst[y1:y2, x1:x2] = blended.astype(np.uint8)
        return dst


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def restore_accessories(
    source_image: Image.Image,
    target_image: Image.Image,
    model_path: str = "yoloe-11l-seg.pt",
    class_names: Optional[List[str]] = None,
    conf: float = 0.25,
) -> AccessoryRestorationResult:
    """One-shot accessory restoration.

    Detects accessories in source_image and pastes them onto target_image.

    Args:
        source_image: Original image with accessories (PIL).
        target_image: Try-on result image (PIL).
        model_path:   YOLOE weights path.
        class_names:  Accessory classes to detect.
        conf:         Confidence threshold.

    Returns:
        AccessoryRestorationResult.
    """
    restorer = AccessoryRestorer(
        model_path=model_path,
        class_names=class_names,
        conf=conf,
    )
    return restorer.restore(source_image, target_image)
