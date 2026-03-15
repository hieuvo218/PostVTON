"""Accessory restoration tool for virtual try-on post-processing.

After a try-on model removes accessories (bracelets, watches, etc.) from the
output, this tool detects them in the original source image and composites them
back onto the try-on result using precise segmentation masks.

Pipeline::

    source image ──► MissingAccessoryDetector.detect_accessories() ──► detected masks + bboxes ──┐
    target image (try-on result) ──────────────────────────────────────► composite ──► restored image

Usage::

    restorer = AccessoryRestorer()
    result = restorer.restore(
        source_image="person_original.jpg",
        target_image="tryon_output.png",
        output_path="restored_output.png",
    )
    if result.success:
        print(f"Restored {result.restored_count} accessory/ies → {result.output_path}")

    # One-shot functional API
    result = restore_accessories("original.jpg", "tryon.png", "out.png")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

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
    output_path: Optional[str]
    restored_count: int
    labels_restored: List[str] = field(default_factory=list)
    detection_result: Optional[AccessoryDetectionResult] = None
    error: Optional[str] = None

    def __str__(self) -> str:
        if not self.success:
            return f"AccessoryRestorationResult(failed: {self.error})"
        return (
            f"AccessoryRestorationResult(restored={self.restored_count}, "
            f"labels={self.labels_restored}, output={self.output_path})"
        )

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_path": self.output_path,
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

    Detection is performed on the *source* image (the original photo that has
    accessories). The detected accessories are then composited onto the *target*
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
        source_image: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
    ) -> AccessoryRestorationResult:
        """Detect accessories in source and paste them onto target.

        Args:
            source_image: Original image containing accessories (path or BGR array).
            target_image: Try-on result image to restore accessories onto
                          (path or BGR array).
            output_path:  Where to save the restored image. If None, the image
                          is only returned in the result (not saved).

        Returns:
            AccessoryRestorationResult.
        """
        # ---- load images ----
        src, src_path = self._load(source_image, "source")
        if src is None:
            return AccessoryRestorationResult(
                success=False,
                output_path=None,
                restored_count=0,
                error=f"Could not load source image: {src_path}",
            )

        dst, dst_path = self._load(target_image, "target")
        if dst is None:
            return AccessoryRestorationResult(
                success=False,
                output_path=None,
                restored_count=0,
                error=f"Could not load target image: {dst_path}",
            )

        # ---- detect accessories in source ----
        detection = self.detector.detect_accessories(src)
        if detection.error:
            return AccessoryRestorationResult(
                success=False,
                output_path=None,
                restored_count=0,
                detection_result=detection,
                error=f"Detection failed: {detection.error}",
            )

        if not detection.found:
            print("[AccessoryRestorer] No accessories detected in source image.")
            # Still save dst unchanged if output_path given
            saved_path = self._save(dst, output_path)
            return AccessoryRestorationResult(
                success=True,
                output_path=saved_path,
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

        # ---- save ----
        saved_path = self._save(result_img, output_path)

        return AccessoryRestorationResult(
            success=True,
            output_path=saved_path,
            restored_count=len(labels_restored),
            labels_restored=labels_restored,
            detection_result=detection,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(image: Union[str, Path, np.ndarray], name: str) -> Tuple[Optional[np.ndarray], str]:
        """Load BGR array; returns (array, path_string) or (None, path_string) on error."""
        if isinstance(image, np.ndarray):
            return image, f"<{name}_array>"
        path = str(image)
        img = cv2.imread(path)
        return img, path

    @staticmethod
    def _save(image: np.ndarray, output_path: Optional[Union[str, Path]]) -> Optional[str]:
        if output_path is None:
            return None
        out = str(output_path)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out, image)
        return out

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

        # Extract masked accessory pixels from (already-resized) source
        obj = cv2.bitwise_and(src, src, mask=acc.mask)

        obj_crop = obj[y1:y2, x1:x2]
        mask_crop = acc.mask[y1:y2, x1:x2]
        roi = dst[y1:y2, x1:x2]

        # Build 3-channel binary mask for alpha blending
        mask_3ch = cv2.merge([mask_crop, mask_crop, mask_crop])
        mask_bin = (mask_3ch > 0).astype(np.uint8)

        # Ensure shapes match (float safety)
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
    source_image: Union[str, Path, np.ndarray],
    target_image: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    model_path: str = "yoloe-11l-seg.pt",
    class_names: Optional[List[str]] = None,
    conf: float = 0.25,
) -> AccessoryRestorationResult:
    """One-shot accessory restoration.

    Detects accessories in source_image and pastes them onto target_image.

    Args:
        source_image: Original image with accessories (path or BGR array).
        target_image: Try-on result image (path or BGR array).
        output_path:  Save path for restored image (optional).
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
    return restorer.restore(source_image, target_image, output_path)
