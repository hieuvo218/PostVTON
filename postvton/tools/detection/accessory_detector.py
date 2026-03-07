"""Accessory detection tool using YOLOE segmentation.

Detects wearable accessories (bracelet, watch, etc.) in an image and returns
per-instance bounding boxes, segmentation masks, labels and confidences.

Usage::

    detector = AccessoryDetector()
    result = detector.detect("person_original.jpg")
    if result.found:
        for acc in result.accessories:
            print(acc.label, acc.bbox, acc.confidence)

    # One-shot functional API
    result = detect_accessories("person_original.jpg")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DetectedAccessory:
    """One detected accessory instance."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2) in image coords
    mask: np.ndarray                    # uint8 binary mask, shape (H, W)

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (
            f"DetectedAccessory(label={self.label!r}, conf={self.confidence:.2f}, "
            f"bbox=({x1},{y1},{x2},{y2}))"
        )

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "mask_shape": list(self.mask.shape),
        }


@dataclass
class AccessoryDetectionResult:
    """Detection result for a single image."""
    image_path: str
    image_size: Tuple[int, int]        # (width, height)
    accessories: List[DetectedAccessory] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def found(self) -> bool:
        return len(self.accessories) > 0

    @property
    def labels(self) -> List[str]:
        return [a.label for a in self.accessories]

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "found": self.found,
            "count": len(self.accessories),
            "accessories": [a.to_dict() for a in self.accessories],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AccessoryDetector:
    """Detect accessories in images using YOLOE instance segmentation.

    Args:
        model_path:   Path to the YOLOE weights file (default: yoloe-11l-seg.pt).
        class_names:  List of accessory class names to detect.
                      Defaults to ["bracelet", "watch"].
        conf:         Minimum confidence threshold (0-1).
    """

    DEFAULT_CLASSES: List[str] = ["bracelet", "watch"]

    def __init__(
        self,
        model_path: str = "yoloe-11l-seg.pt",
        class_names: Optional[List[str]] = None,
        conf: float = 0.25,
    ):
        self.model_path = model_path
        self.class_names = list(class_names or self.DEFAULT_CLASSES)
        self.conf = conf
        self._model = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> AccessoryDetectionResult:
        """Detect accessories in an image.

        Args:
            image: File path (str/Path) or BGR numpy array.

        Returns:
            AccessoryDetectionResult with detected accessories and masks.
        """
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img = cv2.imread(image_path)
            if img is None:
                return AccessoryDetectionResult(
                    image_path=image_path,
                    image_size=(0, 0),
                    error=f"Could not read image: {image_path}",
                )
        else:
            image_path = "<array>"
            img = image

        h, w = img.shape[:2]

        try:
            model = self._get_model()
            results = model.predict(img, conf=self.conf, verbose=False)[0]
        except Exception as exc:
            return AccessoryDetectionResult(
                image_path=image_path,
                image_size=(w, h),
                error=f"YOLOE prediction failed: {exc}",
            )

        accessories = self._parse_results(results, img_shape=(h, w))

        return AccessoryDetectionResult(
            image_path=image_path,
            image_size=(w, h),
            accessories=accessories,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLOE  # lazy import — avoids ultralytics load at module level
            model = YOLOE(self.model_path)
            model.set_classes(self.class_names, model.get_text_pe(self.class_names))
            self._model = model
        return self._model

    def _parse_results(self, results, img_shape: Tuple[int, int]) -> List[DetectedAccessory]:
        """Convert raw YOLOE results into DetectedAccessory instances."""
        h, w = img_shape
        accessories: List[DetectedAccessory] = []

        if results.masks is None or results.boxes is None:
            return accessories

        class_names = results.names  # dict {id: name}

        for i, raw_mask in enumerate(results.masks.data):
            # ---- mask: resize from model output resolution to image size ----
            mask_np = raw_mask.cpu().numpy()
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_np > 0.5).astype(np.uint8) * 255

            # ---- bbox: clamp to image boundaries ----
            x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # ---- label & confidence ----
            cls_id = int(results.boxes.cls[i].cpu().numpy())
            label = class_names.get(cls_id, str(cls_id))
            conf = float(results.boxes.conf[i].cpu().numpy())

            accessories.append(DetectedAccessory(
                label=label,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                mask=mask_bin,
            ))

        return accessories


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_accessories(
    image: Union[str, Path, np.ndarray],
    model_path: str = "yoloe-11l-seg.pt",
    class_names: Optional[List[str]] = None,
    conf: float = 0.25,
) -> AccessoryDetectionResult:
    """One-shot accessory detection.

    Args:
        image:       File path or BGR numpy array.
        model_path:  Path to YOLOE weights.
        class_names: Accessory classes to detect.
        conf:        Confidence threshold.

    Returns:
        AccessoryDetectionResult.
    """
    detector = AccessoryDetector(
        model_path=model_path,
        class_names=class_names,
        conf=conf,
    )
    return detector.detect(image)