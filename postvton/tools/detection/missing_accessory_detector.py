"""Missing accessory detection tool.

Compares accessories detected in an original image versus a try-on result and
reports which accessories are missing in the try-on image.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class DetectedAccessory:
    """One accessory instance detected in an image."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray

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
    """Accessory detection output for one image."""
    image_path: str
    image_size: Tuple[int, int]
    accessories: List[DetectedAccessory] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def labels(self) -> List[str]:
        return [a.label for a in self.accessories]

    @property
    def found(self) -> bool:
        return bool(self.accessories)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "found": self.found,
            "count": len(self.accessories),
            "labels": self.labels,
            "accessories": [a.to_dict() for a in self.accessories],
            "error": self.error,
        }


@dataclass
class MissingAccessoryResult:
    """Comparison output for original vs try-on accessories."""
    original_detection: AccessoryDetectionResult
    tryon_detection: AccessoryDetectionResult
    missing_by_label: Dict[str, int] = field(default_factory=dict)

    @property
    def missing_labels(self) -> List[str]:
        return sorted(self.missing_by_label.keys())

    @property
    def total_missing(self) -> int:
        return int(sum(self.missing_by_label.values()))

    @property
    def has_missing(self) -> bool:
        return self.total_missing > 0

    @property
    def error(self) -> Optional[str]:
        if self.original_detection.error:
            return f"Original detection failed: {self.original_detection.error}"
        if self.tryon_detection.error:
            return f"Try-on detection failed: {self.tryon_detection.error}"
        return None

    def to_dict(self) -> dict:
        return {
            "has_missing": self.has_missing,
            "total_missing": self.total_missing,
            "missing_labels": self.missing_labels,
            "missing_by_label": dict(self.missing_by_label),
            "original_detection": self.original_detection.to_dict(),
            "tryon_detection": self.tryon_detection.to_dict(),
            "error": self.error,
        }


class MissingAccessoryDetector:
    """Detect and compare accessories between original and try-on images."""

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

    def detect_accessories(self, image: Union[str, Path, np.ndarray]) -> AccessoryDetectionResult:
        """Detect accessories in a single image."""
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

    def detect_missing(
        self,
        original_image: Union[str, Path, np.ndarray],
        tryon_image: Union[str, Path, np.ndarray],
    ) -> MissingAccessoryResult:
        """Compare original and try-on detections and report missing items.

        Label-count aware example:
            original: [watch, watch, bracelet]
            try-on:   [watch]
            missing_by_label -> {"watch": 1, "bracelet": 1}
        """
        original_result = self.detect_accessories(original_image)
        tryon_result = self.detect_accessories(tryon_image)

        if original_result.error or tryon_result.error:
            return MissingAccessoryResult(
                original_detection=original_result,
                tryon_detection=tryon_result,
                missing_by_label={},
            )

        original_counts = Counter(label.strip().lower() for label in original_result.labels)
        tryon_counts = Counter(label.strip().lower() for label in tryon_result.labels)

        missing_by_label: Dict[str, int] = {}
        for label, count in original_counts.items():
            missing = count - tryon_counts.get(label, 0)
            if missing > 0:
                missing_by_label[label] = int(missing)

        return MissingAccessoryResult(
            original_detection=original_result,
            tryon_detection=tryon_result,
            missing_by_label=missing_by_label,
        )

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLOE

            model = YOLOE(self.model_path)
            model.set_classes(self.class_names, model.get_text_pe(self.class_names))
            self._model = model
        return self._model

    def _parse_results(self, results, img_shape: Tuple[int, int]) -> List[DetectedAccessory]:
        h, w = img_shape
        accessories: List[DetectedAccessory] = []

        if results.masks is None or results.boxes is None:
            return accessories

        class_names = results.names

        for i, raw_mask in enumerate(results.masks.data):
            mask_np = raw_mask.cpu().numpy()
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_np > 0.5).astype(np.uint8) * 255

            x1, y1, x2, y2 = results.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cls_id = int(results.boxes.cls[i].cpu().numpy())
            label = class_names.get(cls_id, str(cls_id))
            conf = float(results.boxes.conf[i].cpu().numpy())

            accessories.append(
                DetectedAccessory(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    mask=mask_bin,
                )
            )

        return accessories


def detect_missing_accessories(
    original_image: Union[str, Path, np.ndarray],
    tryon_image: Union[str, Path, np.ndarray],
    model_path: str = "yoloe-11l-seg.pt",
    class_names: Optional[List[str]] = None,
    conf: float = 0.25,
) -> MissingAccessoryResult:
    """One-shot API for missing accessory detection."""
    detector = MissingAccessoryDetector(
        model_path=model_path,
        class_names=class_names,
        conf=conf,
    )
    return detector.detect_missing(original_image=original_image, tryon_image=tryon_image)
