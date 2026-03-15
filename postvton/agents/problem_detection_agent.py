"""Problem Detection Agent for virtual try-on quality assessment.

Detects visual problems in try-on output images using the available
detection tools. Runs both hand distortion and accessory detection.

Usage::

    agent = ProblemDetectionAgent(api_keys=["key1", "key2"])
    report = agent.detect("outputs/tryon_result.jpg")
    if report.has_problems:
        print(report.problems)
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from postvton.tools.detection.hand_detector import (
    HandDistortionDetector,
    HandDetectionResult,
)
from postvton.tools.detection.missing_accessory_detector import (
    MissingAccessoryDetector,
    MissingAccessoryResult,
    AccessoryDetectionResult as MissingAccessoryDetectionResult,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DetectedProblem:
    """A single detected problem in a try-on image."""
    problem_type: str          # e.g. "hand_distortion"
    severity: str              # "high" | "medium" | "low"
    description: str           # human-readable explanation
    detail: dict = field(default_factory=dict)  # raw detector output

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.problem_type}: {self.description}"


@dataclass
class ProblemDetectionReport:
    """Full detection report for one try-on image."""
    image_path: str
    has_problems: bool
    problems: List[DetectedProblem] = field(default_factory=list)
    accessory_detection: Optional[dict] = None
    error: Optional[str] = None

    # ---- helpers ----

    def summary(self) -> str:
        if self.error:
            return f"Detection failed for {self.image_path}: {self.error}"
        if not self.has_problems:
            return f"No problems detected in {self.image_path}"
        lines = [f"{len(self.problems)} problem(s) found in {self.image_path}:"]
        lines += [f"  • {p}" for p in self.problems]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "has_problems": self.has_problems,
            "problems": [
                {
                    "problem_type": p.problem_type,
                    "severity": p.severity,
                    "description": p.description,
                    "detail": p.detail,
                }
                for p in self.problems
            ],
            "accessory_detection": self.accessory_detection,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ProblemDetectionAgent:
    """Detect visual problems in virtual try-on images.

    Args:
        api_keys:             Gemini API key(s) for VLM-based detectors.
        max_retries_per_key:  Per-key attempt limit before rotating.
        max_total_retries:    Total rotation loops before giving up.
    """

    def __init__(
        self,
        api_keys: List[str],
        max_retries_per_key: int = 2,
        max_total_retries: int = 2,
    ):
        if not api_keys:
            raise ValueError("At least one Gemini API key is required.")
        self.api_keys = list(api_keys)
        self.max_retries_per_key = max_retries_per_key
        self.max_total_retries = max_total_retries

        # Lazy-initialised detectors
        self._hand_detector: Optional[HandDistortionDetector] = None
        self._missing_accessory_detector: Optional[MissingAccessoryDetector] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(
        self,
        image_path: str,
        original_image_path: Optional[str] = None,
        expected_accessories: Optional[List[str]] = None,
    ) -> ProblemDetectionReport:
        """Run all enabled detectors on a try-on image.

        Args:
            image_path: Path to the generated try-on image.
            original_image_path: Optional path to the original (pre-try-on) image.
                When provided, uses two-image comparison to detect missing accessories.
                When omitted, falls back to single-image detection with expected_accessories.
            expected_accessories: Optional accessory labels expected in output.
                Only used when original_image_path is not provided.

        Returns:
            ProblemDetectionReport describing any detected problems.
        """
        path = Path(image_path)
        if not path.exists():
            return ProblemDetectionReport(
                image_path=image_path,
                has_problems=True,
                error=f"Image not found: {image_path}",
            )

        problems: List[DetectedProblem] = []
        accessory_detection: Optional[dict] = None

        # ---- Hand distortion (always on) ----
        hand_result = self._run_hand_detection(image_path)
        if hand_result.error:
            logger.warning("Hand detector warning: %s", hand_result.error)
        if hand_result.distorted:
            problems.append(DetectedProblem(
                problem_type="hand_distortion",
                severity="high",
                description=hand_result.reason or "Hand distortion detected.",
                detail=hand_result.to_dict(),
            ))

        # ---- Accessory detection (always on) ----
        if original_image_path:
            # Two-image comparison: detect what is missing relative to the original
            missing_result = self._run_missing_accessory_detection(original_image_path, image_path)
            accessory_detection = missing_result.to_dict()
            if missing_result.error:
                logger.warning("Missing accessory detector warning: %s", missing_result.error)
                problems.append(DetectedProblem(
                    problem_type="accessory_detection_error",
                    severity="low",
                    description="Accessory detector failed during two-image comparison.",
                    detail=missing_result.to_dict(),
                ))
            elif missing_result.has_missing:
                problems.append(DetectedProblem(
                    problem_type="missing_accessories",
                    severity="medium",
                    description=f"Accessories missing in try-on: {', '.join(missing_result.missing_labels)}",
                    detail=missing_result.to_dict(),
                ))
        else:
            return ProblemDetectionReport(
                image_path=image_path,
                has_problems=True,
                problems=problems,
                error="original_image_path is required for accessory detection.",
            )

        report = ProblemDetectionReport(
            image_path=image_path,
            has_problems=bool(problems),
            problems=problems,
            accessory_detection=accessory_detection,
        )
        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------
    # Convenience: detect from TryOnResult
    # ------------------------------------------------------------------

    def detect_from_tryon_result(self, tryon_result) -> ProblemDetectionReport:
        """Run detection on the output of a TryOnAgent result.

        Args:
            tryon_result: TryOnResult dataclass (from tryon_agent.py).

        Returns:
            ProblemDetectionReport, or a failed report if no output path.
        """
        if not tryon_result.success or not tryon_result.output_path:
            return ProblemDetectionReport(
                image_path="",
                has_problems=True,
                error="TryOnResult has no valid output image.",
            )

        original_image_path = getattr(tryon_result, "original_image_path", None)
        if original_image_path is None:
            original_image_path = getattr(tryon_result, "person_image_path", None)

        return self.detect(
            tryon_result.output_path,
            original_image_path=original_image_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_hand_detector(self) -> HandDistortionDetector:
        if self._hand_detector is None:
            self._hand_detector = HandDistortionDetector(
                api_keys=self.api_keys,
                max_retries_per_key=self.max_retries_per_key,
                max_total_retries=self.max_total_retries,
            )
        return self._hand_detector

    def _get_missing_accessory_detector(self) -> MissingAccessoryDetector:
        if self._missing_accessory_detector is None:
            self._missing_accessory_detector = MissingAccessoryDetector()
        return self._missing_accessory_detector

    def _run_hand_detection(self, image_path: str) -> HandDetectionResult:
        logger.debug("Running hand distortion check on %s", Path(image_path).name)
        try:
            return self._get_hand_detector().detect(image_path)
        except Exception as exc:
            from postvton.tools.detection.hand_detector import HandDetectionResult
            return HandDetectionResult(
                distorted=False,
                error=f"Hand detector raised an exception: {exc}",
            )

    def _run_missing_accessory_detection(
        self, original_image_path: str, tryon_image_path: str
    ) -> MissingAccessoryResult:
        logger.debug(
            "Comparing accessories: %s vs %s",
            Path(original_image_path).name,
            Path(tryon_image_path).name,
        )
        try:
            return self._get_missing_accessory_detector().detect_missing(
                original_image_path, tryon_image_path
            )
        except Exception as exc:
            error_msg = f"Missing accessory detector raised an exception: {exc}"
            error_result = MissingAccessoryDetectionResult(
                image_path=tryon_image_path,
                image_size=(0, 0),
                error=error_msg,
            )
            return MissingAccessoryResult(
                original_detection=error_result,
                tryon_detection=error_result,
                missing_by_label={},
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_problems(
    image_path: str,
    api_keys: List[str],
    original_image_path: Optional[str] = None,
    expected_accessories: Optional[List[str]] = None,
    max_retries_per_key: int = 2,
    max_total_retries: int = 2,
) -> ProblemDetectionReport:
    """One-shot problem detection for a try-on image.

    Args:
        image_path:           Path to the generated try-on image.
        api_keys:             Gemini API key(s).
        original_image_path:  Optional path to the original (pre-try-on) image.
            When provided, two-image comparison detects missing accessories.
        expected_accessories: Optional labels expected in final output.
            Only used when original_image_path is not provided.

    Returns:
        ProblemDetectionReport.
    """
    agent = ProblemDetectionAgent(
        api_keys=api_keys,
        max_retries_per_key=max_retries_per_key,
        max_total_retries=max_total_retries,
    )
    return agent.detect(
        image_path,
        original_image_path=original_image_path,
        expected_accessories=expected_accessories,
    )
