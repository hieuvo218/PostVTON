"""Problem Detection Agent for virtual try-on quality assessment.

Detects visual problems in try-on output images using the available
detection tools. Currently supports hand distortion detection via Gemini VLM.

Usage::

    agent = ProblemDetectionAgent(api_keys=["key1", "key2"])
    report = agent.detect("outputs/tryon_result.jpg")
    if report.has_problems:
        print(report.problems)
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from postvton.tools.detection.hand_detector import (
    HandDistortionDetector,
    HandDetectionResult,
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
        check_hands:          Enable hand distortion detection (default True).
    """

    def __init__(
        self,
        api_keys: List[str],
        max_retries_per_key: int = 2,
        max_total_retries: int = 2,
        check_hands: bool = True,
    ):
        if not api_keys:
            raise ValueError("At least one Gemini API key is required.")
        self.api_keys = list(api_keys)
        self.max_retries_per_key = max_retries_per_key
        self.max_total_retries = max_total_retries
        self.check_hands = check_hands

        # Lazy-initialised detectors
        self._hand_detector: Optional[HandDistortionDetector] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> ProblemDetectionReport:
        """Run all enabled detectors on a try-on image.

        Args:
            image_path: Path to the generated try-on image.

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

        # ---- Hand distortion ----
        if self.check_hands:
            hand_result = self._run_hand_detection(image_path)
            if hand_result.error:
                print(f"[ProblemDetectionAgent] Hand detector warning: {hand_result.error}")
            if hand_result.distorted:
                problems.append(DetectedProblem(
                    problem_type="hand_distortion",
                    severity="high",
                    description=hand_result.reason or "Hand distortion detected.",
                    detail=hand_result.to_dict(),
                ))

        report = ProblemDetectionReport(
            image_path=image_path,
            has_problems=bool(problems),
            problems=problems,
        )
        print(f"[ProblemDetectionAgent] {report.summary()}")
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
        return self.detect(tryon_result.output_path)

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

    def _run_hand_detection(self, image_path: str) -> HandDetectionResult:
        print(f"[ProblemDetectionAgent] Running hand distortion check on {Path(image_path).name}")
        try:
            return self._get_hand_detector().detect(image_path)
        except Exception as exc:
            from postvton.tools.detection.hand_detector import HandDetectionResult
            return HandDetectionResult(
                distorted=False,
                error=f"Hand detector raised an exception: {exc}",
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_problems(
    image_path: str,
    api_keys: List[str],
    check_hands: bool = True,
    max_retries_per_key: int = 2,
    max_total_retries: int = 2,
) -> ProblemDetectionReport:
    """One-shot problem detection for a try-on image.

    Args:
        image_path:   Path to the generated try-on image.
        api_keys:     Gemini API key(s).
        check_hands:  Enable hand distortion check.

    Returns:
        ProblemDetectionReport.
    """
    agent = ProblemDetectionAgent(
        api_keys=api_keys,
        max_retries_per_key=max_retries_per_key,
        max_total_retries=max_total_retries,
        check_hands=check_hands,
    )
    return agent.detect(image_path)
