"""Problem Detection Agent for virtual try-on quality assessment.

Detects visual problems in try-on output images using the available
detection tools. Runs both hand distortion and accessory detection.

Usage::

	agent = ProblemDetectionAgent(api_keys=["key1"])
	report = agent.detect(tryon_image, original_image=person_image)
	print(report.to_dict())
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

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
class AccessoriesReport:
	"""Accessory detection summary."""
	missing: bool
	details: List[dict] = field(default_factory=list)

	def to_dict(self) -> dict:
		return {
			"missing": self.missing,
			"details": list(self.details),
		}


@dataclass
class HandsReport:
	"""Hand detection summary."""
	distorted: bool
	analysis: str = ""

	def to_dict(self) -> dict:
		return {
			"distorted": self.distorted,
			"analysis": self.analysis,
		}


@dataclass
class ProblemDetectionReport:
	"""Full detection report for one try-on image."""
	image_id: str
	accessories: AccessoriesReport
	hands: HandsReport
	error: Optional[str] = None

	# ---- helpers ----

	def summary(self) -> str:
		if self.error:
			return f"Detection failed for {self.image_id}: {self.error}"
		parts = []
		if self.hands.distorted:
			parts.append("hand_distortion")
		if self.accessories.missing:
			parts.append("missing_accessories")
		if not parts:
			return f"No problems detected in {self.image_id}"
		return f"Problems detected in {self.image_id}: {', '.join(parts)}"

	def to_dict(self) -> dict:
		return {
			"accessories": self.accessories.to_dict(),
			"hands": self.hands.to_dict(),
		}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ProblemDetectionAgent:
	"""Detect visual problems in virtual try-on images.

	Args:
		api_keys:             Optional API key(s) for VLM-based detectors.
		max_retries_per_key:  Per-key attempt limit before rotating.
		max_total_retries:    Total rotation loops before giving up.
	"""

	def __init__(
		self,
		api_keys: Optional[List[str]] = None,
		max_retries_per_key: int = 2,
		max_total_retries: int = 2,
	):
		self.api_keys = list(api_keys or [])
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
		image: Image.Image,
		original_image: Optional[Image.Image] = None,
		image_id: str = "image",
	) -> ProblemDetectionReport:
		"""Run all enabled detectors on a try-on image.

		Args:
			image: PIL image for the generated try-on output.
			original_image: Optional PIL image for the original (pre-try-on) image.
				When provided, uses two-image comparison to detect missing accessories.
				When omitted, accessory detection is not performed.
			image_id: Optional identifier for logging/reporting.

		Returns:
			ProblemDetectionReport describing any detected problems.
		"""
		accessories_report = AccessoriesReport(missing=False, details=[])
		hands_report = HandsReport(distorted=False, analysis="")

		if not isinstance(image, Image.Image):
			return ProblemDetectionReport(
				image_id=image_id,
				accessories=accessories_report,
				hands=hands_report,
				error=f"Expected PIL.Image.Image, got {type(image).__name__}",
			)

		# ---- Hand distortion (always on) ----
		hand_result = self._run_hand_detection(image)
		if hand_result.error:
			logger.warning("Hand detector warning: %s", hand_result.error)
		hands_report = HandsReport(
			distorted=hand_result.distorted,
			analysis=hand_result.reason or hand_result.description or "",
		)

		# ---- Accessory detection (always on) ----
		error: Optional[str] = None
		if original_image is not None:
			# Two-image comparison: detect what is missing relative to the original
			missing_result = self._run_missing_accessory_detection(original_image, image)
			if missing_result.error:
				logger.warning("Missing accessory detector warning: %s", missing_result.error)
				accessories_report = AccessoriesReport(missing=False, details=[])
				error = missing_result.error
			else:
				details = [
					{"class": label, "count": int(count)}
					for label, count in missing_result.missing_by_label.items()
				]
				accessories_report = AccessoriesReport(
					missing=missing_result.has_missing,
					details=details,
				)
				error = None
		else:
			return ProblemDetectionReport(
				image_id=image_id,
				accessories=accessories_report,
				hands=hands_report,
				error="original_image is required for accessory detection.",
			)

		report = ProblemDetectionReport(
			image_id=image_id,
			accessories=accessories_report,
			hands=hands_report,
			error=error,
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
		output_image = getattr(tryon_result, "output_image", None)
		if not tryon_result.success or output_image is None:
			return ProblemDetectionReport(
				image_id="",
				accessories=AccessoriesReport(missing=False, details=[]),
				hands=HandsReport(distorted=False, analysis=""),
				error="TryOnResult has no valid output image.",
			)

		original_image = getattr(tryon_result, "original_image", None)
		if original_image is None:
			original_image = getattr(tryon_result, "person_image", None)

		return self.detect(
			output_image,
			original_image=original_image,
		)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _get_hand_detector(self) -> HandDistortionDetector:
		if self._hand_detector is None:
			self._hand_detector = HandDistortionDetector(api_keys=self.api_keys)
		return self._hand_detector

	def _get_missing_accessory_detector(self) -> MissingAccessoryDetector:
		if self._missing_accessory_detector is None:
			self._missing_accessory_detector = MissingAccessoryDetector()
		return self._missing_accessory_detector

	def _run_hand_detection(self, image: Image.Image) -> HandDetectionResult:
		logger.debug("Running hand distortion check")
		try:
			return self._get_hand_detector().detect(image)
		except Exception as exc:
			from postvton.tools.detection.hand_detector import HandDetectionResult
			return HandDetectionResult(
				distorted=False,
				error=f"Hand detector raised an exception: {exc}",
			)

	def _run_missing_accessory_detection(
		self, original_image: Image.Image, tryon_image: Image.Image
	) -> MissingAccessoryResult:
		logger.debug("Comparing accessories between original and try-on images")
		try:
			return self._get_missing_accessory_detector().detect_missing(
				original_image, tryon_image
			)
		except Exception as exc:
			error_msg = f"Missing accessory detector raised an exception: {exc}"
			orig_error = MissingAccessoryDetectionResult(
				image_id="original",
				image_size=(0, 0),
				error=error_msg,
			)
			tryon_error = MissingAccessoryDetectionResult(
				image_id="tryon",
				image_size=(0, 0),
				error=error_msg,
			)
			return MissingAccessoryResult(
				original_detection=orig_error,
				tryon_detection=tryon_error,
				missing_by_label={},
			)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_problems(
	image: Image.Image,
	api_keys: Optional[List[str]] = None,
	original_image: Optional[Image.Image] = None,
	image_id: str = "image",
	max_retries_per_key: int = 2,
	max_total_retries: int = 2,
) -> ProblemDetectionReport:
	"""One-shot problem detection for a try-on image.

	Args:
		image:               PIL try-on image.
		api_keys:            Optional API keys for the vision model.
		original_image:      Optional original (pre-try-on) image.
			When provided, two-image comparison detects missing accessories.
		image_id:            Optional identifier for reporting.

	Returns:
		ProblemDetectionReport.
	"""
	agent = ProblemDetectionAgent(
		api_keys=api_keys,
		max_retries_per_key=max_retries_per_key,
		max_total_retries=max_total_retries,
	)
	return agent.detect(
		image,
		original_image=original_image,
		image_id=image_id,
	)
