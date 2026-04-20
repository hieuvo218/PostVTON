"""Execution agent for post-processing try-on outputs.

This agent orchestrates editing tools in sequence:
1) hand refinement
2) accessory restoration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStepResult:
	"""One step outcome inside execution flow."""

	name: str
	success: bool
	output_image: Optional[Image.Image] = None
	error: Optional[str] = None
	detail: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict:
		return {
			"name": self.name,
			"success": self.success,
			"output_image": self.output_image is not None,
			"error": self.error,
			"detail": self.detail,
		}


@dataclass
class ExecutionResult:
	"""Overall result of execution agent post-processing."""

	success: bool
	final_image: Optional[Image.Image]
	steps: List[ExecutionStepResult] = field(default_factory=list)
	error: Optional[str] = None

	def to_dict(self) -> dict:
		return {
			"success": self.success,
			"final_image": self.final_image is not None,
			"steps": [step.to_dict() for step in self.steps],
			"error": self.error,
		}


class ExecutionAgent:
	"""Run post-processing steps for a try-on result image.

	The default flow is:
	- refine malformed hands
	- restore missing accessories from the original image
	"""

	def __init__(
		self,
		hand_refiner: Optional[Any] = None,
		accessory_restorer: Optional[Any] = None,
		hand_model_path: str = "ovedrive/qwen-image-edit-4bit",
		hand_device: Optional[str] = None,
		accessory_model_path: str = "yoloe-11l-seg.pt",
		accessory_class_names: Optional[List[str]] = None,
		accessory_conf: float = 0.25,
	):
		self._hand_refiner = hand_refiner
		self._accessory_restorer = accessory_restorer

		self.hand_model_path = hand_model_path
		self.hand_device = hand_device

		self.accessory_model_path = accessory_model_path
		self.accessory_class_names = accessory_class_names
		self.accessory_conf = accessory_conf

	def _get_hand_refiner(self) -> Any:
		if self._hand_refiner is None:
			from postvton.tools.editing.hand_refinement import HandRefiner

			self._hand_refiner = HandRefiner(
				model_path=self.hand_model_path,
				device=self.hand_device,
			)
		return self._hand_refiner

	def _get_accessory_restorer(self) -> Any:
		if self._accessory_restorer is None:
			from postvton.tools.editing.accessory_restoration import AccessoryRestorer

			self._accessory_restorer = AccessoryRestorer(
				model_path=self.accessory_model_path,
				class_names=self.accessory_class_names,
				conf=self.accessory_conf,
			)
		return self._accessory_restorer


	def run_hand_refinement(
		self,
		tryon_image: Image.Image,
		hand_params: Optional[Dict[str, Any]] = None,
	) -> ExecutionStepResult:
		"""Run only the hand refinement step and return step result."""
		if not isinstance(tryon_image, Image.Image):
			return ExecutionStepResult(
				name="hand_refinement",
				success=False,
				error=f"Expected PIL.Image.Image for tryon_image, got {type(tryon_image).__name__}",
			)

		hand_refine_result = self._get_hand_refiner().refine(
			image=tryon_image,
			**dict(hand_params or {}),
		)

		return ExecutionStepResult(
			name="hand_refinement",
			success=hand_refine_result.success,
			output_image=hand_refine_result.output_image,
			error=hand_refine_result.error,
			detail=hand_refine_result.to_dict(),
		)

	def run_accessory_restoration(
		self,
		original_image: Image.Image,
		target_image: Image.Image,
		accessory_params: Optional[Dict[str, Any]] = None,
	) -> ExecutionStepResult:
		"""Run only the accessory restoration step and return step result."""
		if not isinstance(original_image, Image.Image):
			return ExecutionStepResult(
				name="accessory_restoration",
				success=False,
				error=f"Expected PIL.Image.Image for original_image, got {type(original_image).__name__}",
			)

		if not isinstance(target_image, Image.Image):
			return ExecutionStepResult(
				name="accessory_restoration",
				success=False,
				error=f"Expected PIL.Image.Image for target_image, got {type(target_image).__name__}",
			)

		accessory_result = self._get_accessory_restorer().restore(
			source_image=original_image,
			target_image=target_image,
			**dict(accessory_params or {}),
		)

		return ExecutionStepResult(
			name="accessory_restoration",
			success=accessory_result.success,
			output_image=accessory_result.output_image,
			error=accessory_result.error,
			detail=accessory_result.to_dict(),
		)

	def execute(
		self,
		original_image: Image.Image,
		tryon_image: Image.Image,
		refine_hands: bool = True,
		restore_accessories: bool = True,
		hand_params: Optional[Dict[str, Any]] = None,
		accessory_params: Optional[Dict[str, Any]] = None,
	) -> ExecutionResult:
		"""Execute post-processing pipeline.

		Args:
			original_image: Original person image (PIL).
			tryon_image: Raw try-on output image (PIL).
			refine_hands: Whether to run HandRefiner first.
			restore_accessories: Whether to run AccessoryRestorer after that.
			hand_params: Extra kwargs for HandRefiner.refine().
			accessory_params: Extra kwargs for AccessoryRestorer.restore().
		"""
		if not isinstance(original_image, Image.Image):
			return ExecutionResult(
				success=False,
				final_image=None,
				error=f"Expected PIL.Image.Image for original_image, got {type(original_image).__name__}",
			)

		if not isinstance(tryon_image, Image.Image):
			return ExecutionResult(
				success=False,
				final_image=None,
				error=f"Expected PIL.Image.Image for tryon_image, got {type(tryon_image).__name__}",
			)

		steps: List[ExecutionStepResult] = []
		hand_params = dict(hand_params or {})
		accessory_params = dict(accessory_params or {})

		current_image = tryon_image

		# Step 1: hand refinement
		if refine_hands:
			hand_step = self.run_hand_refinement(
				tryon_image=current_image,
				hand_params=hand_params,
			)
			steps.append(hand_step)
			if not hand_step.success or hand_step.output_image is None:
				return ExecutionResult(
					success=False,
					final_image=None,
					steps=steps,
					error=hand_step.error or "Hand refinement failed.",
				)
			current_image = hand_step.output_image
			logger.info("Hand refinement completed")

		# Step 2: accessory restoration
		if restore_accessories:
			accessory_step = self.run_accessory_restoration(
				original_image=original_image,
				target_image=current_image,
				accessory_params=accessory_params,
			)
			steps.append(accessory_step)
			if not accessory_step.success:
				return ExecutionResult(
					success=False,
					final_image=None,
					steps=steps,
					error=accessory_step.error or "Accessory restoration failed.",
				)

			final_image = accessory_step.output_image or current_image
			return ExecutionResult(
				success=True,
				final_image=final_image,
				steps=steps,
			)

		# If restoration is disabled, return the latest image as final output.
		return ExecutionResult(
			success=True,
			final_image=current_image,
			steps=steps,
		)


def run_execution_agent(
	original_image: Image.Image,
	tryon_image: Image.Image,
	refine_hands: bool = True,
	restore_accessories: bool = True,
	hand_params: Optional[Dict[str, Any]] = None,
	accessory_params: Optional[Dict[str, Any]] = None,
) -> ExecutionResult:
	"""One-shot API for execution agent flow."""
	agent = ExecutionAgent()
	return agent.execute(
		original_image=original_image,
		tryon_image=tryon_image,
		refine_hands=refine_hands,
		restore_accessories=restore_accessories,
		hand_params=hand_params,
		accessory_params=accessory_params,
	)

