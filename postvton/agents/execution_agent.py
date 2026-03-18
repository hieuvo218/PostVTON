"""Execution agent for post-processing try-on outputs.

This agent orchestrates editing tools in sequence:
1) hand refinement
2) accessory restoration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import shutil

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStepResult:
	"""One step outcome inside execution flow."""

	name: str
	success: bool
	output_path: Optional[str] = None
	error: Optional[str] = None
	detail: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict:
		return {
			"name": self.name,
			"success": self.success,
			"output_path": self.output_path,
			"error": self.error,
			"detail": self.detail,
		}


@dataclass
class ExecutionResult:
	"""Overall result of execution agent post-processing."""

	success: bool
	original_image_path: str
	tryon_image_path: str
	final_output_path: Optional[str]
	steps: List[ExecutionStepResult] = field(default_factory=list)
	error: Optional[str] = None

	def to_dict(self) -> dict:
		return {
			"success": self.success,
			"original_image_path": self.original_image_path,
			"tryon_image_path": self.tryon_image_path,
			"final_output_path": self.final_output_path,
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

	def execute(
		self,
		original_image_path: str,
		tryon_image_path: str,
		output_path: Optional[str] = None,
		refine_hands: bool = True,
		restore_accessories: bool = True,
		work_dir: str = "outputs/execution_agent",
		hand_params: Optional[Dict[str, Any]] = None,
		accessory_params: Optional[Dict[str, Any]] = None,
	) -> ExecutionResult:
		"""Execute post-processing pipeline.

		Args:
			original_image_path: Original person image path.
			tryon_image_path: Raw try-on output image path.
			output_path: Final output path. Auto-generated if None.
			refine_hands: Whether to run HandRefiner first.
			restore_accessories: Whether to run AccessoryRestorer after that.
			work_dir: Directory for intermediate outputs.
			hand_params: Extra kwargs for HandRefiner.refine().
			accessory_params: Extra kwargs for AccessoryRestorer.restore().
		"""
		original = Path(original_image_path)
		tryon = Path(tryon_image_path)

		if not original.exists():
			return ExecutionResult(
				success=False,
				original_image_path=original_image_path,
				tryon_image_path=tryon_image_path,
				final_output_path=None,
				error=f"Original image not found: {original_image_path}",
			)
		if not tryon.exists():
			return ExecutionResult(
				success=False,
				original_image_path=original_image_path,
				tryon_image_path=tryon_image_path,
				final_output_path=None,
				error=f"Try-on image not found: {tryon_image_path}",
			)

		steps: List[ExecutionStepResult] = []
		hand_params = dict(hand_params or {})
		accessory_params = dict(accessory_params or {})

		work = Path(work_dir)
		work.mkdir(parents=True, exist_ok=True)

		final_path = Path(output_path) if output_path else (work / f"{tryon.stem}_postprocessed{tryon.suffix}")
		current_image = tryon

		# Step 1: hand refinement
		if refine_hands:
			hand_output = work / f"{tryon.stem}_hand_refined{tryon.suffix}"
			hand_refine_result = self._get_hand_refiner().refine(
				image=str(current_image),
				output_path=str(hand_output),
				**hand_params,
			)
			steps.append(
				ExecutionStepResult(
					name="hand_refinement",
					success=hand_refine_result.success,
					output_path=hand_refine_result.output_path,
					error=hand_refine_result.error,
					detail=hand_refine_result.to_dict(),
				)
			)
			if not hand_refine_result.success or not hand_refine_result.output_path:
				return ExecutionResult(
					success=False,
					original_image_path=str(original),
					tryon_image_path=str(tryon),
					final_output_path=None,
					steps=steps,
					error=hand_refine_result.error or "Hand refinement failed.",
				)
			current_image = Path(hand_refine_result.output_path)
			logger.info("Hand refinement completed: %s", current_image)

		# Step 2: accessory restoration
		if restore_accessories:
			accessory_result = self._get_accessory_restorer().restore(
				source_image=str(original),
				target_image=str(current_image),
				output_path=str(final_path),
				**accessory_params,
			)
			steps.append(
				ExecutionStepResult(
					name="accessory_restoration",
					success=accessory_result.success,
					output_path=accessory_result.output_path,
					error=accessory_result.error,
					detail=accessory_result.to_dict(),
				)
			)
			if not accessory_result.success:
				return ExecutionResult(
					success=False,
					original_image_path=str(original),
					tryon_image_path=str(tryon),
					final_output_path=None,
					steps=steps,
					error=accessory_result.error or "Accessory restoration failed.",
				)

			resolved_output = accessory_result.output_path
			if not resolved_output:
				# Defensive fallback: restorer may be configured without output_path.
				final_path.parent.mkdir(parents=True, exist_ok=True)
				shutil.copy2(current_image, final_path)
				resolved_output = str(final_path)

			return ExecutionResult(
				success=True,
				original_image_path=str(original),
				tryon_image_path=str(tryon),
				final_output_path=resolved_output,
				steps=steps,
			)

		# If restoration is disabled, persist the latest image as final output.
		final_path.parent.mkdir(parents=True, exist_ok=True)
		if current_image.resolve() != final_path.resolve():
			shutil.copy2(current_image, final_path)

		return ExecutionResult(
			success=True,
			original_image_path=str(original),
			tryon_image_path=str(tryon),
			final_output_path=str(final_path),
			steps=steps,
		)


def run_execution_agent(
	original_image_path: str,
	tryon_image_path: str,
	output_path: Optional[str] = None,
	refine_hands: bool = True,
	restore_accessories: bool = True,
	work_dir: str = "outputs/execution_agent",
	hand_params: Optional[Dict[str, Any]] = None,
	accessory_params: Optional[Dict[str, Any]] = None,
) -> ExecutionResult:
	"""One-shot API for execution agent flow."""
	agent = ExecutionAgent()
	return agent.execute(
		original_image_path=original_image_path,
		tryon_image_path=tryon_image_path,
		output_path=output_path,
		refine_hands=refine_hands,
		restore_accessories=restore_accessories,
		work_dir=work_dir,
		hand_params=hand_params,
		accessory_params=accessory_params,
	)

