"""Manager agent using LangGraph for orchestration.

The manager coordinates try-on generation, problem detection, planning, and
execution in a closed-loop refinement process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ManagerState:
	"""Shared state for the manager graph."""

	person_image: Any
	cloth_image: Any
	cloth_type: str
	output_path: Optional[str] = None
	output_dir: str = "output"
	max_iterations: int = 2
	iterations: int = 0

	tryon_result: Optional[Any] = None
	tryon_output_path: Optional[str] = None
	person_image_pil: Optional[Any] = None
	tryon_image_pil: Optional[Any] = None
	detection_report: Optional[Any] = None
	plan: Optional[Dict[str, Any]] = None
	execution_result: Optional[Any] = None
	history: List[Dict[str, Any]] = field(default_factory=list)


class ManagerAgent:
	"""Orchestrate agent interactions using LangGraph."""

	def __init__(
		self,
		device: str = "cuda",
		max_iterations: int = 2,
	):
		self.device = device
		self.max_iterations = max_iterations

	def build_graph(self):
		"""Build and return the LangGraph executable graph."""
		try:
			from langgraph.graph import StateGraph, END
		except Exception as exc:
			raise ImportError(
				"langgraph is required to use ManagerAgent. "
				"Install it with `pip install langgraph`."
			) from exc

		graph = StateGraph(ManagerState)

		graph.add_node("tryon", self._node_tryon)
		graph.add_node("detect", self._node_detect)
		graph.add_node("plan", self._node_plan)
		graph.add_node("execute", self._node_execute)

		graph.set_entry_point("tryon")
		graph.add_edge("tryon", "detect")
		graph.add_edge("detect", "plan")
		graph.add_edge("plan", "execute")
		graph.add_conditional_edges(
			"execute",
			self._should_continue,
			{"continue": "detect", "end": END},
		)

		return graph.compile()

	def run(
		self,
		person_image: Any,
		cloth_image: Any,
		cloth_type: str,
		output_path: Optional[str] = None,
		output_dir: str = "output",
	) -> ManagerState:
		"""Run the manager orchestration loop and return final state."""
		state = ManagerState(
			person_image=person_image,
			cloth_image=cloth_image,
			cloth_type=cloth_type,
			output_path=output_path,
			output_dir=output_dir,
			max_iterations=self.max_iterations,
		)
		graph = self.build_graph()
		return graph.invoke(state)

	# ------------------------------------------------------------------
	# Graph nodes
	# ------------------------------------------------------------------

	def _node_tryon(self, state: ManagerState) -> ManagerState:
		from postvton.agents.tryon_agent import TryOnAgent
		person_path = self._resolve_image_path(state.person_image, "person", state.output_dir)
		cloth_path = self._resolve_image_path(state.cloth_image, "cloth", state.output_dir)
		if person_path is None or cloth_path is None:
			state.history.append({
				"stage": "tryon",
				"success": False,
				"error": "Could not resolve person/cloth inputs to valid image paths.",
			})
			return state

		agent = TryOnAgent(device=self.device)
		result = agent.generate(
			person_image_path=person_path,
			cloth_image_path=cloth_path,
			cloth_type=state.cloth_type,
			output_path=None,
		)
		state.tryon_result = result
		state.tryon_output_path = result.output_path
		state.person_image_pil = self._to_pil(state.person_image)
		if result.output_path:
			state.tryon_image_pil = self._to_pil(result.output_path)
		if state.person_image_pil is None or state.tryon_image_pil is None:
			state.history.append({
				"stage": "tryon",
				"success": False,
				"error": "Failed to load PIL images for manager pipeline.",
			})
			return state
		state.history.append({"stage": "tryon", "success": result.success})
		return state

	def _node_detect(self, state: ManagerState) -> ManagerState:
		from postvton.agents.problem_detection_agent import ProblemDetectionAgent

		if not state.tryon_result or not state.tryon_result.output_path:
			return state
		if state.tryon_image_pil is None or state.person_image_pil is None:
			state.history.append({
				"stage": "detect",
				"error": "PIL images not available for detection.",
			})
			return state

		detector = ProblemDetectionAgent(api_keys=None)
		report = detector.detect(
			image=state.tryon_image_pil,
			original_image=state.person_image_pil,
			image_id=Path(state.tryon_output_path).name if state.tryon_output_path else "tryon",
		)
		state.detection_report = report
		state.history.append({"stage": "detect", "report": report.to_dict()})
		return state

	def _node_plan(self, state: ManagerState) -> ManagerState:
		"""Form a correction plan based on detection report.

		This default planner is rule-based to avoid hard dependency on a
		separate planning agent.
		"""
		report = state.detection_report
		if report is None:
			state.plan = {"refine_hands": False, "restore_accessories": False}
			return state

		plan = {
			"refine_hands": bool(report.hands.distorted),
			"restore_accessories": bool(report.accessories.missing),
		}
		state.plan = plan
		state.history.append({"stage": "plan", "plan": plan})
		return state

	def _node_execute(self, state: ManagerState) -> ManagerState:
		from postvton.agents.execution_agent import ExecutionAgent

		if not state.tryon_result or not state.tryon_result.output_path:
			return state
		if state.tryon_image_pil is None or state.person_image_pil is None:
			state.history.append({
				"stage": "execute",
				"error": "PIL images not available for execution.",
			})
			return state

		plan = state.plan or {"refine_hands": False, "restore_accessories": False}
		executor = ExecutionAgent()
		result = executor.execute(
			original_image=state.person_image_pil,
			tryon_image=state.tryon_image_pil,
			refine_hands=plan.get("refine_hands", False),
			restore_accessories=plan.get("restore_accessories", False),
		)
		state.execution_result = result
		if result.success and result.final_image is not None:
			final_output = self._resolve_final_output(state)
			if final_output:
				Path(final_output).parent.mkdir(parents=True, exist_ok=True)
				result.final_image.save(final_output)
				state.tryon_output_path = final_output
				if state.tryon_result is not None:
					state.tryon_result.output_path = final_output
			state.tryon_image_pil = result.final_image
		state.iterations += 1
		state.history.append({"stage": "execute", "success": result.success})
		return state

	@staticmethod
	def _to_pil(image_input: Any) -> Optional[Any]:
		"""Load an image as PIL.Image when possible."""
		try:
			from PIL import Image
		except Exception:
			Image = None

		try:
			import numpy as np
		except Exception:
			np = None

		if Image is None:
			return None

		if isinstance(image_input, Image.Image):
			return image_input
		if np is not None and isinstance(image_input, np.ndarray):
			return Image.fromarray(image_input).convert("RGB")
		try:
			path = Path(image_input)
			if path.exists():
				return Image.open(path).convert("RGB")
		except Exception:
			return None
		return None

	@staticmethod
	def _resolve_final_output(state: ManagerState) -> Optional[str]:
		if state.output_path:
			return state.output_path
		result_dir = Path(state.output_dir)
		result_dir.mkdir(parents=True, exist_ok=True)
		if state.tryon_result and state.tryon_result.output_path:
			tryon_path = Path(state.tryon_result.output_path)
			return str(result_dir / f"{tryon_path.stem}_fixed{tryon_path.suffix}")
		return str(result_dir / "final.png")

	@staticmethod
	def _resolve_image_path(image_input: Any, label: str, output_dir: str) -> Optional[str]:
		"""Resolve an input image into a filesystem path for try-on models.

		Try-on modules currently expect file paths. If a PIL image or numpy array
		is provided, this helper persists it under output_dir/tmp_inputs/.
		"""
		try:
			from PIL import Image
		except Exception:
			Image = None

		try:
			import numpy as np
		except Exception:
			np = None

		if isinstance(image_input, (str, Path)):
			path = Path(image_input)
			if path.exists():
				return str(path)
			return None

		if Image is not None and isinstance(image_input, Image.Image):
			tmp_dir = Path(output_dir) / "tmp_inputs"
			tmp_dir.mkdir(parents=True, exist_ok=True)
			tmp_path = tmp_dir / f"{label}_{uuid.uuid4().hex[:8]}.png"
			image_input.convert("RGB").save(tmp_path)
			return str(tmp_path)

		if np is not None and isinstance(image_input, np.ndarray):
			tmp_dir = Path(output_dir) / "tmp_inputs"
			tmp_dir.mkdir(parents=True, exist_ok=True)
			tmp_path = tmp_dir / f"{label}_{uuid.uuid4().hex[:8]}.png"
			if Image is None:
				return None
			Image.fromarray(image_input).convert("RGB").save(tmp_path)
			return str(tmp_path)

		return None

	# ------------------------------------------------------------------
	# Flow control
	# ------------------------------------------------------------------

	def _should_continue(self, state: ManagerState) -> str:
		if state.iterations >= state.max_iterations:
			return "end"

		report = state.detection_report
		if report is None:
			return "end"

		if report.hands.distorted or report.accessories.missing:
			return "continue"
		return "end"

