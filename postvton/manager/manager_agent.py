"""Manager agent using LangGraph for orchestration.

The manager coordinates try-on generation, problem detection, planning, and
execution in a closed-loop refinement process.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Callable, Dict, List, Optional
import logging
import os
from pathlib import Path
import uuid
import json

try:
	from dotenv import load_dotenv
except Exception:
	load_dotenv = None

try:
	from google import genai
	from google.genai import types as genai_types
except Exception:
	genai = None
	genai_types = None

try:
	from huggingface_hub import InferenceClient
except Exception:
	InferenceClient = None

logger = logging.getLogger(__name__)


@dataclass
class ManagerState:
	"""Shared state for the manager graph."""

	person_image: Any
	cloth_image: Any
	cloth_type: str
	api_keys: List[str] = field(default_factory=list)
	tryon_server_url: Optional[str] = None
	output_path: Optional[str] = None
	output_dir: str = "output"
	max_iterations: int = 2
	num_inference_steps: int = 5
	workflow_mode: str = "whole_workflow"
	iterations: int = 0

	tryon_result: Optional[Any] = None
	tryon_output_path: Optional[str] = None
	raw_tryon_output_path: Optional[str] = None
	tryon_candidate_outputs: Optional[Dict[str, str]] = None
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
		planning_llm: Optional[Callable[[str], str]] = None,
		planning_model_id: str = "gemini-3.1-flash-lite-preview",
		hf_planning_model_id: str = "zai-org/GLM-5.1:zai-org",
	):
		self.device = device
		self.max_iterations = max_iterations
		self.planning_model_id = planning_model_id
		self.hf_planning_model_id = hf_planning_model_id
		self._gemini_client = None
		self._hf_client = None

		# Load API keys from .env if python-dotenv is available.
		if load_dotenv is not None:
			try:
				load_dotenv()
			except Exception:
				pass

		if planning_llm is not None:
			self.planning_llm = planning_llm
		else:
			# Default: prefer Gemini when available; fall back to HF; else deterministic fallback.
			gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
			hf_key = os.environ.get("HF_TOKEN")
			if gemini_key and genai is not None and genai_types is not None:
				self.planning_llm = self._gemini_planning_llm
			elif hf_key and InferenceClient is not None:
				self.planning_llm = self._hf_chat_llm
			else:
				self.planning_llm = self._default_planning_llm

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
		graph.add_conditional_edges(
			"tryon",
			self._after_tryon,
			{"detect": "detect", "execute": "execute", "end": END},
		)
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
		api_keys: Optional[List[str]] = None,
		tryon_server_url: Optional[str] = None,
		output_path: Optional[str] = None,
		output_dir: str = "output",
		num_inference_steps: int = 5,
		workflow_mode: str = "whole_workflow",
	) -> ManagerState:
		"""Run the manager orchestration loop and return final state."""
		state = ManagerState(
			person_image=person_image,
			cloth_image=cloth_image,
			cloth_type=cloth_type,
			api_keys=list(api_keys or []),
			tryon_server_url=tryon_server_url,
			output_path=output_path,
			output_dir=output_dir,
			max_iterations=self.max_iterations,
			num_inference_steps=num_inference_steps,
			workflow_mode=self._normalize_workflow_mode(workflow_mode),
		)
		graph = self.build_graph()
		result = graph.invoke(state)

		if isinstance(result, ManagerState):
			return result

		if isinstance(result, dict):
			merged = asdict(state)
			allowed = {f.name for f in fields(ManagerState)}
			for key in allowed:
				if key in result:
					merged[key] = result[key]
			return ManagerState(**merged)

		raise TypeError(f"Unexpected graph result type: {type(result).__name__}")

	# ------------------------------------------------------------------
	# Graph nodes
	# ------------------------------------------------------------------

	def _node_tryon(self, state: ManagerState) -> ManagerState:
		from postvton.agents.tryon_agent import TryOnAgent
		agent = TryOnAgent(device=self.device, server_url=state.tryon_server_url)
		result = agent.generate(
			person_image_path=state.person_image,
			cloth_image_path=state.cloth_image,
			cloth_type=state.cloth_type,
			output_path=None,
			num_inference_steps=state.num_inference_steps,
		)
		state.tryon_result = result
		state.tryon_output_path = result.output_path
		state.raw_tryon_output_path = result.output_path
		state.tryon_candidate_outputs = getattr(result, "candidate_output_paths", None)
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
		if state.workflow_mode == "pose_only":
			final_output = self._resolve_final_output(state)
			if final_output:
				Path(final_output).parent.mkdir(parents=True, exist_ok=True)
				state.tryon_image_pil.save(final_output)
				state.tryon_output_path = final_output
				if state.tryon_result is not None:
					state.tryon_result.output_path = final_output
		state.history.append({"stage": "tryon", "success": result.success})
		# import sys
		# sys.exit(0)  # Temporary exit to isolate try-on execution during development
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

		detector = ProblemDetectionAgent(api_keys=state.api_keys)
		report = detector.detect(
			image=state.tryon_image_pil,
			original_image=state.person_image_pil,
			image_id=Path(state.tryon_output_path).name if state.tryon_output_path else "tryon",
		)
		state.detection_report = report
		state.history.append({"stage": "detect", "report": report.to_dict()})
		return state

	def _node_plan(self, state: ManagerState) -> ManagerState:
		"""Form a correction plan via PlanningAgent from detection report."""
		from postvton.agents.planning_agent import PlanningAgent

		report = state.detection_report
		if report is None:
			state.plan = {"refine_hands": False, "restore_accessories": False}
			state.history.append({
				"stage": "plan",
				"plan": state.plan,
				"source": "fallback-no-report",
			})
			return state

		report_dict = report.to_dict() if hasattr(report, "to_dict") else {}
		planner = PlanningAgent(llm=self.planning_llm)
		try:
			plan_result = planner.run(report_dict)
		except Exception as exc:
			fallback_plan = self._fallback_plan_from_report(report)
			state.plan = fallback_plan
			state.history.append({
				"stage": "plan",
				"plan": fallback_plan,
				"source": "fallback-exception",
				"planning_error": str(exc),
			})
			return state

		if plan_result.error:
			fallback_plan = self._fallback_plan_from_report(report)
			state.plan = fallback_plan
			state.history.append({
				"stage": "plan",
				"plan": fallback_plan,
				"source": "fallback-error",
				"planning_error": plan_result.error,
			})
			return state

		state.plan = self._map_plan_actions_to_flags(plan_result)
		state.history.append({
			"stage": "plan",
			"plan": state.plan,
			"plan_actions": [a.to_dict() for a in plan_result.actions],
			"raw": plan_result.raw,
			"source": "planning-agent",
		})
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

		plan = state.plan or self._workflow_mode_plan(state.workflow_mode)
		executor = ExecutionAgent()

		# Pass hand refinement prompt from detection report when available.
		hand_params = {}
		if getattr(state, "detection_report", None) is not None:
			try:
				edit_prompt = getattr(state.detection_report, "edit_prompt", None)
				# fall back to dict form if dataclass-like object doesn't expose attribute
				if not edit_prompt and hasattr(state.detection_report, "to_dict"):
					report_dict = state.detection_report.to_dict()
					edit_prompt = report_dict.get("edit_prompt")
				if edit_prompt:
					hand_params["edit_prompt"] = edit_prompt
			except Exception:
				pass

		# Allow detection to override planning: if hands are detected as distorted,
		# force refine_hands to True regardless of planner output.
		refine_flag = bool(plan.get("refine_hands", False))
		try:
			if getattr(state, "detection_report", None) is not None:
				det_distorted = getattr(state.detection_report, "distorted", None)
				if det_distorted is None and hasattr(state.detection_report, "to_dict"):
					det_distorted = state.detection_report.to_dict().get("distorted")
				if bool(det_distorted):
					refine_flag = True
		except Exception:
			# If anything goes wrong inspecting the report, keep planner decision.
			pass

		result = executor.execute(
			original_image=state.person_image_pil,
			tryon_image=state.tryon_image_pil,
			refine_hands=refine_flag,
			restore_accessories=plan.get("restore_accessories", False),
			hand_params=hand_params,
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
	def _normalize_workflow_mode(mode: str) -> str:
		value = (mode or "whole_workflow").strip().lower().replace("-", "_").replace(" ", "_")
		aliases = {
			"whole": "whole_workflow",
			"full": "whole_workflow",
			"full_workflow": "whole_workflow",
			"whole_workflow": "whole_workflow",
			"pose": "pose_only",
			"pose_only": "pose_only",
			"pose_accessory": "pose_accessory_edit",
			"pose_accessory_edit": "pose_accessory_edit",
			"pose_plus_accessory_edit": "pose_accessory_edit",
			"pose_+_accessory_edit": "pose_accessory_edit",
			"pose_hand": "pose_hand_fix",
			"pose_hand_fix": "pose_hand_fix",
			"pose_plus_hand_fix": "pose_hand_fix",
			"pose_+_hand_fix": "pose_hand_fix",
		}
		if value not in aliases:
			raise ValueError(
				"workflow_mode must be one of: whole_workflow, pose_only, "
				"pose_accessory_edit, pose_hand_fix"
			)
		return aliases[value]

	@staticmethod
	def _workflow_mode_plan(mode: str) -> Dict[str, bool]:
		if mode == "pose_accessory_edit":
			return {"refine_hands": False, "restore_accessories": True}
		if mode == "pose_hand_fix":
			return {"refine_hands": True, "restore_accessories": False}
		return {"refine_hands": False, "restore_accessories": False}

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

	@staticmethod
	def _fallback_plan_from_report(report: Any) -> Dict[str, bool]:
		return {
			"refine_hands": bool(getattr(report.hands, "distorted", False)),
			"restore_accessories": bool(getattr(report.accessories, "missing", False)),
		}

	@staticmethod
	def _map_plan_actions_to_flags(plan_result) -> Dict[str, bool]:
		"""Map PlanningAgent actions into ExecutionAgent boolean flags."""
		flags = {
			"refine_hands": False,
			"restore_accessories": False,
		}
		for item in plan_result.actions:
			action = str(getattr(item, "action", "")).strip().lower().replace("-", "_").replace(" ", "_")
			if action in {
				"refine_hands",
				"hand_refinement",
				"fix_hands",
				"fix_hand",
				"repair_hands",
				"repair_hand",
			} or (
				any(token in action for token in ("hand", "finger", "palm", "anatomical"))
				and any(token in action for token in ("refine", "fix", "repair", "inpaint", "mask", "high_resolution"))
			):
				flags["refine_hands"] = True
			if action in {
				"restore_accessories",
				"accessory_restoration",
				"restore_accessory",
				"fix_accessories",
				"fix_accessory",
			} or (
				any(token in action for token in ("accessory", "accessories", "watch", "ring", "bracelet", "necklace", "earring"))
				and any(token in action for token in ("restore", "fix", "repair", "recover", "paste", "composite"))
			):
				flags["restore_accessories"] = True
		return flags

	@staticmethod
	def _default_planning_llm(prompt: str) -> str:
		"""Default deterministic planner response when no external LLM is provided."""
		report = {}
		marker = "Perception report:"
		if marker in prompt:
			try:
				report = json.loads(prompt.split(marker, 1)[1].strip())
			except Exception:
				report = {}

		hands_distorted = bool(report.get("hands", {}).get("distorted", False))
		accessories_missing = bool(report.get("accessories", {}).get("missing", False))

		actions: List[Dict[str, str]] = []
		if hands_distorted:
			actions.append(
				{
					"action": "refine_hands",
					"justification": "Hand distortion detected in report.",
					"fallback": "skip_hand_refinement",
				}
			)
		if accessories_missing:
			actions.append(
				{
					"action": "restore_accessories",
					"justification": "Accessory loss detected compared to original image.",
					"fallback": "skip_accessory_restoration",
				}
			)

		if not actions:
			actions.append(
				{
					"action": "noop",
					"justification": "No actionable visual defects detected.",
					"fallback": "noop",
				}
			)

		return json.dumps({"actions": actions})

	def _hf_chat_llm(self, prompt: str) -> str:
		"""Call Hugging Face chat completions using GLM (requires HF_TOKEN)."""
		token = os.environ.get("HF_TOKEN")
		if not token:
			raise ValueError("HF_TOKEN is required to call the planning LLM.")
		if InferenceClient is None:
			raise ImportError("huggingface_hub is required to call the planning LLM.")

		if self._hf_client is None:
			self._hf_client = InferenceClient(api_key=token)

		completion = self._hf_client.chat.completions.create(
			model=self.hf_planning_model_id,
			messages=[{"role": "user", "content": prompt}],
		)
		try:
			message = completion.choices[0].message
			return message.content if isinstance(message.content, str) else str(message.content)
		except Exception:
			return str(completion)

	def _gemini_planning_llm(self, prompt: str) -> str:
		"""Call Google Gemini to produce JSON planning output.

		Requires GEMINI_API_KEY or GOOGLE_API_KEY and the `google-genai` package.
		"""
		api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
		if not api_key:
			raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required to call the planning LLM.")
		if genai is None or genai_types is None:
			raise ImportError("google-genai is required to call the planning LLM.")

		if self._gemini_client is None:
			self._gemini_client = genai.Client(api_key=api_key)

		response = self._gemini_client.models.generate_content(
			model=self.planning_model_id,
			contents=[prompt],
			config=genai_types.GenerateContentConfig(
				temperature=0.2,
				response_mime_type="application/json",
			),
		)
		text = getattr(response, "text", None)
		return text if isinstance(text, str) else str(text or response)

	# ------------------------------------------------------------------
	# Flow control
	# ------------------------------------------------------------------

	def _after_tryon(self, state: ManagerState) -> str:
		if state.workflow_mode == "pose_only":
			return "end"
		if state.workflow_mode in ("pose_accessory_edit", "pose_hand_fix"):
			return "execute"
		return "detect"

	def _should_continue(self, state: ManagerState) -> str:
		if state.workflow_mode != "whole_workflow":
			return "end"

		if state.iterations >= state.max_iterations:
			return "end"

		report = state.detection_report
		if report is None:
			return "end"

		if report.hands.distorted or report.accessories.missing:
			return "continue"
		return "end"

