"""Planning agent using LangGraph.

The planning agent consumes perception reports and produces an ordered list of
corrective actions with justifications and fallback options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json
import logging
import os

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

logger = logging.getLogger(__name__)


@dataclass
class PlanAction:
	"""One corrective action in the plan."""

	action: str
	justification: str
	fallback: str

	def to_dict(self) -> dict:
		return {
			"action": self.action,
			"justification": self.justification,
			"fallback": self.fallback,
		}


@dataclass
class PlanResult:
	"""Structured plan output from the reasoning layer."""

	actions: List[PlanAction] = field(default_factory=list)
	raw: Optional[str] = None
	error: Optional[str] = None

	def to_dict(self) -> dict:
		return {
			"actions": [a.to_dict() for a in self.actions],
			"raw": self.raw,
			"error": self.error,
		}


@dataclass
class PlanningState:
	"""LangGraph state for planning."""

	report: Dict[str, Any]
	plan: Optional[PlanResult] = None


class PlanningAgent:
	"""LLM-driven planning agent using LangGraph.

	The LLM must return a JSON object with the following structure:
	{
	  "actions": [
		{"action": "...", "justification": "...", "fallback": "..."}
	  ]
	}
	"""

	def __init__(
		self,
		llm: Optional[Callable[[str], str]] = None,
		model_id: str = "gemini-3.1-flash-lite-preview",
	):
		if load_dotenv is not None:
			try:
				load_dotenv()
			except Exception:
				pass

		self.model_id = model_id
		self._gemini_client = None
		if llm is not None:
			self.llm = llm
		else:
			api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
			if api_key and genai is not None and genai_types is not None:
				self.llm = self._gemini_llm
			else:
				self.llm = None

	def build_graph(self):
		try:
			from langgraph.graph import StateGraph, END
		except Exception as exc:
			raise ImportError(
				"langgraph is required to use PlanningAgent. "
				"Install it with `pip install langgraph`."
			) from exc

		graph = StateGraph(PlanningState)
		graph.add_node("plan", self._node_plan)
		graph.set_entry_point("plan")
		graph.add_edge("plan", END)
		return graph.compile()

	def run(self, report: Dict[str, Any]) -> PlanResult:
		if self.llm is None:
			raise ValueError(
				"PlanningAgent requires an LLM callable or GEMINI_API_KEY/GOOGLE_API_KEY with google-genai installed."
			)
		graph = self.build_graph()
		initial_state = PlanningState(report=report)
		result = graph.invoke(initial_state)

		if isinstance(result, PlanningState):
			return result.plan or PlanResult(error="No plan produced")

		if isinstance(result, dict):
			state = PlanningState(
				report=result.get("report", report),
				plan=result.get("plan"),
			)
			return state.plan or PlanResult(error="No plan produced")

		return PlanResult(error=f"Unexpected planning state type: {type(result).__name__}")

	# ------------------------------------------------------------------
	# Internal node
	# ------------------------------------------------------------------

	def _node_plan(self, state: PlanningState) -> PlanningState:
		prompt = self._build_prompt(state.report)
		try:
			raw = self.llm(prompt)
		except Exception as exc:
			state.plan = PlanResult(error=f"LLM call failed: {exc}")
			return state

		plan = self._parse_plan(raw)
		state.plan = plan
		return state

	@staticmethod
	def _build_prompt(report: Dict[str, Any]) -> str:
		return (
			"You are a planning agent. Analyze the perception report and produce "
			"a structured, ordered list of corrective actions. The reasoning must "
			"include: (1) identify problem category, (2) select correction operator, "
			"(3) determine execution order. Return ONLY JSON with this schema:\n"
			"{\"actions\": [{\"action\": \"...\", \"justification\": \"...\", "
			"\"fallback\": \"...\"}]}\n\n"
			f"Perception report:\n{json.dumps(report, indent=2)}"
		)

	@staticmethod
	def _parse_plan(raw: str) -> PlanResult:
		try:
			data = json.loads(raw)
		except Exception:
			return PlanResult(raw=raw, error="Failed to parse LLM output as JSON")

		actions: List[PlanAction] = []
		for item in data.get("actions", []):
			if not isinstance(item, dict):
				continue
			action = str(item.get("action", "")).strip()
			justification = str(item.get("justification", "")).strip()
			fallback = str(item.get("fallback", "")).strip()
			if action:
				actions.append(
					PlanAction(
						action=action,
						justification=justification,
						fallback=fallback,
					)
				)

		if not actions:
			return PlanResult(raw=raw, error="No valid actions found in plan")

		return PlanResult(actions=actions, raw=raw)

	def _gemini_llm(self, prompt: str) -> str:
		api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
		if not api_key:
			raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini planning.")
		if genai is None or genai_types is None:
			raise ImportError("google-genai is required for Gemini planning.")

		if self._gemini_client is None:
			self._gemini_client = genai.Client(api_key=api_key)

		response = self._gemini_client.models.generate_content(
			model=self.model_id,
			contents=[prompt],
			config=genai_types.GenerateContentConfig(
				temperature=0.2,
				response_mime_type="application/json",
			),
		)
		text = getattr(response, "text", None)
		return text if isinstance(text, str) else str(text or response)

