"""Run detection and execution agents for PostVTON.

Example:
	python scripts/run_postvton.py \
		--person path/to/person.jpg \
		--tryon path/to/tryon.png \
		--api-keys key1,key2 \
		--output output/final.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image

from postvton.agents.problem_detection_agent import ProblemDetectionAgent
from postvton.agents.execution_agent import ExecutionAgent


def _parse_api_keys(raw: str) -> List[str]:
	return [k.strip() for k in raw.split(",") if k.strip()]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run PostVTON detection + execution")
	parser.add_argument("--person", required=True, help="Path to original person image")
	parser.add_argument("--tryon", required=True, help="Path to try-on output image")
	parser.add_argument("--api-keys", default="", help="Comma-separated VLM API keys")
	parser.add_argument("--output", default="output/final.png", help="Final output path")
	parser.add_argument("--skip-execution", action="store_true", help="Only run detection")
	return parser


def main() -> int:
	args = build_parser().parse_args()

	person_path = Path(args.person)
	tryon_path = Path(args.tryon)
	if not person_path.exists():
		raise SystemExit(f"Person image not found: {person_path}")
	if not tryon_path.exists():
		raise SystemExit(f"Try-on image not found: {tryon_path}")

	api_keys = _parse_api_keys(args.api_keys)

	person_image = Image.open(person_path).convert("RGB")
	tryon_image = Image.open(tryon_path).convert("RGB")

	detector = ProblemDetectionAgent(api_keys=api_keys)
	report = detector.detect(
		image=tryon_image,
		original_image=person_image,
		image_id=tryon_path.name,
	)
	print("Detection report:")
	print(report.to_dict())

	if args.skip_execution:
		return 0

	executor = ExecutionAgent()
	result = executor.execute(
		original_image=person_image,
		tryon_image=tryon_image,
	)
	if result.success and result.final_image is not None:
		Path(args.output).parent.mkdir(parents=True, exist_ok=True)
		result.final_image.save(args.output)
	print("Execution result:")
	print(result.to_dict())

	return 0 if result.success else 1


if __name__ == "__main__":
	raise SystemExit(main())
