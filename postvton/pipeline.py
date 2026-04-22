"""Main pipeline entry point for PostVTON.

This module validates and loads user inputs, runs the manager orchestration,
and persists the final virtual try-on result image.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from postvton.manager.manager_agent import ManagerAgent, ManagerState


def _parse_api_keys(raw: str) -> List[str]:
	"""Parse comma-separated API keys into a clean list."""
	return [token.strip() for token in raw.split(",") if token.strip()]


def _load_input_image(image_path: str, label: str) -> Image.Image:
	"""Load an image path as RGB PIL image with clear errors."""
	path = Path(image_path)
	if not path.exists():
		raise FileNotFoundError(f"{label} image not found: {path}")
	if not path.is_file():
		raise ValueError(f"{label} image path is not a file: {path}")

	try:
		return Image.open(path).convert("RGB")
	except Exception as exc:  # pragma: no cover - depends on corrupt file content
		raise ValueError(f"Failed to open {label} image '{path}': {exc}") from exc


def _build_output_path(output_dir: Path, output_name: Optional[str] = None) -> Path:
	"""Build final output path using explicit name or timestamp convention."""
	output_dir.mkdir(parents=True, exist_ok=True)
	if output_name:
		name = output_name if output_name.lower().endswith(".png") else f"{output_name}.png"
		return output_dir / name

	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return output_dir / f"tryon_result_{stamp}.png"


def run_pipeline(
	model_image_path: str,
	garment_image_path: str,
	cloth_type: str = "upper",
	output_dir: str = "output",
	output_name: Optional[str] = None,
	device: str = "cuda",
	max_iterations: int = 2,
) -> Tuple[Path, ManagerState]:
	"""Run full virtual try-on orchestration and save the final image.

	Args:
		model_image_path: Path to person/model image.
		garment_image_path: Path to garment image.
		cloth_type: Garment category expected by downstream modules.
		api_keys: Optional API keys for VLM-assisted detection.
		output_dir: Directory where final result is saved.
		output_name: Optional explicit filename (".png" appended if missing).
		device: Device hint for model components ("cuda" or "cpu").
		max_iterations: Manager refinement loop upper bound.

	Returns:
		Tuple of (saved_output_path, final_manager_state).
	"""
	person_image = _load_input_image(model_image_path, "Model")
	cloth_image = _load_input_image(garment_image_path, "Garment")

	final_output_path = _build_output_path(Path(output_dir), output_name)

	manager = ManagerAgent(device=device, max_iterations=max_iterations)
	state = manager.run(
		person_image=person_image,
		cloth_image=cloth_image,
		cloth_type=cloth_type,
		output_path=str(final_output_path),
		output_dir=output_dir,
	)

	if not final_output_path.exists():
		if state.execution_result and state.execution_result.success and state.execution_result.final_image is not None:
			final_output_path.parent.mkdir(parents=True, exist_ok=True)
			state.execution_result.final_image.save(final_output_path)
		else:
			raise RuntimeError("Pipeline finished without producing a final output image.")

	return final_output_path, state


def build_parser() -> argparse.ArgumentParser:
	"""Create CLI parser for running the full PostVTON pipeline."""
	parser = argparse.ArgumentParser(description="Run full PostVTON manager pipeline")
	parser.add_argument("--model-image", required=True, help="Path to model/person image")
	parser.add_argument("--garment-image", required=True, help="Path to garment image")
	parser.add_argument("--cloth-type", default="upper", help="Garment category")
	parser.add_argument("--output-dir", default="output", help="Directory to save final output")
	parser.add_argument("--output-name", default=None, help="Optional output filename")
	parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
	parser.add_argument("--max-iterations", type=int, default=2, help="Manager refinement loop cap")
	return parser


def main() -> int:
	"""CLI entry point for the PostVTON pipeline."""
	args = build_parser().parse_args()

	try:
		output_path, state = run_pipeline(
			model_image_path=args.model_image,
			garment_image_path=args.garment_image,
			cloth_type=args.cloth_type,
			output_dir=args.output_dir,
			output_name=args.output_name,
			device=args.device,
			max_iterations=args.max_iterations,
		)
	except (FileNotFoundError, ValueError, RuntimeError) as exc:
		print(f"[ERROR] {exc}")
		return 1
	except Exception as exc:  # pragma: no cover - defensive top-level guard
		print(f"[ERROR] Unexpected pipeline failure: {exc}")
		return 1

	print(f"[OK] Final try-on result saved to: {output_path}")
	if state.tryon_result is not None:
		print(f"[OK] Try-on stage success: {state.tryon_result.success}")
	if state.execution_result is not None:
		print(f"[OK] Execution stage success: {state.execution_result.success}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
