"""Gradio UI wrapper for PostVTON.

This UI is only a front-end. All backend logic is executed via the `postvton`
package (ManagerAgent/TryOnAgent/etc.).

Run:
    python scripts/run_gradio_ui.py

Requirements:
- A running try-on server (recommended) at TRYON_SERVER_URL (POST /tryon)
- gradio installed in the active Python environment
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "gradio is required to run the UI. Install with: pip install gradio"
    ) from exc


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "gradio_ui"
_WORKFLOW_MODE_CHOICES = [
    "Whole workflow",
    "Pose only",
    "Pose + accessory edit",
    "Pose + hand fix",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _map_cloth_type(ui_value: str) -> str:
    """Map UI cloth type -> PostVTON cloth_type (upper|lower|overall)."""
    v = (ui_value or "").strip().lower()
    if v.startswith("upper"):
        return "upper"
    if v.startswith("lower"):
        return "lower"
    return "overall"


def _map_workflow_mode(ui_value: str) -> str:
    """Map UI workflow mode -> pipeline workflow mode."""
    value = (ui_value or "").strip().lower()
    if value == "pose only":
        return "pose_only"
    if "accessory" in value:
        return "pose_accessory_edit"
    if "hand" in value:
        return "pose_hand_fix"
    return "whole_workflow"


def _pick_candidate_paths(candidate_outputs: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
    """Pick best-effort paths for CatVTON and OOTDiffusion from downloaded candidates."""
    if not isinstance(candidate_outputs, dict):
        return None, None
    cat_path = None
    oot_path = None
    for k, v in candidate_outputs.items():
        name = str(k).lower()
        if cat_path is None and name == "catvton":
            cat_path = v
        if oot_path is None and name.startswith("ootdiffusion"):
            oot_path = v
    return cat_path, oot_path


def run_postvton(
    person_image_path: str,
    cloth_image_path: str,
    cloth_type: str,
    workflow_mode: str,
    gemini_api_key: str,
    tryon_server_url: str,
    device: str,
    max_iterations: int,
    num_inference_step: int,
):
    if not person_image_path or not cloth_image_path:
        raise gr.Error("Please upload both a person image and a clothing image.")

    # Prefer per-run API key from UI; fall back to env.
    key = (gemini_api_key or "").strip()
    if key:
        os.environ["GEMINI_API_KEY"] = key

    server_url = (tryon_server_url or "").strip()
    if server_url:
        os.environ["TRYON_SERVER_URL"] = server_url

    run_id = uuid.uuid4().hex[:10]
    out_dir = _OUTPUT_ROOT
    _ensure_dir(out_dir)

    from postvton.pipeline import run_pipeline

    final_output_path, state = run_pipeline(
        model_image_path=person_image_path,
        garment_image_path=cloth_image_path,
        cloth_type=_map_cloth_type(cloth_type),
        api_keys=[key] if key else [],
        tryon_server_url=server_url or None,
        output_dir=str(out_dir),
        output_name=f"{run_id}.png",
        device=(device or "cuda"),
        max_iterations=int(max_iterations),
        num_inference_steps=int(num_inference_step),
        workflow_mode=_map_workflow_mode(workflow_mode),
    )

    cat_path, oot_path = _pick_candidate_paths(getattr(state, "tryon_candidate_outputs", None))
    final_path = str(final_output_path)

    return cat_path, oot_path, final_path


def _build_examples():
    model_dir = _PROJECT_ROOT / "external" / "ootdiffusion" / "run" / "examples" / "model"
    garment_dir = _PROJECT_ROOT / "external" / "ootdiffusion" / "run" / "examples" / "garment"

    if not model_dir.exists() or not garment_dir.exists():
        return []

    models = sorted([p for p in model_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    garments = sorted([p for p in garment_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    if not models or not garments:
        return []

    # A few simple pairs (not necessarily matching IDs)
    pairs = []
    for i in range(min(4, len(models), len(garments))):
        pairs.append([
            str(models[i]),
            str(garments[i]),
            "Upper",
            "Whole workflow",
            "",  # gemini_api_key
            "",  # tryon_server_url
            "cuda",
            2,
            5,
        ])
    return pairs


def build_ui() -> gr.Blocks:
    examples = _build_examples()

    with gr.Blocks(title="PostVTON | Virtual Try-On + Post-Processing") as demo:
        gr.Markdown("## PostVTON | Virtual Try-On + Post-Processing")

        with gr.Row():
            with gr.Column(scale=1):
                person_image = gr.Image(label="Person Image", type="filepath")
                cloth_image = gr.Image(label="Clothing Image", type="filepath")
                cloth_type = gr.Radio(["Upper", "Lower", "Dress"], value="Upper", label="Cloth Type")
                workflow_mode = gr.Radio(
                    _WORKFLOW_MODE_CHOICES,
                    value="Whole workflow",
                    label="Workflow Mode",
                )

                run_btn = gr.Button("Run PostVTON")

                with gr.Accordion("Advanced Options", open=False):
                    gr.Markdown("### Gemini Settings")
                    gemini_api_key = gr.Textbox(label="Gemini API Key (GEMINI_API_KEY)", type="password", placeholder="optional")
                    tryon_server_url = gr.Textbox(
                        label="Try-On Server URL (TRYON_SERVER_URL)",
                        placeholder="http://127.0.0.1:8000",
                    )

                    device = gr.Dropdown(["cuda", "cpu"], value="cuda", label="Device")
                    max_iterations = gr.Slider(1, 4, value=2, step=1, label="Max Iterations")
                    num_inference_step = gr.Slider(1, 50, value=5, step=1, label="Num Inference Step")

            with gr.Column(scale=2):
                with gr.Row():
                    catvton_out = gr.Image(label="CatVTON Output", type="filepath")
                    oot_out = gr.Image(label="OOTDiffusion Output", type="filepath")
                final_out = gr.Image(label="Final Edited Output", type="filepath")

        run_btn.click(
            fn=run_postvton,
            inputs=[
                person_image,
                cloth_image,
                cloth_type,
                workflow_mode,
                gemini_api_key,
                tryon_server_url,
                device,
                max_iterations,
                num_inference_step,
            ],
            outputs=[catvton_out, oot_out, final_out],
        )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[
                    person_image,
                    cloth_image,
                    cloth_type,
                    workflow_mode,
                    gemini_api_key,
                    tryon_server_url,
                    device,
                    max_iterations,
                    num_inference_step,
                ],
                outputs=[catvton_out, oot_out, final_out],
                fn=run_postvton,
                cache_examples=False,
            )

    return demo


def main() -> None:
    _ensure_dir(_OUTPUT_ROOT)

    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=True,
    )


if __name__ == "__main__":
    main()
