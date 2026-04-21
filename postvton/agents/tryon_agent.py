"""TryOn Agent for Virtual Try-On using CatVTON and OOTDiffusion.

The agent always runs both models, scores each result with MediaPipe pose
cosine similarity against the original person image, and returns the best.
"""

import shutil
import sys
import uuid
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import cv2
import mediapipe as mp
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from postvton.tools.tryon.catvton import CatVTONInference
from postvton.tools.tryon.ootdiffusion import OOTDiffusionInference


class VTONModel(str, Enum):
    """Available virtual try-on models"""
    CATVTON = "catvton"
    OOTDIFFUSION = "ootdiffusion"


@dataclass
class TryOnResult:
    """Result from a try-on operation"""
    success: bool
    output_path: Optional[str] = None
    message: str = ""
    model_used: Optional[str] = None
    inference_time: float = 0.0
    pose_score: float = 0.0   # cosine pose similarity vs. person image (0.0 = not yet scored)


# ---------------------------------------------------------------------------
# Pose comparison utilities
# ---------------------------------------------------------------------------

_mp_pose = mp.solutions.pose


def _extract_pose_keypoints(image_path: str) -> Optional[List[tuple]]:
    """Extract MediaPipe pose landmarks from an image file."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with _mp_pose.Pose(static_image_mode=True) as pose_estimator:
        results = pose_estimator.process(img_rgb)
    if not results.pose_landmarks:
        return None
    return [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]


def _pose_to_vector(keypoints: List[tuple], visibility_threshold: float = 0.6) -> List[float]:
    """Flatten visible pose keypoints into a 1-D feature vector."""
    return [
        coord
        for (x, y, conf) in keypoints
        if conf > visibility_threshold
        for coord in (x, y)
    ]


def _cosine_score(pose_a: List[tuple], pose_b: List[tuple]) -> float:
    """Return cosine similarity in [0, 1] between two pose keypoint sets."""
    vec_a = _pose_to_vector(pose_a)
    vec_b = _pose_to_vector(pose_b)
    min_len = min(len(vec_a), len(vec_b))
    if min_len == 0:
        return 0.0
    score = cosine_similarity([vec_a[:min_len]], [vec_b[:min_len]])[0][0]
    # Clamp to [0.0, 1.0] to handle floating-point precision errors
    return score


def score_pose_similarity(person_image_path: str, tryon_image_path: str) -> float:
    """Compute pose cosine similarity between a person and a try-on image.

    Args:
        person_image_path: Path to the original person photo.
        tryon_image_path:  Path to the generated try-on result.

    Returns:
        Cosine similarity in [0, 1]; 0.0 if pose detection fails for either image.
    """
    person_pose = _extract_pose_keypoints(person_image_path)
    tryon_pose  = _extract_pose_keypoints(tryon_image_path)
    if person_pose is None or tryon_pose is None:
        return 0.0
    return _cosine_score(person_pose, tryon_pose)


class TryOnAgent:
    """Runs both CatVTON and OOTDiffusion, scores each result with MediaPipe
    pose cosine similarity, and returns the best one.

    Args:
        device: "cuda" or "cpu".
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._models: Dict[VTONModel, object] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        person_image_path: str,
        cloth_image_path: str,
        cloth_type: str = "upper",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> TryOnResult:
        """Run both try-on models and return the best result.

        Args:
            person_image_path: Path to the person image.
            cloth_image_path:  Path to the cloth/garment image.
            cloth_type:        Clothing type.
                               CatVTON: "upper", "lower", "overall", "inner", "outer"
                               OOTDiffusion: "upperbody", "lowerbody", "dress"
            output_path:       Where to save the best result.
                               Auto-generated under outputs/tryon_agent/ if None.
            **kwargs:          Extra parameters forwarded to both models
                               (e.g. num_inference_steps, guidance_scale, seed).

        Returns:
            TryOnResult of the highest-scoring model with pose_score populated.
        """
        results: List[TryOnResult] = []

        for model_name in VTONModel:
            result = self._generate_single(
                model_name=model_name,
                person_image_path=person_image_path,
                cloth_image_path=cloth_image_path,
                cloth_type=cloth_type,
                **kwargs,
            )
            if result.success and result.output_path:
                result.pose_score = score_pose_similarity(
                    person_image_path, result.output_path
                )
                print(f"[TryOnAgent] {model_name} pose_score={result.pose_score:.4f}")
            results.append(result)

        successful = [r for r in results if r.success]
        if not successful:
            return TryOnResult(
                success=False,
                message="Both models failed. " + " | ".join(r.message for r in results),
            )

        best = max(successful, key=lambda r: r.pose_score)
        print(f"[TryOnAgent] Best: {best.model_used} (pose_score={best.pose_score:.4f})")

        if output_path and best.output_path and best.output_path != output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best.output_path, output_path)
            best.output_path = output_path

        return best

    def unload(self):
        """Unload all loaded models to free GPU memory."""
        for model_name, model in self._models.items():
            if hasattr(model, "unload"):
                model.unload()
            print(f"[TryOnAgent] Unloaded {model_name}")
        self._models.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_name: VTONModel):
        if model_name in self._models:
            return
        print(f"[TryOnAgent] Loading {model_name} on {self.device}...")
        if model_name == VTONModel.CATVTON:
            self._models[model_name] = CatVTONInference(device=self.device)
        elif model_name == VTONModel.OOTDIFFUSION:
            gpu_id = 0 if self.device.startswith("cuda") else -1
            self._models[model_name] = OOTDiffusionInference(model_type="hd", gpu_id=gpu_id)
        print(f"[TryOnAgent] {model_name} loaded.")

    def _generate_single(
        self,
        model_name: VTONModel,
        person_image_path: str,
        cloth_image_path: str,
        cloth_type: str,
        **kwargs,
    ) -> TryOnResult:
        """Run inference for one model and save output to a temp path."""
        start = time.time()
        try:
            self._load_model(model_name)
            model = self._models[model_name]

            person_pil = Image.open(person_image_path).convert("RGB")
            cloth_pil = Image.open(cloth_image_path).convert("RGB")

            if "seed" in kwargs and kwargs["seed"] is None:
                kwargs["seed"] = -1

            if model_name == VTONModel.CATVTON:
                raw = model.generate(
                    person_image=person_pil,
                    cloth_image=cloth_pil,
                    cloth_type=cloth_type,
                    **kwargs,
                )
            else:
                raw = model.generate(
                    person_image=person_pil,
                    cloth_image=cloth_pil,
                    category=cloth_type,
                    **kwargs,
                )

            # Normalise to a single PIL Image
            if isinstance(raw, list):
                raw = raw[0]
            elif hasattr(raw, "images"):
                raw = raw.images[0]
            if not isinstance(raw, Image.Image):
                raise RuntimeError(f"Unexpected output type: {type(raw)}")

            out_dir = project_root / "outputs" / "tryon_agent"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir / f"{model_name}_{uuid.uuid4().hex[:8]}.png")
            raw.save(out_path)

            elapsed = time.time() - start
            print(f"[TryOnAgent] {model_name} → {out_path} ({elapsed:.2f}s)")
            return TryOnResult(
                success=True,
                output_path=out_path,
                message=f"Generated with {model_name}",
                model_used=model_name,
                inference_time=elapsed,
            )

        except Exception as exc:
            traceback.print_exc()
            return TryOnResult(
                success=False,
                message=f"{model_name} failed: {exc}",
                model_used=model_name,
                inference_time=time.time() - start,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tryon_agent_sync(
    person_image_path: str,
    cloth_image_path: str,
    cloth_type: str = "upper",
    output_path: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> TryOnResult:
    """Run both try-on models and return the best result by pose similarity.

    Args:
        person_image_path: Path to the person image.
        cloth_image_path:  Path to the cloth/garment image.
        cloth_type:        Clothing type string.
                           CatVTON: "upper", "lower", "overall", "inner", "outer"
                           OOTDiffusion: "upperbody", "lowerbody", "dress"
        output_path:       Where to save the best result (auto-generated if None).
        device:            "cuda" or "cpu".
        **kwargs:          Extra model parameters (num_inference_steps, guidance_scale, seed).

    Returns:
        TryOnResult of the best model with pose_score set.
    """
    agent = TryOnAgent(device=device)
    result = agent.generate(
        person_image_path=person_image_path,
        cloth_image_path=cloth_image_path,
        cloth_type=cloth_type,
        output_path=output_path,
        **kwargs,
    )
    agent.unload()
    return result
