"""FastAPI try-on server.

Endpoints:
- GET  /health
- POST /tryon (multipart: person_image, cloth_image, cloth_type, params...)
- GET  /outputs/{name}

The server runs CatVTON and OOTDiffusion, scores outputs by MediaPipe pose
similarity to the original person image, saves the best image, and returns a
download URL.
"""

from __future__ import annotations

import io
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

try:
	import mediapipe as mp
except Exception as exc:  # pragma: no cover
	raise ImportError("mediapipe is required to run the try-on server") from exc

try:
	from sklearn.metrics.pairwise import cosine_similarity
except Exception as exc:  # pragma: no cover
	raise ImportError("scikit-learn is required to run the try-on server") from exc


app = FastAPI(title="PostVTON Try-On Server", version="0.1.0")


def _get_output_dir() -> Path:
	root = Path(os.environ.get("TRYON_SERVER_OUTPUT_DIR", "outputs"))
	out_dir = root / "tryon_server"
	out_dir.mkdir(parents=True, exist_ok=True)
	return out_dir


_OUTPUT_DIR = _get_output_dir()
_INFER_LOCK = threading.Lock()


_mp_pose = mp.solutions.pose


def _pil_from_upload(upload: UploadFile) -> Image.Image:
	try:
		data = upload.file.read()
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f"Failed to read upload '{upload.filename}': {exc}")
	try:
		return Image.open(io.BytesIO(data)).convert("RGB")
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f"Invalid image upload '{upload.filename}': {exc}")


def _extract_pose_keypoints(image: Image.Image) -> Optional[List[Tuple[float, float, float]]]:
	"""Return list of (x,y,visibility) for pose landmarks, or None."""
	rgb = np.array(image.convert("RGB"))
	with _mp_pose.Pose(static_image_mode=True) as pose_estimator:
		results = pose_estimator.process(rgb)
	if not results.pose_landmarks:
		return None
	return [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]


def _pose_to_vector(keypoints: List[Tuple[float, float, float]], visibility_threshold: float = 0.6) -> List[float]:
	return [coord for (x, y, conf) in keypoints if conf > visibility_threshold for coord in (x, y)]


def _cosine_score(pose_a: List[Tuple[float, float, float]], pose_b: List[Tuple[float, float, float]]) -> float:
	vec_a = _pose_to_vector(pose_a)
	vec_b = _pose_to_vector(pose_b)
	min_len = min(len(vec_a), len(vec_b))
	if min_len == 0:
		return 0.0
	score = float(cosine_similarity([vec_a[:min_len]], [vec_b[:min_len]])[0][0])
	return max(0.0, min(1.0, score))


def score_pose_similarity(person: Image.Image, tryon: Image.Image) -> float:
	pose_person = _extract_pose_keypoints(person)
	pose_tryon = _extract_pose_keypoints(tryon)
	if pose_person is None or pose_tryon is None:
		return 0.0
	return _cosine_score(pose_person, pose_tryon)


def _map_cloth_type_for_ootdiffusion(cloth_type: str) -> str:
	t = (cloth_type or "").strip().lower()
	if t in ("upper", "upperbody"):
		return "upperbody"
	if t in ("lower", "lowerbody"):
		return "lowerbody"
	if t in ("overall", "dress"):
		return "dress"
	# Default to upperbody for unknown values
	return "upperbody"


@dataclass
class _TryOnCandidate:
	model_used: str
	image: Optional[Image.Image] = None
	pose_score: float = 0.0
	error: Optional[str] = None
	inference_time: float = 0.0

	def to_dict(self) -> dict:
		return {
			"model_used": self.model_used,
			"pose_score": float(self.pose_score),
			"inference_time": float(self.inference_time),
			"error": self.error,
		}


class TryOnService:
	"""Lazy-loaded service for CatVTON and OOTDiffusion."""

	def __init__(self, device: str = "cuda"):
		self.device = device
		self._catvton = None
		self._ootd_hd = None
		self._ootd_dc = None

	def _get_catvton(self):
		if self._catvton is None:
			from postvton.tools.tryon.catvton import CatVTONInference

			self._catvton = CatVTONInference(device=self.device)
		return self._catvton

	def _get_ootd(self, model_type: str):
		model_type = model_type.lower()
		if model_type == "hd":
			if self._ootd_hd is None:
				from postvton.tools.tryon.ootdiffusion import OOTDiffusionInference

				gpu_id = 0 if str(self.device).startswith("cuda") else -1
				self._ootd_hd = OOTDiffusionInference(model_type="hd", gpu_id=gpu_id)
			return self._ootd_hd
		if model_type == "dc":
			if self._ootd_dc is None:
				from postvton.tools.tryon.ootdiffusion import OOTDiffusionInference

				gpu_id = 0 if str(self.device).startswith("cuda") else -1
				self._ootd_dc = OOTDiffusionInference(model_type="dc", gpu_id=gpu_id)
			return self._ootd_dc
		raise ValueError("model_type must be 'hd' or 'dc'")

	def run(
		self,
		person: Image.Image,
		cloth: Image.Image,
		cloth_type: str,
		num_inference_steps: int = 10,
		guidance_scale: float = 2.5,
		seed: int = -1,
	) -> Tuple[Optional[_TryOnCandidate], List[_TryOnCandidate]]:
		candidates: List[_TryOnCandidate] = []

		# CatVTON
		start = time.time()
		try:
			cat = self._get_catvton()
			img = cat.generate(
				person_image=person,
				cloth_image=cloth,
				cloth_type=cloth_type,
				num_inference_steps=num_inference_steps,
				guidance_scale=guidance_scale,
				seed=None if seed == -1 else seed,
			)
			score = score_pose_similarity(person, img)
			candidates.append(
				_TryOnCandidate(
					model_used="catvton",
					image=img,
					pose_score=score,
					inference_time=time.time() - start,
				)
			)
		except Exception as exc:
			candidates.append(
				_TryOnCandidate(
					model_used="catvton",
					error=str(exc),
					inference_time=time.time() - start,
				)
			)

		# OOTDiffusion
		start = time.time()
		try:
			oot_category = _map_cloth_type_for_ootdiffusion(cloth_type)
			oot_mode = "hd" if oot_category == "upperbody" else "dc"
			oot = self._get_ootd(oot_mode)
			img = oot.generate(
				person_image=person,
				cloth_image=cloth,
				category=oot_category,
				num_inference_steps=num_inference_steps,
				guidance_scale=guidance_scale if oot_mode == "dc" else 2.0,
				num_samples=1,
				seed=seed,
			)
			if isinstance(img, (list, tuple)):
				img = img[0]
			score = score_pose_similarity(person, img)
			candidates.append(
				_TryOnCandidate(
					model_used=f"ootdiffusion-{oot_mode}",
					image=img,
					pose_score=score,
					inference_time=time.time() - start,
				)
			)
		except Exception as exc:
			candidates.append(
				_TryOnCandidate(
					model_used="ootdiffusion",
					error=str(exc),
					inference_time=time.time() - start,
				)
			)

		ok = [c for c in candidates if c.image is not None]
		if not ok:
			return None, candidates

		best = max(ok, key=lambda c: float(c.pose_score))
		return best, candidates


_SERVICE = TryOnService(device=os.environ.get("TRYON_SERVER_DEVICE", "cuda"))


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.post("/tryon")
def tryon(
	cloth_type: str = Form("upper"),
	num_inference_steps: int = Form(10),
	guidance_scale: float = Form(2.5),
	seed: int = Form(-1),
	person_image: UploadFile = File(...),
	cloth_image: UploadFile = File(...),
) -> Dict[str, Any]:
	person = _pil_from_upload(person_image)
	cloth = _pil_from_upload(cloth_image)

	job_id = uuid.uuid4().hex[:12]
	filename = f"{job_id}.png"
	out_path = _OUTPUT_DIR / filename

	with _INFER_LOCK:
		best, candidates = _SERVICE.run(
			person=person,
			cloth=cloth,
			cloth_type=cloth_type,
			num_inference_steps=num_inference_steps,
			guidance_scale=guidance_scale,
			seed=seed,
		)

	if best is None or best.image is None:
		errors = [c.to_dict() for c in candidates]
		raise HTTPException(status_code=500, detail={"message": "Both models failed", "details": errors})

	best.image.save(out_path)

	return {
		"success": True,
		"job_id": job_id,
		"model_used": best.model_used,
		"pose_score": float(best.pose_score),
		"inference_time": float(best.inference_time),
		"output_url": f"/outputs/{filename}",
		"candidates": [c.to_dict() for c in candidates],
	}


@app.get("/outputs/{name}")
def outputs(name: str):
	path = _OUTPUT_DIR / name
	if not path.exists():
		raise HTTPException(status_code=404, detail="Output not found")
	return FileResponse(path)
