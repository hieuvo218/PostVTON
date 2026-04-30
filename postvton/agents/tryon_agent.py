"""TryOn agent client.

This repo no longer runs CatVTON/OOTDiffusion locally. Try-on is performed by a
remote FastAPI server (see postvton.tryon_server).
"""

import sys
import uuid
import time
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    import requests
except Exception:
    requests = None

from urllib.parse import urljoin

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class TryOnResult:
    """Result from a try-on operation"""
    success: bool
    output_path: Optional[str] = None
    message: str = ""
    model_used: Optional[str] = None
    inference_time: float = 0.0
    pose_score: float = 0.0   # cosine pose similarity vs. person image (0.0 = not yet scored)


class TryOnAgent:
    """Client that calls the remote try-on server."""

    def __init__(self, device: str = "cuda", server_url: Optional[str] = None, timeout_s: int = 300):
        self.device = device
        # server_url is required for try-on; allow env-var fallback.
        env_url = os.environ.get("TRYON_SERVER_URL")
        self.server_url = (server_url or env_url or "").strip() or None
        self.timeout_s = int(timeout_s)

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
        """Call the remote try-on server and return a result.

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
        if not self.server_url:
            return TryOnResult(
                success=False,
                message=(
                    "Remote try-on server URL is not set. "
                    "Pass --tryon-server-url or set TRYON_SERVER_URL."
                ),
                model_used="remote",
            )

        return self._generate_remote(
            person_image_path=person_image_path,
            cloth_image_path=cloth_image_path,
            cloth_type=cloth_type,
            output_path=output_path,
            **kwargs,
        )

    def _generate_remote(
        self,
        person_image_path: str,
        cloth_image_path: str,
        cloth_type: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> TryOnResult:
        if requests is None:
            return TryOnResult(
                success=False,
                message="Remote try-on requires 'requests' (pip install requests).",
                model_used="remote",
            )

        start = time.time()
        server_url = self.server_url.rstrip("/") + "/"
        tryon_url = urljoin(server_url, "tryon")
        try:
            person_p = Path(person_image_path)
            cloth_p = Path(cloth_image_path)
            if not person_p.exists():
                raise FileNotFoundError(f"Person image not found: {person_p}")
            if not cloth_p.exists():
                raise FileNotFoundError(f"Cloth image not found: {cloth_p}")

            with open(person_image_path, "rb") as f_person, open(cloth_image_path, "rb") as f_cloth:
                files = {
                    "person_image": (Path(person_image_path).name, f_person, "application/octet-stream"),
                    "cloth_image": (Path(cloth_image_path).name, f_cloth, "application/octet-stream"),
                }
                data = {
                    "cloth_type": cloth_type,
                    "num_inference_steps": int(kwargs.get("num_inference_steps", 10)),
                    "guidance_scale": float(kwargs.get("guidance_scale", 2.5)),
                    "seed": int(kwargs.get("seed", -1) if kwargs.get("seed", -1) is not None else -1),
                }
                r = requests.post(tryon_url, data=data, files=files, timeout=self.timeout_s)
            r.raise_for_status()
            payload = r.json()
            if not payload.get("success", False):
                raise RuntimeError(f"Try-on server returned success=false: {payload}")

            output_url = payload.get("output_url")
            if not output_url:
                raise RuntimeError(f"Try-on server response missing output_url: {payload}")

            download_url = urljoin(server_url, output_url.lstrip("/"))
            img_resp = requests.get(download_url, timeout=self.timeout_s)
            img_resp.raise_for_status()

            if output_path is None:
                out_dir = project_root / "outputs" / "tryon_agent"
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(out_dir / f"remote_{uuid.uuid4().hex[:8]}.png")
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f_out:
                f_out.write(img_resp.content)

            elapsed = time.time() - start
            server_time = float(payload.get("inference_time") or 0.0)
            pose_score = float(payload.get("pose_score") or 0.0)
            model_used = payload.get("model_used") or "remote"

            return TryOnResult(
                success=True,
                output_path=output_path,
                message=f"Generated via try-on server: {tryon_url}",
                model_used=str(model_used),
                inference_time=server_time if server_time > 0 else elapsed,
                pose_score=pose_score,
            )
        except Exception as exc:
            return TryOnResult(
                success=False,
                message=f"Remote try-on failed: {exc}",
                model_used="remote",
                inference_time=time.time() - start,
            )

    def unload(self):
        """No-op (models run remotely)."""
        return


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
    return result
