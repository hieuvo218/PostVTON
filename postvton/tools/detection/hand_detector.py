"""Hand distortion detection tool.

Primary backend: Google Gemini via google-genai.
Legacy fallback: GLM-4.6V via Hugging Face InferenceClient.

Provides HandDistortionDetector class for use by ProblemDetectionAgent.
"""

import base64
import io
import json
import logging
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional dependency
    genai = None
    genai_types = None

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - optional dependency
    InferenceClient = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from PIL import Image
except Exception:
    Image = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HandDetectionResult:
    """Result from hand distortion detection."""

    distorted: bool
    description: str = ""
    reason: str = ""
    observed_pose: str = "unknown"
    intended_pose: str = "unknown"
    missing_parts: list[str] = field(default_factory=list)
    edit_prompt: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "distorted": self.distorted,
            "description": self.description,
            "analysis": self.reason,
            "observed_pose": self.observed_pose,
            "intended_pose": self.intended_pose,
            "missing_parts": list(self.missing_parts),
            "edit_prompt": self.edit_prompt,
            "error": self.error,
        }


_GEMINI_PROMPT = """
Analyze the human hands in the image.

Tasks:
1. Detect the visible hand pose.
2. Infer the intended gesture if partially broken.
3. Determine whether the hand is distorted.

Gestures:
thumbs_up, peace_sign, pointing, open_palm,
fist, finger_heart, waving, holding_object, resting, unknown

Important rules:

- If a hand resembles a fist but appears to miss a thumb,
  infer intended_pose="thumbs_up".

- Do NOT hallucinate distortions.
  Resting or partially occluded fingers are normal.

- Only mark distorted=true when there is clear evidence:
  missing fingers,
  extra fingers,
  impossible anatomy,
  fused fingers,
  broken gesture structure.

- Mild blur, perspective, compression,
  or relaxed/resting hands are NOT distortions.

- Prefer semantic gesture understanding over raw shape.

Editing prompt rules:
- Describe ONLY the necessary repair.
- Preserve original pose and composition.
- Preserve clothing and identity.
- Keep prompt short and direct.
- Always ask the model to edit only the hand area, never the whole image.

Return ONLY valid JSON:

{
  "distorted": true or false,
  "observed_pose": "...",
  "intended_pose": "...",
  "missing_parts": [],
  "edit_prompt": "...",
  "reason": "..."
}
""".strip()




def _safe_json_from_text(text: str) -> dict:
    """Extract and parse the first JSON object from text."""
    json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        bracket_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if bracket_match:
            text = bracket_match.group(1)
    try:
        return json.loads(text)
    except Exception:
        distorted = "true" in text.lower()
        return {"distorted": distorted, "reason": text}


class HandDistortionDetector:
    """Detect hand distortions in virtual try-on images."""

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model_id: str = "gemini-3-flash-preview",
    ):
        if load_dotenv is not None:
            load_dotenv()

        env_token = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("HF_TOKEN")
        )
        self.token = env_token or (api_keys[0] if api_keys else None)
        if not self.token:
            raise ValueError(
                "Missing API key. Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY; "
                "HF_TOKEN is accepted only for the legacy HF backend."
            )
        self.model_id = model_id
        self._client = None

    def detect(self, image: "Image.Image") -> HandDetectionResult:
        """Detect hand distortion in a try-on image."""
        if Image is None:
            return HandDetectionResult(distorted=False, error="PIL is required for hand detection.")

        if not isinstance(image, Image.Image):
            return HandDetectionResult(
                distorted=False,
                error=f"Expected PIL.Image.Image, got {type(image).__name__}",
            )

        image_pil = image.convert("RGB")

        if genai is not None and genai_types is not None:
            parsed, raw_text, err = self._detect_with_gemini(image_pil)
            if err:
                return HandDetectionResult(distorted=True, error=err)
            return self._parsed_result(parsed, raw_text)

        description = self._describe(image_pil)
        if description is None:
            return HandDetectionResult(
                distorted=True,
                error="VLM description failed (Gemini unavailable; HF backend failed).",
            )

        parsed = self._analyse(description)
        if parsed is None:
            return HandDetectionResult(
                distorted=True,
                description=description,
                error="Analysis failed (Gemini unavailable; HF backend failed).",
            )

        distorted = bool(parsed.get("distorted", False))
        reason = str(parsed.get("reason", "") or "")

        return HandDetectionResult(
            distorted=distorted,
            description=description,
            reason=reason,
        )

    def detect_from_path(self, image_path: str) -> HandDetectionResult:
        """Detect hand distortion from an image path using Gemini when available."""
        if Image is None:
            return HandDetectionResult(distorted=False, error="PIL is required for hand detection.")

        if not os.path.exists(image_path):
            return HandDetectionResult(distorted=False, error=f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as image:
                image_pil = image.convert("RGB")
        except Exception as exc:
            return HandDetectionResult(distorted=False, error=f"Failed to load image: {exc}")

        if genai is not None and genai_types is not None:
            parsed, raw_text, err = self._detect_with_gemini_from_path(image_path)
            if err:
                return HandDetectionResult(distorted=True, error=err)
            return self._parsed_result(parsed, raw_text)

        return self.detect(image_pil)

    def _parsed_result(self, parsed: dict, raw_text: str) -> HandDetectionResult:
        distorted = bool(parsed.get("distorted", False)) if isinstance(parsed, dict) else False
        reason = ""
        observed_pose = "unknown"
        intended_pose = "unknown"
        missing_parts: list[str] = []
        edit_prompt = ""
        description = ""

        if isinstance(parsed, dict):
            reason = str(parsed.get("reason", "") or "")
            observed_pose = str(parsed.get("observed_pose", "unknown") or "unknown")
            intended_pose = str(parsed.get("intended_pose", "unknown") or "unknown")
            edit_prompt = str(parsed.get("edit_prompt", "") or "")
            description = str(parsed.get("hand_description", "") or "")
            if not description:
                description = reason or raw_text
            raw_missing = parsed.get("missing_parts", [])
            if isinstance(raw_missing, list):
                missing_parts = [str(item) for item in raw_missing]
            elif raw_missing:
                missing_parts = [str(raw_missing)]

        return HandDetectionResult(
            distorted=distorted,
            description=description,
            reason=reason,
            observed_pose=observed_pose,
            intended_pose=intended_pose,
            missing_parts=missing_parts,
            edit_prompt=edit_prompt,
        )

    def _detect_with_gemini(self, image: "Image.Image") -> Tuple[dict, str, Optional[str]]:
        """Return (parsed_json, raw_text, error)."""
        if genai is None or genai_types is None:
            return {}, "", "google-genai is not installed. Install it to use Gemini hand detection."

        try:
            image_bytes = self._image_to_bytes(image)
        except Exception as exc:
            return {}, "", f"Failed to encode image for Gemini: {exc}"

        try:
            client = self._get_gemini_client()
            response = client.models.generate_content(
                model=self.model_id,
                contents=[
                    genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    _GEMINI_PROMPT,
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            result_text = getattr(response, "text", "") or ""
        except Exception as exc:
            return {}, "", f"Gemini call failed: {exc}"

        parsed = _safe_json_from_text(result_text)
        if not isinstance(parsed, dict):
            return {}, result_text, "Gemini returned non-JSON response."
        return parsed, result_text, None

    def _detect_with_gemini_from_path(self, image_path: str) -> Tuple[dict, str, Optional[str]]:
        """Return (parsed_json, raw_text, error) for an image path."""
        if genai is None or genai_types is None:
            return {}, "", "google-genai is not installed. Install it to use Gemini hand detection."

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as exc:
            return {}, "", f"Failed to read image file: {exc}"

        try:
            client = self._get_gemini_client()
            response = client.models.generate_content(
                model=self.model_id,
                contents=[
                    genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    _GEMINI_PROMPT,
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            result_text = getattr(response, "text", "") or ""
        except Exception as exc:
            return {}, "", f"Gemini call failed: {exc}"

        parsed = _safe_json_from_text(result_text)
        if not isinstance(parsed, dict):
            return {}, result_text, "Gemini returned non-JSON response."
        return parsed, result_text, None

    def _describe(self, image: "Image.Image") -> Optional[str]:
        response = self._chat_with_image(_VLM_PROMPT, image)
        if response is None:
            return None
        return response.strip()

    def _analyse(self, description: str) -> Optional[dict]:
        prompt = _ANALYSIS_PROMPT_TEMPLATE.format(description=description)
        response = self._chat_with_text(prompt)
        if response is None:
            return None
        return _safe_json_from_text(response)

    def _chat_with_image(self, prompt: str, image: "Image.Image") -> Optional[str]:
        if InferenceClient is None:
            logger.error("huggingface_hub is not installed; cannot use legacy HF backend")
            return None
        client = self._get_client()
        data_url = self._image_to_data_url(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        try:
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
            )
        except Exception as exc:
            logger.error("VLM call failed: %s", exc)
            return None

        return self._extract_message_text(completion)

    def _chat_with_text(self, prompt: str) -> Optional[str]:
        if InferenceClient is None:
            logger.error("huggingface_hub is not installed; cannot use legacy HF backend")
            return None
        client = self._get_client()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
            )
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return None

        return self._extract_message_text(completion)

    def _get_client(self):
        if self._client is None:
            self._client = InferenceClient(api_key=self.token)
        return self._client

    def _get_gemini_client(self):
        if self._client is None or not hasattr(self._client, "models"):
            self._client = genai.Client(api_key=self.token)
        return self._client

    @staticmethod
    def _extract_message_text(completion: Any) -> str:
        try:
            message = completion.choices[0].message
            if isinstance(message.content, str):
                return message.content
            return str(message.content)
        except Exception:
            return str(completion)

    @staticmethod
    def _image_to_data_url(image: "Image.Image") -> str:
        buffer = io.BytesIO()
        image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _image_to_bytes(image: "Image.Image") -> bytes:
        image = image.convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            image.save(tmp, format="JPEG", quality=85, optimize=True)
            tmp.flush()
            with open(tmp.name, "rb") as f:
                return f.read()

    @staticmethod
    def _coerce_image(image: "Image.Image") -> Tuple[Optional["Image.Image"], Optional[str]]:
        if Image is None:
            return None, "PIL is required for hand detection."
        if isinstance(image, Image.Image):
            return image.convert("RGB"), None
        return None, f"Expected PIL.Image.Image, got {type(image).__name__}"
