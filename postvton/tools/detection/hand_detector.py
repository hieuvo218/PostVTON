"""Hand distortion detection tool.

Primary backend: Google Gemini via google-genai.
Legacy fallback: GLM-4.6V via Hugging Face InferenceClient.

Provides HandDistortionDetector class for use by ProblemDetectionAgent.
"""

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass
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
    used_vlm_key: Optional[str] = None
    used_llm_key: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "distorted": self.distorted,
            "description": self.description,
            "analysis": self.reason,
            "used_vlm_key": self.used_vlm_key,
            "used_llm_key": self.used_llm_key,
            "error": self.error,
        }


_GEMINI_PROMPT = """
Analyze both hands and arms in the image.

Definition of "distorted":
- missing / extra / fused fingers
- abnormal length or angle
- unclear, blurred, covered, obscured, partially hidden
- blended with clothes or objects
- overlapping or clasped hands that obscure each other

Return ONLY JSON:

{{
    "distorted": true/false,
    "reason": "short reason",
    "hand_description": "brief description"
}}
""".strip()

# Legacy two-step prompt flow (HF GLM fallback)
_VLM_PROMPT = (
        "Describe both hands and arms in detail. "
        "If any hand is partially hidden, blurred, or unclear, mention it explicitly. "
        "In this task, 'distorted' means that the hand appears abnormal in shape or proportion -- "
        "for example, when fingers are missing, extra fingers, fused together, overly long or short, "
        "covered or blended with clothing, or have uneven texture, color, or boundary. "
        "DO NOT LIE, try to answer honestly."
)

_ANALYSIS_PROMPT_TEMPLATE = """
You are a highly specialized visual reasoning expert. Your only task is to analyze \
the preceding visual description of hands and set the 'distorted' flag to TRUE or FALSE.

Description:
{description}

IMPORTANT RULE:
- If the description states that the hands are hidden by the item they hold (e.g. handbags),
    or covered by strange objects like furs/sleeves -> set "distorted": true, ignore other criteria.
- If the description states the hand or fingers are hidden by dress, or the hand is extended
    downwards / palm facing inwards -> set "distorted": false, ignore distortion criteria unless
    something explicitly abnormal is described.

CRITERIA FOR "distorted": true
1. Missing, extra, fused, abnormally sized, or improperly angled fingers/limbs.
2. Hands are unnatural, unclear, blurred, covered, obscured, partially/mostly hidden, or blended.
3. Hands are overlapping, clasped, or holding each other so one hand obscures the other.

Respond ONLY in JSON format:
{{"distorted": true/false, "reason": "short analysis citing a phrase from the description"}}
"""


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

        # Prefer Gemini key names; keep HF_TOKEN for backward compatibility.
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

        # Prefer Gemini when available.
        if genai is not None and genai_types is not None:
            parsed, raw_text, err = self._detect_with_gemini(image_pil)
            if err:
                return HandDetectionResult(distorted=True, error=err)

            distorted = bool(parsed.get("distorted", False)) if isinstance(parsed, dict) else False
            reason = ""
            if isinstance(parsed, dict):
                reason = str(parsed.get("reason", "") or "")
            hand_desc = ""
            if isinstance(parsed, dict):
                hand_desc = str(parsed.get("hand_description", "") or "")

            return HandDetectionResult(
                distorted=distorted,
                description=hand_desc,
                reason=reason,
                used_llm_key="GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY",
            )

        # Legacy fallback (HF GLM) when google-genai isn't installed.
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
        reason = parsed.get("reason", "")

        return HandDetectionResult(
            distorted=distorted,
            description=description,
            reason=reason,
            used_vlm_key="HF_TOKEN" if os.environ.get("HF_TOKEN") else None,
            used_llm_key="HF_TOKEN" if os.environ.get("HF_TOKEN") else None,
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
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            result_text = getattr(response, "text", "") or ""
        except Exception as exc:
            return {}, "", f"Gemini call failed: {exc}"

        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, dict):
                return parsed, result_text, None
        except Exception:
            pass

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
        # The google-genai client is lightweight; keep one per detector.
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
        # JPEG is significantly smaller than PNG for photographic images,
        # which helps avoid 413 Payload Too Large from hosted inference APIs.
        image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _image_to_bytes(image: "Image.Image") -> bytes:
        buffer = io.BytesIO()
        image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return buffer.getvalue()

    @staticmethod
    def _coerce_image(image: "Image.Image") -> Tuple[Optional["Image.Image"], Optional[str]]:
        if Image is None:
            return None, "PIL is required for hand detection."
        if isinstance(image, Image.Image):
            return image.convert("RGB"), None
        return None, f"Expected PIL.Image.Image, got {type(image).__name__}"
