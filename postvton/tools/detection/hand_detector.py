"""Hand distortion detection tool using GLM-4.6V via Hugging Face.

Provides HandDistortionDetector class for use by ProblemDetectionAgent.
"""

import base64
import io
import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from huggingface_hub import InferenceClient

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
{"distorted": true/false, "reason": "short analysis citing a phrase from the description"}
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
    """Detect hand distortions in virtual try-on images using GLM-4.6V."""

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model_id: str = "zai-org/GLM-4.6V",
    ):
        if load_dotenv is not None:
            load_dotenv()
        env_token = os.environ.get("HF_TOKEN")
        self.token = env_token or (api_keys[0] if api_keys else None)
        if not self.token:
            raise ValueError("HF_TOKEN is required for GLM-4.6V inference.")
        self.model_id = model_id
        self._client: Optional[InferenceClient] = None

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

        description = self._describe(image_pil)
        if description is None:
            return HandDetectionResult(
                distorted=True,
                error="VLM description failed.",
            )

        parsed = self._analyse(description)
        if parsed is None:
            return HandDetectionResult(
                distorted=True,
                description=description,
                error="Analysis failed.",
            )

        distorted = bool(parsed.get("distorted", False))
        reason = parsed.get("reason", "")

        return HandDetectionResult(
            distorted=distorted,
            description=description,
            reason=reason,
        )

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
        return self._chat_completion(messages, error_label="VLM")

    def _chat_with_text(self, prompt: str) -> Optional[str]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._chat_completion(messages, error_label="LLM")

    def _chat_completion(self, messages: list, error_label: str) -> Optional[str]:
        """Call chat completions on Hugging Face.

        HF has changed behavior over time; the per-model endpoint
        `/models/<id>/v1/chat/completions` may 404 on the hosted API.
        We therefore try the OpenAI-compatible router endpoint first.
        """

        # 1) Preferred: HF router OpenAI-compatible endpoint
        router_url = os.environ.get(
            "HF_CHAT_COMPLETIONS_URL",
            "https://api-inference.huggingface.co/v1/chat/completions",
        )
        try:
            text = self._chat_completion_via_router(router_url, messages)
            if text is not None:
                return text
        except Exception as exc:
            # Router failures fall back to InferenceClient below.
            logger.warning("%s router call failed: %s", error_label, exc)

        # 2) Fallback: huggingface_hub InferenceClient (may work on some setups)
        try:
            client = self._get_client()
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
            )
            return self._extract_message_text(completion)
        except Exception as exc:
            logger.error("%s call failed: %s", error_label, exc)
            return None

    def _chat_completion_via_router(self, url: str, messages: list) -> Optional[str]:
        payload = {
            "model": self.model_id,
            "messages": messages,
        }
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise RuntimeError(f"HTTP {exc.code} from HF chat completions: {body or exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach HF chat completions endpoint: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Non-JSON response from HF chat completions: {raw[:2000]}") from exc

        try:
            return parsed["choices"][0]["message"]["content"]
        except Exception:
            # Return full JSON for debugging when schema differs.
            return json.dumps(parsed)

    def _get_client(self) -> InferenceClient:
        if self._client is None:
            self._client = InferenceClient(api_key=self.token)
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
    def _coerce_image(image: "Image.Image") -> Tuple[Optional["Image.Image"], Optional[str]]:
        if Image is None:
            return None, "PIL is required for hand detection."
        if isinstance(image, Image.Image):
            return image.convert("RGB"), None
        return None, f"Expected PIL.Image.Image, got {type(image).__name__}"
