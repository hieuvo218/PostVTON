"""Hand distortion detection tool using Gemini VLM.

Provides HandDistortionDetector class for use by ProblemDetectionAgent,
and a standalone detect_hand_distortion() function for direct use.
"""

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai


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


# ---------------------------------------------------------------------------
# Internal key-rotation helper
# ---------------------------------------------------------------------------

_VLM_PROMPT = (
    "Describe both hands and arms in detail. "
    "If any hand is partially hidden, blurred, or unclear, mention it explicitly. "
    "In this task, 'distorted' means that the hand appears abnormal in shape or proportion â€” "
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


def _call_with_key_rotation(
    api_keys: List[str],
    build_payload,
    max_retries_per_key: int,
    max_total_retries: int,
    label: str = "VLM",
):
    """Try ``build_payload()`` against each api_key in rotation.

    Returns (genai response, key_used) or (None, None) on total failure.
    """
    rate_limited: dict = {}   # key -> cooldown-until datetime
    banned: set = set()

    for loop in range(max_total_retries):
        print(f"[{label}] Retry loop {loop + 1}/{max_total_retries}")
        random.shuffle(api_keys)

        for key in api_keys:
            if key in banned:
                continue
            if key in rate_limited and datetime.now() < rate_limited[key]:
                continue

            print(f"[{label}] Trying key {key[:12]}...")
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            payload = build_payload()

            for attempt in range(max_retries_per_key):
                try:
                    resp = model.generate_content(
                        contents=payload,
                        generation_config={"temperature": 0.5},
                    )
                    print(f"[{label}] Success on key {key[:12]}")
                    return resp, key
                except Exception as exc:
                    err = str(exc)
                    print(f"[{label}] Error: {err}")
                    if "429" in err or "rate" in err.lower():
                        rate_limited[key] = datetime.now() + timedelta(seconds=40)
                        break
                    if "permission" in err.lower() or "auth" in err.lower():
                        banned.add(key)
                        break
                    time.sleep(0.5)

        time.sleep(1.0)

    print(f"[{label}] All keys exhausted.")
    return None, None


def _force_parse_json(text: str) -> dict:
    """Extract and parse the first JSON object from ``text``."""
    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        bracket_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if bracket_match:
            text = bracket_match.group(1)
    try:
        return json.loads(text)
    except Exception:
        distorted = "true" in text.lower()
        return {"distorted": distorted, "reason": text}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HandDistortionDetector:
    """Detect hand distortions in virtual try-on images using Gemini VLM.

    Intended to be used as a tool by ProblemDetectionAgent.

    Example::

        detector = HandDistortionDetector(api_keys=[...])
        result = detector.detect("path/to/tryon.jpg")
        if result.distorted:
            print("Hand distortion detected:", result.reason)
    """

    def __init__(
        self,
        api_keys: List[str],
        max_retries_per_key: int = 2,
        max_total_retries: int = 2,
    ):
        if not api_keys:
            raise ValueError("At least one Gemini API key is required.")
        self.api_keys = list(api_keys)
        self.max_retries_per_key = max_retries_per_key
        self.max_total_retries = max_total_retries

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> HandDetectionResult:
        """Detect hand distortion in a try-on image.

        Args:
            image_path: Path to the try-on image (JPEG/PNG).

        Returns:
            HandDetectionResult with distorted flag, description, and reason.
        """
        path = Path(image_path)
        if not path.exists():
            return HandDetectionResult(
                distorted=True,
                error=f"Image not found: {image_path}",
            )

        print(f"[HandDetector] Analysing: {path.name}")
        image_bytes = path.read_bytes()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

        # --- Stage 1: VLM description ---
        description, vlm_key = self._describe(image_bytes, mime)
        if description is None:
            return HandDetectionResult(
                distorted=True,
                error="VLM description failed: all API keys exhausted.",
            )

        print(f"[HandDetector] Description:\n{description}")

        # --- Stage 2: Reasoning / classification ---
        parsed, llm_key = self._analyse(description)
        if parsed is None:
            return HandDetectionResult(
                distorted=True,
                description=description,
                error="Analysis failed: all API keys exhausted.",
                used_vlm_key=vlm_key,
            )

        distorted = bool(parsed.get("distorted", False))
        reason = parsed.get("reason", "")
        print(f"[HandDetector] distorted={distorted}  reason={reason}")

        return HandDetectionResult(
            distorted=distorted,
            description=description,
            reason=reason,
            used_vlm_key=vlm_key,
            used_llm_key=llm_key,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _describe(self, image_bytes: bytes, mime: str):
        """Call VLM to get a textual description of the hands."""
        def build():
            return [
                {"mime_type": mime, "data": image_bytes},
                {"text": _VLM_PROMPT},
            ]

        resp, key = _call_with_key_rotation(
            self.api_keys, build,
            self.max_retries_per_key, self.max_total_retries,
            label="HAND-DESC",
        )
        if resp is None:
            return None, None
        return (resp.text or "").strip(), key

    def _analyse(self, description: str):
        """Call LLM to classify whether the description indicates distortion."""
        prompt = _ANALYSIS_PROMPT_TEMPLATE.format(description=description)

        def build():
            return prompt

        resp, key = _call_with_key_rotation(
            self.api_keys, build,
            self.max_retries_per_key, self.max_total_retries,
            label="HAND-ANALYSIS",
        )
        if resp is None:
            return None, None
        parsed = _force_parse_json((resp.text or "").strip())
        return parsed, key


# ---------------------------------------------------------------------------
# Functional wrapper (backward-compatible)
# ---------------------------------------------------------------------------

def detect_hand_distortion(
    tryon_path: str,
    api_keys: List[str],
    max_retries_per_key: int = 2,
    max_total_retries: int = 2,
) -> dict:
    """Detect hand distortion in a try-on image.

    Thin wrapper around HandDistortionDetector for backward compatibility.

    Returns a dict with keys: distorted, description, analysis,
    used_vlm_key, used_llm_key.
    """
    detector = HandDistortionDetector(
        api_keys=api_keys,
        max_retries_per_key=max_retries_per_key,
        max_total_retries=max_total_retries,
    )
    return detector.detect(tryon_path).to_dict()
