"""Tests for ProblemDetectionAgent and its underlying detection tools.

Controls:
    1.  Imports and class instantiation — lazy detectors start as None.
    2.  Report helpers with accessories/hands schema.
    3.  detect() gracefully handles a missing image file.
    4.  detect() without original_image returns an error report.
    5.  detect() two-image mode populates accessories missing/details.
    6.  detect_problems() signature includes original_image.
    7.  MissingAccessoryDetector import and public interface.
    8.  MissingAccessoryResult shape via to_dict().
    9.  detect_from_tryon_result() with a failed TryOnResult.
    10. detect_from_tryon_result() with a successful TryOnResult (bad key, no crash).
    11. HandDetectionResult.to_dict() shape.
    12. HandDistortionDetector instantiation and public interface.
    13. detect() bad API key does NOT crash the agent.
    14. Live detect() two-image mode with real HF_TOKEN
        (skipped automatically when HF_TOKEN env var is not set).

Run from the project root:
    python tests/test_detection_agent.py

For live test (test 14), set the environment variable first:
    $env:HF_TOKEN = "your-token-here"
    python tests/test_detection_agent.py
"""

import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_IMAGE = str(
    PROJECT_ROOT
    / "external/catvton/resource/demo/example/person/women/049713_0.jpg"
)

TRYON_IMAGE = str(
    PROJECT_ROOT
    / "outputs/test_tryon_agent/best_result.png"
)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def _sample_exists() -> bool:
    return Path(SAMPLE_IMAGE).exists()


def _tryon_exists() -> bool:
    return Path(TRYON_IMAGE).exists()


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Test 1 — imports and instantiation
# ---------------------------------------------------------------------------

def test_imports():
    print("\n[TEST 1] Imports and class instantiation")
    try:
        from postvton.agents.problem_detection_agent import (
            ProblemDetectionAgent,
            ProblemDetectionReport,
            detect_problems,
        )
        from postvton.tools.detection.hand_detector import (
            HandDistortionDetector,
            HandDetectionResult,
        )
        from postvton.tools.detection.missing_accessory_detector import (
            MissingAccessoryDetector,
            MissingAccessoryResult,
        )

        agent = ProblemDetectionAgent(api_keys=["dummy-key"])
        assert agent._hand_detector is None, "_hand_detector should be lazy"
        assert agent._missing_accessory_detector is None, "_missing_accessory_detector should be lazy"
        assert hasattr(agent, "_get_hand_detector")
        assert hasattr(agent, "_get_missing_accessory_detector")

        print(f"{PASS} All imports OK; agent instantiation correct")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 2 — dataclass helpers
# ---------------------------------------------------------------------------

def test_dataclass_helpers():
    print("\n[TEST 2] Report helpers and schema")
    try:
        from postvton.agents.problem_detection_agent import (
            AccessoriesReport,
            HandsReport,
            ProblemDetectionReport,
        )

        accessories = AccessoriesReport(missing=False, details=[])
        hands = HandsReport(distorted=False, analysis="")

        clean = ProblemDetectionReport(
            image_id="img.png",
            accessories=accessories,
            hands=hands,
        )
        d = clean.to_dict()
        assert d["accessories"]["missing"] is False
        assert d["accessories"]["details"] == []
        assert d["hands"]["distorted"] is False
        assert d["hands"]["analysis"] == ""
        assert "No problems" in clean.summary()

        dirty = ProblemDetectionReport(
            image_id="img.png",
            accessories=AccessoriesReport(missing=True, details=[{"class": "watch", "count": 1}]),
            hands=HandsReport(distorted=True, analysis="Fingers fused."),
        )
        d2 = dirty.to_dict()
        assert d2["accessories"]["missing"] is True
        assert d2["hands"]["distorted"] is True
        assert "Problems detected" in dirty.summary()

        err = ProblemDetectionReport(
            image_id="img.png",
            accessories=accessories,
            hands=hands,
            error="oops",
        )
        assert "oops" in err.summary()

        print(f"{PASS} Dataclass helpers correct")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 3 — graceful missing-file handling
# ---------------------------------------------------------------------------

def test_missing_file():
    print("\n[TEST 3] detect() on a non-image input")
    try:
        from postvton.agents.problem_detection_agent import ProblemDetectionAgent

        agent = ProblemDetectionAgent(api_keys=["dummy-key"])
        report = agent.detect("/nonexistent/image.png")

        assert report.error is not None
        assert "pil" in report.error.lower()
        assert report.accessories.missing is False
        assert report.hands.distorted is False
        print(f"{PASS} Missing file → error report: '{report.error}'")
        return True
    except Exception as exc:
        print(f"{FAIL} Unexpected exception: {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 4 — missing original_image
# ---------------------------------------------------------------------------

def test_missing_original_image_path():
    print("\n[TEST 4] detect() without original_image returns error report")
    if not _sample_exists():
        print(f"{SKIP} Sample image not found — skipping")
        return True
    try:
        from postvton.agents.problem_detection_agent import ProblemDetectionAgent

        agent = ProblemDetectionAgent(
            api_keys=["INVALID"],
            max_retries_per_key=1,
            max_total_retries=1,
        )
        report = agent.detect(_load_image(SAMPLE_IMAGE))

        assert report.error is not None
        assert "original_image is required" in report.error
        assert report.accessories.details == []
        assert report.accessories.missing is False

        print(f"{PASS} Missing original_image returns report error: {report.error}")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 5 — two-image mode (with original_image)
# ---------------------------------------------------------------------------

def test_two_image_mode():
    print("\n[TEST 5] detect() two-image mode — MissingAccessoryResult shape")
    if not _sample_exists():
        print(f"{SKIP} Sample image not found — skipping")
        return True
    try:
        from postvton.agents.problem_detection_agent import ProblemDetectionAgent

        agent = ProblemDetectionAgent(
            api_keys=["INVALID"],
            max_retries_per_key=1,
            max_total_retries=1,
        )
        # Use same image for both → missing should be empty
        sample = _load_image(SAMPLE_IMAGE)
        report = agent.detect(sample, original_image=sample)

        assert report.accessories.missing is False
        assert report.accessories.details == []

        print(f"{PASS} Two-image mode: has_missing={report.accessories.missing}")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 6 — detect_problems() signature
# ---------------------------------------------------------------------------

def test_detect_problems_signature():
    print("\n[TEST 6] detect_problems() signature includes original_image")
    try:
        from postvton.agents.problem_detection_agent import detect_problems

        sig = inspect.signature(detect_problems)
        params = list(sig.parameters)
        for expected in ("image", "api_keys", "original_image"):
            assert expected in params, f"Missing param '{expected}' in detect_problems()"

        print(f"{PASS} detect_problems() signature: {params}")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 7 — MissingAccessoryDetector interface
# ---------------------------------------------------------------------------

def test_missing_accessory_detector_interface():
    print("\n[TEST 7] MissingAccessoryDetector import and public interface")
    try:
        from postvton.tools.detection.missing_accessory_detector import (
            MissingAccessoryDetector,
            MissingAccessoryResult,
            AccessoryDetectionResult,
            detect_missing_accessories,
        )

        det = MissingAccessoryDetector()
        assert callable(getattr(det, "detect_accessories", None)), "Missing .detect_accessories()"
        assert callable(getattr(det, "detect_missing", None)), "Missing .detect_missing()"

        sig_missing = inspect.signature(det.detect_missing)
        assert "original_image" in sig_missing.parameters
        assert "tryon_image" in sig_missing.parameters

        sig_fn = inspect.signature(detect_missing_accessories)
        assert "original_image" in sig_fn.parameters
        assert "tryon_image" in sig_fn.parameters

        print(f"{PASS} MissingAccessoryDetector interface correct")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 8 — MissingAccessoryResult.to_dict() shape
# ---------------------------------------------------------------------------

def test_missing_accessory_result_shape():
    print("\n[TEST 8] MissingAccessoryResult.to_dict() shape")
    try:
        from postvton.tools.detection.missing_accessory_detector import (
            MissingAccessoryResult,
            AccessoryDetectionResult,
        )

        orig = AccessoryDetectionResult(image_id="orig", image_size=(640, 480))
        tryon = AccessoryDetectionResult(image_id="tryon", image_size=(640, 480))

        # No missing
        r_clean = MissingAccessoryResult(
            original_detection=orig,
            tryon_detection=tryon,
            missing_by_label={},
        )
        assert r_clean.has_missing is False
        assert r_clean.total_missing == 0
        assert r_clean.missing_labels == []
        assert r_clean.error is None

        # Some missing
        r_missing = MissingAccessoryResult(
            original_detection=orig,
            tryon_detection=tryon,
            missing_by_label={"watch": 1, "bracelet": 2},
        )
        assert r_missing.has_missing is True
        assert r_missing.total_missing == 3
        assert sorted(r_missing.missing_labels) == ["bracelet", "watch"]

        d = r_missing.to_dict()
        for key in ("has_missing", "total_missing", "missing_labels", "missing_by_label",
                    "original_detection", "tryon_detection", "error"):
            assert key in d, f"Missing key '{key}' in to_dict()"
        assert d["missing_by_label"] == {"watch": 1, "bracelet": 2}

        print(f"{PASS} MissingAccessoryResult shape correct")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 9 — detect_from_tryon_result() failed TryOnResult
# ---------------------------------------------------------------------------

def test_detect_from_failed_tryon_result():
    print("\n[TEST 9] detect_from_tryon_result() with failed TryOnResult")
    try:
        from postvton.agents.problem_detection_agent import ProblemDetectionAgent

        @dataclass
        class FakeTryOnResult:
            success: bool
            output_image: Optional[Image.Image] = None

        agent = ProblemDetectionAgent(api_keys=["dummy-key"])
        report = agent.detect_from_tryon_result(FakeTryOnResult(success=False))

        assert report.error is not None
        assert "no valid output" in report.error.lower()
        print(f"{PASS} Failed TryOnResult → error report: '{report.error}'")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 10 — detect_from_tryon_result() successful TryOnResult (bad key)
# ---------------------------------------------------------------------------

def test_detect_from_successful_tryon_result():
    print("\n[TEST 10] detect_from_tryon_result() with valid output path (bad key, no crash)")
    if not _sample_exists():
        print(f"{SKIP} Sample image not found — skipping")
        return True
    try:
        from postvton.agents.problem_detection_agent import ProblemDetectionAgent

        @dataclass
        class FakeTryOnResult:
            success: bool
            output_image: Optional[Image.Image]
            original_image: Optional[Image.Image] = None

        agent = ProblemDetectionAgent(
            api_keys=["INVALID"],
            max_retries_per_key=1,
            max_total_retries=1,
        )
        report = agent.detect_from_tryon_result(
            FakeTryOnResult(
                success=True,
                output_image=_load_image(SAMPLE_IMAGE),
                original_image=_load_image(SAMPLE_IMAGE),
            )
        )
        assert report.image_id
        assert report.accessories is not None
        assert isinstance(report.accessories.missing, bool)
        print(f"{PASS} detect_from_tryon_result OK — accessories.missing={report.accessories.missing}")
        return True
    except Exception as exc:
        print(f"{FAIL} Unexpected exception: {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 11 — HandDetectionResult.to_dict() shape
# ---------------------------------------------------------------------------

def test_hand_detection_result_shape():
    print("\n[TEST 11] HandDetectionResult.to_dict() shape")
    try:
        from postvton.tools.detection.hand_detector import HandDetectionResult

        r = HandDetectionResult(
            distorted=True,
            description="Fingers look fused.",
            reason="Two fingers merged.",
            used_vlm_key="key1",
            used_llm_key="key2",
            error=None,
        )
        d = r.to_dict()
        for key in ("distorted", "description", "analysis", "used_vlm_key", "used_llm_key", "error"):
            assert key in d, f"Missing key '{key}' in to_dict()"
        assert d["distorted"] is True
        assert d["analysis"] == "Two fingers merged."

        r2 = HandDetectionResult(distorted=False, error="timeout")
        d2 = r2.to_dict()
        assert d2["distorted"] is False
        assert d2["error"] == "timeout"

        print(f"{PASS} HandDetectionResult.to_dict() shape correct")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 12 — HandDistortionDetector public interface
# ---------------------------------------------------------------------------

def test_hand_detector_interface():
    print("\n[TEST 12] HandDistortionDetector interface")
    try:
        from postvton.tools.detection.hand_detector import HandDistortionDetector

        det = HandDistortionDetector(api_keys=["dummy-key"])
        assert callable(getattr(det, "detect", None)), "Missing .detect() method"
        sig = inspect.signature(det.detect)
        assert "image" in sig.parameters

        print(f"{PASS} HandDistortionDetector has correct interface")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 13 — bad API key does NOT crash agent
# ---------------------------------------------------------------------------

def test_bad_api_key():
    print("\n[TEST 13] detect() with invalid API key does not raise")
    if not _sample_exists():
        print(f"{SKIP} Sample image not found — skipping")
        return True
    try:
        from postvton.agents.problem_detection_agent import ProblemDetectionAgent

        agent = ProblemDetectionAgent(
            api_keys=["INVALID_KEY_FOR_TESTING"],
            max_retries_per_key=1,
            max_total_retries=1,
        )
        sample = _load_image(SAMPLE_IMAGE)
        report = agent.detect(sample, original_image=sample)

        assert isinstance(report.accessories.missing, bool)
        print(f"{PASS} No crash. accessories.missing={report.accessories.missing}, error={report.error}")
        return True
    except Exception as exc:
        print(f"{FAIL} Agent raised an exception (should not): {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 14 — live two-image detection with real Gemini key (optional)
# ---------------------------------------------------------------------------

def test_live_detection():
    print("\n[TEST 14] Live two-image detect() with real HF_TOKEN")
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print(f"{SKIP} HF_TOKEN not set — skipping live test")
        return True
    if not _sample_exists():
        print(f"{SKIP} Sample image not found — skipping live test")
        return True
    if not _tryon_exists():
        print(f"{SKIP} Try-on image not found — skipping live test")
        return True

    try:
        from postvton.agents.problem_detection_agent import ProblemDetectionAgent

        agent = ProblemDetectionAgent(api_keys=[api_key])
        report = agent.detect(
            image=_load_image(TRYON_IMAGE),
            original_image=_load_image(SAMPLE_IMAGE),
            image_id=Path(TRYON_IMAGE).name,
        )

        print(f"  error             : {report.error}")
        print(f"  hands.distorted   : {report.hands.distorted}")
        print(f"  hands.analysis    : {report.hands.analysis}")
        print(f"  accessories.missing: {report.accessories.missing}")
        print(f"  accessories.details: {report.accessories.details}")

        assert report.image_id == Path(TRYON_IMAGE).name
        assert isinstance(report.hands.distorted, bool)
        assert isinstance(report.accessories.missing, bool)

        print(f"{PASS} Live two-image detection returned accessories/hands schema")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_imports,
        test_dataclass_helpers,
        test_missing_file,
        test_missing_original_image_path,
        test_two_image_mode,
        test_detect_problems_signature,
        test_missing_accessory_detector_interface,
        test_missing_accessory_result_shape,
        test_detect_from_failed_tryon_result,
        test_detect_from_successful_tryon_result,
        test_hand_detection_result_shape,
        test_hand_detector_interface,
        test_bad_api_key,
        test_live_detection,
    ]

    results = {}
    for fn in tests:
        results[fn.__name__] = fn()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = 0
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
