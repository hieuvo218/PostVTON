"""Tests for ExecutionAgent post-processing flow.

These tests use lightweight fake tools to avoid loading heavy ML models.
Run from project root:
    python tests/test_execution_agent.py
"""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TMP_DIR = PROJECT_ROOT / "outputs" / "test_execution_agent"

PASS = "[PASS]"
FAIL = "[FAIL]"


@dataclass
class FakeHandResult:
    success: bool
    output_path: Optional[str]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "error": self.error,
        }


@dataclass
class FakeAccessoryResult:
    success: bool
    output_path: Optional[str]
    error: Optional[str] = None
    restored_count: int = 1

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "error": self.error,
            "restored_count": self.restored_count,
        }


class FakeHandRefiner:
    def __init__(self):
        self.calls = []

    def refine(self, image, output_path, **kwargs):
        self.calls.append({"image": image, "output_path": output_path, "kwargs": kwargs})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, output_path)
        return FakeHandResult(success=True, output_path=str(output_path))


class FakeAccessoryRestorer:
    def __init__(self):
        self.calls = []

    def restore(self, source_image, target_image, output_path=None, **kwargs):
        self.calls.append(
            {
                "source_image": source_image,
                "target_image": target_image,
                "output_path": output_path,
                "kwargs": kwargs,
            }
        )
        if output_path is None:
            return FakeAccessoryResult(success=True, output_path=None)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_image, output_path)
        return FakeAccessoryResult(success=True, output_path=str(output_path))


def _write_dummy_image(path: Path):
    # A tiny valid PNG payload (1x1 transparent pixel)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\x0d\x0a\x2d\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


def test_import_and_init():
    print("\n[TEST 1] Import and initialization")
    try:
        from postvton.agents.execution_agent import ExecutionAgent, ExecutionResult

        agent = ExecutionAgent(
            hand_refiner=FakeHandRefiner(),
            accessory_restorer=FakeAccessoryRestorer(),
        )
        assert agent is not None
        assert ExecutionResult is not None
        print(f"{PASS} Import and init OK")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_missing_original_image():
    print("\n[TEST 2] Missing original image returns error")
    try:
        from postvton.agents.execution_agent import ExecutionAgent

        tryon = TMP_DIR / "tryon.png"
        _write_dummy_image(tryon)

        agent = ExecutionAgent(
            hand_refiner=FakeHandRefiner(),
            accessory_restorer=FakeAccessoryRestorer(),
        )
        result = agent.execute(
            original_image_path=str(TMP_DIR / "missing.png"),
            tryon_image_path=str(tryon),
        )
        assert result.success is False
        assert result.error and "Original image not found" in result.error
        print(f"{PASS} Proper error for missing original image")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_full_flow_runs_both_steps():
    print("\n[TEST 3] Full flow runs hand + accessory steps")
    try:
        from postvton.agents.execution_agent import ExecutionAgent

        original = TMP_DIR / "original.png"
        tryon = TMP_DIR / "tryon_full.png"
        output = TMP_DIR / "final_full.png"
        _write_dummy_image(original)
        _write_dummy_image(tryon)

        hand = FakeHandRefiner()
        accessory = FakeAccessoryRestorer()

        agent = ExecutionAgent(hand_refiner=hand, accessory_restorer=accessory)
        result = agent.execute(
            original_image_path=str(original),
            tryon_image_path=str(tryon),
            output_path=str(output),
        )

        assert result.success is True
        assert result.final_output_path == str(output)
        assert Path(result.final_output_path).exists()
        assert len(result.steps) == 2
        assert result.steps[0].name == "hand_refinement"
        assert result.steps[1].name == "accessory_restoration"
        assert len(hand.calls) == 1
        assert len(accessory.calls) == 1

        print(f"{PASS} Full flow completed with both steps")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_accessory_only_flow():
    print("\n[TEST 4] Accessory-only flow skips hand step")
    try:
        from postvton.agents.execution_agent import ExecutionAgent

        original = TMP_DIR / "original_only.png"
        tryon = TMP_DIR / "tryon_only.png"
        output = TMP_DIR / "final_only.png"
        _write_dummy_image(original)
        _write_dummy_image(tryon)

        hand = FakeHandRefiner()
        accessory = FakeAccessoryRestorer()

        agent = ExecutionAgent(hand_refiner=hand, accessory_restorer=accessory)
        result = agent.execute(
            original_image_path=str(original),
            tryon_image_path=str(tryon),
            output_path=str(output),
            refine_hands=False,
            restore_accessories=True,
        )

        assert result.success is True
        assert len(result.steps) == 1
        assert result.steps[0].name == "accessory_restoration"
        assert len(hand.calls) == 0
        assert len(accessory.calls) == 1
        assert accessory.calls[0]["target_image"] == str(tryon)

        print(f"{PASS} Accessory-only flow correct")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_hand_refinement_step():
    print("\n[TEST 5] Direct hand refinement step")
    try:
        from postvton.agents.execution_agent import ExecutionAgent

        tryon = TMP_DIR / "tryon_direct_hand.png"
        output = TMP_DIR / "hand_direct_out.png"
        _write_dummy_image(tryon)

        hand = FakeHandRefiner()
        accessory = FakeAccessoryRestorer()
        agent = ExecutionAgent(hand_refiner=hand, accessory_restorer=accessory)

        step = agent.run_hand_refinement(
            tryon_image_path=str(tryon),
            output_path=str(output),
        )

        assert step.success is True
        assert step.name == "hand_refinement"
        assert step.output_path == str(output)
        assert Path(step.output_path).exists()
        assert len(hand.calls) == 1
        assert len(accessory.calls) == 0

        print(f"{PASS} Direct hand refinement step works")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_accessory_restoration_step():
    print("\n[TEST 6] Direct accessory restoration step")
    try:
        from postvton.agents.execution_agent import ExecutionAgent

        original = TMP_DIR / "original_direct_accessory.png"
        target = TMP_DIR / "target_direct_accessory.png"
        output = TMP_DIR / "accessory_direct_out.png"
        _write_dummy_image(original)
        _write_dummy_image(target)

        hand = FakeHandRefiner()
        accessory = FakeAccessoryRestorer()
        agent = ExecutionAgent(hand_refiner=hand, accessory_restorer=accessory)

        step = agent.run_accessory_restoration(
            original_image_path=str(original),
            target_image_path=str(target),
            output_path=str(output),
        )

        assert step.success is True
        assert step.name == "accessory_restoration"
        assert step.output_path == str(output)
        assert Path(step.output_path).exists()
        assert len(hand.calls) == 0
        assert len(accessory.calls) == 1

        print(f"{PASS} Direct accessory restoration step works")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback
        traceback.print_exc()
        return False


def run_all():
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    tests = [
        test_import_and_init,
        test_missing_original_image,
        test_full_flow_runs_both_steps,
        test_accessory_only_flow,
        test_direct_hand_refinement_step,
        test_direct_accessory_restoration_step,
    ]

    results = {}
    for fn in tests:
        results[fn.__name__] = fn()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = 0
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL}  {name}")
        if ok:
            passed += 1

    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
