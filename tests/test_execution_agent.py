"""Tests for ExecutionAgent post-processing flow.

These tests use lightweight fake tools to avoid loading heavy ML models.
Run from project root:
    python tests/test_execution_agent.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"


@dataclass
class FakeHandResult:
    success: bool
    output_image: Optional[Image.Image]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_image": self.output_image is not None,
            "error": self.error,
        }


@dataclass
class FakeAccessoryResult:
    success: bool
    output_image: Optional[Image.Image]
    error: Optional[str] = None
    restored_count: int = 1

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_image": self.output_image is not None,
            "error": self.error,
            "restored_count": self.restored_count,
        }


class FakeHandRefiner:
    def __init__(self):
        self.calls = []

    def refine(self, image, **kwargs):
        self.calls.append({"image": image, "kwargs": kwargs})
        return FakeHandResult(success=True, output_image=image.copy())


class FakeAccessoryRestorer:
    def __init__(self):
        self.calls = []

    def restore(self, source_image, target_image, **kwargs):
        self.calls.append(
            {
                "source_image": source_image,
                "target_image": target_image,
                "kwargs": kwargs,
            }
        )
        return FakeAccessoryResult(success=True, output_image=target_image.copy())


def _make_dummy_image() -> Image.Image:
    return Image.new("RGB", (4, 4), color=(128, 128, 128))


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

        tryon = _make_dummy_image()
        agent = ExecutionAgent(
            hand_refiner=FakeHandRefiner(),
            accessory_restorer=FakeAccessoryRestorer(),
        )
        result = agent.execute(
            original_image="not-an-image",
            tryon_image=tryon,
        )
        assert result.success is False
        assert result.error and "original_image" in result.error
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

        original = _make_dummy_image()
        tryon = _make_dummy_image()

        hand = FakeHandRefiner()
        accessory = FakeAccessoryRestorer()

        agent = ExecutionAgent(hand_refiner=hand, accessory_restorer=accessory)
        result = agent.execute(
            original_image=original,
            tryon_image=tryon,
        )

        assert result.success is True
        assert result.final_image is not None
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

        original = _make_dummy_image()
        tryon = _make_dummy_image()

        hand = FakeHandRefiner()
        accessory = FakeAccessoryRestorer()

        agent = ExecutionAgent(hand_refiner=hand, accessory_restorer=accessory)
        result = agent.execute(
            original_image=original,
            tryon_image=tryon,
            refine_hands=False,
            restore_accessories=True,
        )

        assert result.success is True
        assert len(result.steps) == 1
        assert result.steps[0].name == "accessory_restoration"
        assert len(hand.calls) == 0
        assert len(accessory.calls) == 1
        assert accessory.calls[0]["target_image"] is tryon

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

        tryon = _make_dummy_image()

        hand = FakeHandRefiner()
        accessory = FakeAccessoryRestorer()
        agent = ExecutionAgent(hand_refiner=hand, accessory_restorer=accessory)

        step = agent.run_hand_refinement(
            tryon_image=tryon,
        )

        assert step.success is True
        assert step.name == "hand_refinement"
        assert step.output_image is not None
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

        original = _make_dummy_image()
        target = _make_dummy_image()

        hand = FakeHandRefiner()
        accessory = FakeAccessoryRestorer()
        agent = ExecutionAgent(hand_refiner=hand, accessory_restorer=accessory)

        step = agent.run_accessory_restoration(
            original_image=original,
            target_image=target,
        )

        assert step.success is True
        assert step.name == "accessory_restoration"
        assert step.output_image is not None
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
