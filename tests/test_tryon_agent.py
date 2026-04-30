"""Tests for TryOnAgent.

Covers:
    1. Imports and class instantiation (no GPU needed).
    2. Remote try-on call (skipped unless TRYON_SERVER_URL is set).

Run from the project root:
    python tests/test_tryon_agent.py
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Sample images (shipped with CatVTON demo data)
# ---------------------------------------------------------------------------
PERSON_IMAGE = str(
    PROJECT_ROOT
    / "external/catvton/resource/demo/example/person/women/049713_0.jpg"
)
CLOTH_IMAGE = str(
    PROJECT_ROOT
    / "external/catvton/resource/demo/example/condition/upper/24083449_54173465_2048.jpg"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_tryon_agent"

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def _server_url() -> str:
    return (os.environ.get("TRYON_SERVER_URL") or "").strip()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_images() -> bool:
    missing = [p for p in (PERSON_IMAGE, CLOTH_IMAGE) if not Path(p).exists()]
    if missing:
        for p in missing:
            print(f"  Missing sample image: {p}")
    return len(missing) == 0


# ---------------------------------------------------------------------------
# Test 1 — imports and instantiation
# ---------------------------------------------------------------------------

def test_imports():
    print("\n[TEST 1] Import and instantiate TryOnAgent")
    try:
        from postvton.agents.tryon_agent import (
            TryOnAgent,
            TryOnResult,
            run_tryon_agent_sync,
        )
        agent = TryOnAgent(device="cuda")
        assert agent.device == "cuda"

        # TryOnResult defaults
        r = TryOnResult(success=True)
        assert r.pose_score == 0.0
        assert r.output_path is None

        print(f"{PASS} Imports and instantiation OK")
        return True
    except Exception as exc:
        print(f"{FAIL} {exc}")
        return False

# ---------------------------------------------------------------------------
# Test 2 — remote generate()
# ---------------------------------------------------------------------------

def test_generate():
    print("\n[TEST 2] TryOnAgent.generate() — remote server")
    if not _server_url():
        print(f"{SKIP} TRYON_SERVER_URL not set — skipping remote try-on test")
        return True
    if not _check_images():
        print(f"{SKIP} Sample images not found — skipping inference test")
        return True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(OUTPUT_DIR / "best_result.png")

    try:
        from postvton.agents.tryon_agent import TryOnAgent

        agent = TryOnAgent(device="cuda")
        t0 = time.time()
        result = agent.generate(
            person_image_path=PERSON_IMAGE,
            cloth_image_path=CLOTH_IMAGE,
            cloth_type="upper",       # CatVTON notation; agent passes "category" for OOTDiffusion
            output_path=out_path,
            num_inference_steps=20,   # keep fast for testing
        )
        elapsed = time.time() - t0

        print(f"  success       : {result.success}")
        print(f"  model_used    : {result.model_used}")
        print(f"  pose_score    : {result.pose_score:.4f}")
        print(f"  inference_time: {result.inference_time:.2f}s  (total: {elapsed:.2f}s)")
        print(f"  output_path   : {result.output_path}")

        assert result.success, f"generate() returned failure: {result.message}"
        assert result.output_path and Path(result.output_path).exists(), \
            f"Output file not found: {result.output_path}"
        assert result.pose_score >= 0.0

        print(f"{PASS} Best result saved to {result.output_path}")
        return True

    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 4 — run_tryon_agent_sync convenience wrapper
# ---------------------------------------------------------------------------

def test_sync_wrapper():
    print("\n[TEST 3] run_tryon_agent_sync convenience function")
    if not _server_url():
        print(f"{SKIP} TRYON_SERVER_URL not set — skipping")
        return True
    if not _check_images():
        print(f"{SKIP} Sample images not found — skipping")
        return True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(OUTPUT_DIR / "sync_best_result.png")

    try:
        from postvton.agents.tryon_agent import run_tryon_agent_sync

        result = run_tryon_agent_sync(
            person_image_path=PERSON_IMAGE,
            cloth_image_path=CLOTH_IMAGE,
            cloth_type="upper",
            output_path=out_path,
            num_inference_steps=20,
        )

        assert result.success, f"run_tryon_agent_sync failed: {result.message}"
        assert result.output_path and Path(result.output_path).exists()
        print(f"{PASS} sync wrapper OK — {result.model_used} (pose_score={result.pose_score:.4f})")
        return True

    except Exception as exc:
        print(f"{FAIL} {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 5 — graceful failure on missing images
# ---------------------------------------------------------------------------

def test_missing_image():
    print("\n[TEST 4] Graceful failure on missing input paths")
    try:
        from postvton.agents.tryon_agent import TryOnAgent

        agent = TryOnAgent(device="cuda")
        result = agent.generate(
            person_image_path="/nonexistent/person.jpg",
            cloth_image_path="/nonexistent/cloth.jpg",
            cloth_type="upper",
        )

        # Both models should fail → success=False, no crash
        assert not result.success, "Expected failure for missing images but got success"
        print(f"{PASS} Graceful failure: '{result.message[:80]}'")
        return True

    except Exception as exc:
        print(f"{FAIL} Unexpected exception: {exc}")
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_imports,
        test_generate,
        test_sync_wrapper,
        test_missing_image,
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
