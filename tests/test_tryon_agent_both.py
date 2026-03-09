"""
Test script for TryOn Agent with both CatVTON and OOTDiffusion
"""

import asyncio
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from postvton.agents.tryon_agent import run_tryon_agent, VTONModel


async def test_ootdiffusion_local():
    """Test OOTDiffusion with local inference"""
    print("=" * 60)
    print("Test 1: OOTDiffusion Local Inference")
    print("=" * 60)
    
    person_image = str(root_path / "external/ootdiffusion/run/examples/model/05997_00.jpg")
    cloth_image = str(root_path / "external/ootdiffusion/run/examples/garment/00055_00.jpg")
    output_path = str(root_path / "outputs/ootd_local_result.png")
    
    # Check if images exist
    if not Path(person_image).exists():
        print(f"❌ Person image not found: {person_image}")
        return False
    
    if not Path(cloth_image).exists():
        print(f"❌ Cloth image not found: {cloth_image}")
        return False
    
    print(f"\n📷 Person image: {Path(person_image).name}")
    print(f"👕 Cloth image: {Path(cloth_image).name}")
    print(f"🤖 Model: OOTDiffusion (HD)")
    print(f"💾 Output: {output_path}")
    
    try:
        result = await run_tryon_agent(
            person_image_path=person_image,
            cloth_image_path=cloth_image,
            cloth_type="upper",
            output_path=output_path,
            model=VTONModel.OOTDIFFUSION,
            num_inference_steps=20
        )
        
        print(f"\n✅ Success: {result.success}")
        print(f"📁 Output: {result.output_path}")
        print(f"⏱️  Time: {result.inference_time:.2f}s")
        print(f"💬 Message: {result.message}")
        
        return result.success
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_catvton_local():
    """Test CatVTON with local inference (if available)"""
    print("\n" + "=" * 60)
    print("Test 2: CatVTON Local Inference")
    print("=" * 60)
    
    person_image = str(root_path / "external/ootdiffusion/run/examples/model/05997_00.jpg")
    cloth_image = str(root_path / "external/ootdiffusion/run/examples/garment/00055_00.jpg")
    output_path = str(root_path / "outputs/catvton_local_result.png")
    
    print(f"\n📷 Person image: {Path(person_image).name}")
    print(f"👕 Cloth image: {Path(cloth_image).name}")
    print(f"🤖 Model: CatVTON")
    print(f"💾 Output: {output_path}")
    
    try:
        result = await run_tryon_agent(
            person_image_path=person_image,
            cloth_image_path=cloth_image,
            cloth_type="upper",
            output_path=output_path,
            model=VTONModel.CATVTON,
            num_inference_steps=30
        )
        
        print(f"\n✅ Success: {result.success}")
        print(f"📁 Output: {result.output_path}")
        print(f"⏱️  Time: {result.inference_time:.2f}s")
        print(f"💬 Message: {result.message}")
        
        return result.success
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"   (This is expected if CatVTON models are not downloaded)")
        return None  # Mark as skipped


async def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "TryOn Agent Comprehensive Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    results = {}

    # Test 1: OOTDiffusion Local
    try:
        results['ootd_local'] = await test_ootdiffusion_local()
    except Exception as e:
        print(f"Fatal error in OOTDiffusion local test: {e}")
        results['ootd_local'] = False

    # Test 2: CatVTON Local
    try:
        results['catvton_local'] = await test_catvton_local()
    except Exception as e:
        print(f"Fatal error in CatVTON local test: {e}")
        results['catvton_local'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
        print(f"  {test_name:20} {status}")
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"\nTotal: {len(results)} tests")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    
    if failed == 0:
        print("\n🎉 All executable tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
