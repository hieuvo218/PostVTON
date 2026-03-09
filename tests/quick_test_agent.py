"""
Quick test for TryOn Agent with OOTDiffusion (synchronous)
"""

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from postvton.agents.tryon_agent import run_tryon_agent_sync, VTONModel


def main():
    print("=" * 60)
    print("TryOn Agent - Quick Test")
    print("=" * 60)
    
    # Use example images
    person_image = str(root_path / "external/ootdiffusion/run/examples/model/05997_00.jpg")
    cloth_image = str(root_path / "external/ootdiffusion/run/examples/garment/00055_00.jpg")
    output_path = str(root_path / "outputs/tryon_quick_test.png")
    
    # Check if images exist
    if not Path(person_image).exists():
        print(f"❌ Error: Person image not found: {person_image}")
        return
    
    if not Path(cloth_image).exists():
        print(f"❌ Error: Cloth image not found: {cloth_image}")
        return
    
    print(f"\n📷 Person image: {Path(person_image).name}")
    print(f"👕 Cloth image: {Path(cloth_image).name}")
    print(f"🤖 Model: OOTDiffusion (HD)")
    print(f"💾 Output: {output_path}")
    
    try:
        print("\n🎨 Generating virtual try-on...")
        result = run_tryon_agent_sync(
            person_image_path=person_image,
            cloth_image_path=cloth_image,
            cloth_type="upper",
            output_path=output_path,
            model=VTONModel.OOTDIFFUSION,
            num_inference_steps=20,
            seed=42
        )
        
        print(f"\n✅ Success: {result.success}")
        print(f"📁 Output: {result.output_path}")
        print(f"⏱️  Inference time: {result.inference_time:.2f}s")
        print(f"🤖 Model used: {result.model_used}")
        print(f"💬 Message: {result.message}")
        
        if result.success:
            print("\n✨ Try-on generated successfully!")
        else:
            print("\n⚠️  Try-on generation failed")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
