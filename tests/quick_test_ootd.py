"""
Quick test for OOTDiffusion with specific example images
"""

import sys
from pathlib import Path
import time

# Add project root to path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

from postvton.tools.tryon.ootdiffusion import OOTDiffusionInference


def main():
    print("=" * 60)
    print("OOTDiffusion Quick Test")
    print("=" * 60)
    
    # Use example images
    person_image = str(root_path / "external/ootdiffusion/run/examples/model/05997_00.jpg")
    cloth_image = str(root_path / "external/ootdiffusion/run/examples/garment/00055_00.jpg")
    
    # Check if images exist
    if not Path(person_image).exists():
        print(f"\n❌ Error: Person image not found: {person_image}")
        return
    
    if not Path(cloth_image).exists():
        print(f"\n❌ Error: Cloth image not found: {cloth_image}")
        return
    
    print(f"\n📷 Person image: {Path(person_image).name}")
    print(f"👕 Cloth image: {Path(cloth_image).name}")
    print(f"🎯 Category: upperbody")
    print(f"🤖 Model: HD (High-Definition)")
    
    try:
        # Initialize OOTDiffusion
        print("\n🔧 Initializing OOTDiffusion HD model...")
        model = OOTDiffusionInference(model_type="hd", gpu_id=0)
        
        # Generate try-on result
        print("🎨 Generating virtual try-on...")
        start_time = time.time()
        
        result = model.generate(
            person_image=person_image,
            cloth_image=cloth_image,
            category="upperbody",
            num_inference_steps=20,
            guidance_scale=2.0,
            seed=42
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Generation completed in {elapsed:.2f}s")
        
        # Save result
        output_dir = root_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "ootd_test_result.png"
        
        result.save(output_path)
        print(f"💾 Result saved to: {output_path}")
        
        # Print info
        print(f"\n📊 Result Info:")
        print(f"   - Size: {result.size}")
        print(f"   - Mode: {result.mode}")
        print(f"   - Format: PNG")
        
        # Cleanup
        model.unload()
        print("\n✨ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
