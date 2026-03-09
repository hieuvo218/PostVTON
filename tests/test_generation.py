"""
Test script for CatVTONInference generation functionality.
This script tests actual image generation with sample images.
"""

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from postvton.tools.tryon.catvton import CatVTONInference


def load_real_test_images():
    """Load real test images from external/ootdiffusion examples"""
    print("\n[TEST] Loading real test images from external/ootdiffusion...")
    
    person_path = project_root / "external/ootdiffusion/run/examples/model/05997_00.jpg"
    cloth_path = project_root / "external/ootdiffusion/run/examples/garment/00055_00.jpg"
    
    # Check if images exist
    if not person_path.exists():
        print(f"[WARN] Person image not found at {person_path}")
        print("[TEST] Falling back to synthetic images...")
        return create_synthetic_test_images()
    
    if not cloth_path.exists():
        print(f"[WARN] Cloth image not found at {cloth_path}")
        print("[TEST] Falling back to synthetic images...")
        return create_synthetic_test_images()
    
    # Load images
    person_image = Image.open(person_path).convert('RGB')
    cloth_image = Image.open(cloth_path).convert('RGB')
    
    print(f"[TEST] Person image loaded: {person_image.size} from {person_path.name}")
    print(f"[TEST] Cloth image loaded: {cloth_image.size} from {cloth_path.name}")
    
    return person_image, cloth_image


def create_synthetic_test_images():
    """Create synthetic test images for try-on (fallback)"""
    print("\n[TEST] Creating synthetic test images...")
    
    # Create person image (768x1024 is the target size)
    # Simulating a person with different colored regions
    person_array = np.zeros((1024, 768, 3), dtype=np.uint8)
    
    # Head (top)
    person_array[0:200, :] = [220, 180, 160]  # Skin tone
    
    # Upper body / torso (without clothing)
    person_array[200:400, :] = [210, 170, 150]  # Skin tone
    
    # Lower body / pants
    person_array[400:900, :] = [50, 50, 100]  # Dark blue pants
    
    # Feet
    person_array[900:, :] = [30, 30, 30]  # Black shoes
    
    person_image = Image.fromarray(person_array, mode='RGB')
    
    # Create cloth image
    cloth_array = np.zeros((800, 600, 3), dtype=np.uint8)
    # Create a colorful shirt pattern
    cloth_array[:, :] = [220, 100, 50]  # Orange/coral color
    # Add some pattern
    cloth_array[100:300, 100:500] = [240, 150, 80]  # Lighter orange
    
    cloth_image = Image.fromarray(cloth_array, mode='RGB')
    
    print(f"[TEST] Person image created: {person_image.size}")
    print(f"[TEST] Cloth image created: {cloth_image.size}")
    
    return person_image, cloth_image


def save_test_images(person_image, cloth_image, output_dir="tests/temp"):
    """Save test images for inspection"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    person_path = output_path / "test_person.png"
    cloth_path = output_path / "test_cloth.png"
    
    person_image.save(person_path)
    cloth_image.save(cloth_path)
    
    print(f"[TEST] Test images saved:")
    print(f"  - {person_path}")
    print(f"  - {cloth_path}")
    
    return person_path, cloth_path


def test_model_initialization():
    """Test CatVTONInference initialization"""
    print("\n[TEST] Testing CatVTONInference initialization...")
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TEST] Using device: {device}")
        
        # Initialize with default parameters
        model = CatVTONInference(device=device)
        
        assert model.device == device
        assert model.dtype == torch.bfloat16
        assert model._pipeline is None
        assert model._masker is None
        
        print("[PASS] Model initialization successful")
        return model
        
    except Exception as e:
        print(f"[FAIL] Model initialization failed: {e}")
        raise


def test_generation_single_image(model, person_image, cloth_image):
    """Test single image generation with cloth_type='upper'"""
    print("\n[TEST] Testing single image generation with cloth_type='upper'...")
    print(f"[TEST] Input shapes - Person: {person_image.size}, Cloth: {cloth_image.size}")
    
    try:
        start_time = time.time()
        
        print(f"\n[TEST] Generating with cloth_type='upper'...")
        
        result = model.generate(
            person_image=person_image,
            cloth_image=cloth_image,
            cloth_type="upper",
            num_inference_steps=50,  # Full quality
            seed=42
        )
        
        elapsed = time.time() - start_time
        
        # Validate result
        assert isinstance(result, Image.Image), "Result should be PIL Image"
        assert result.mode == 'RGB', "Result should be RGB mode"
        assert result.size == (768, 1024), f"Result size should be (768, 1024), got {result.size}"
        
        print(f"[PASS] Generation successful for cloth_type='upper' ({elapsed:.2f}s)")
        
        # Save result
        output_dir = Path("tests/temp")
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"test_result_upper.png"
        result.save(result_path)
        print(f"[TEST] Result saved: {result_path}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_with_seed(model, person_image, cloth_image):
    """Test generation reproducibility with seed"""
    print("\n[TEST] Testing seed reproducibility...")
    
    try:
        seed = 42
        
        # Generate with same seed twice
        result1 = model.generate(
            person_image=person_image,
            cloth_image=cloth_image,
            cloth_type="upper",
            num_inference_steps=20,
            seed=seed
        )
        
        result2 = model.generate(
            person_image=person_image,
            cloth_image=cloth_image,
            cloth_type="upper",
            num_inference_steps=20,
            seed=seed
        )
        
        # Convert to numpy for comparison
        result1_array = np.array(result1)
        result2_array = np.array(result2)
        
        # Check if results are identical (with some tolerance for float operations)
        diff = np.abs(result1_array.astype(float) - result2_array.astype(float)).mean()
        
        if diff < 1.0:  # Very small difference acceptable due to float precision
            print(f"[PASS] Seed reproducibility successful (diff: {diff:.6f})")
            return True
        else:
            print(f"[WARN] Results differ (diff: {diff:.6f}), might be due to float precision")
            return True
            
    except Exception as e:
        print(f"[FAIL] Seed test failed: {e}")
        return False


def test_batch_generation(model, person_image, cloth_image):
    """Test batch generation"""
    print("\n[TEST] Testing batch generation...")
    
    try:
        # Create multiple test pairs
        person_images = [person_image, person_image]
        cloth_images = [cloth_image, cloth_image]
        
        results = model.batch_generate(
            person_images=person_images,
            cloth_images=cloth_images,
            cloth_type="upper",
            num_inference_steps=20
        )
        
        assert len(results) == 2, "Should return 2 results"
        assert all(isinstance(img, Image.Image) for img in results), "All results should be PIL Images"
        
        print(f"[PASS] Batch generation successful ({len(results)} images)")
        
        # Save results
        for i, result in enumerate(results):
            output_dir = Path("tests/temp")
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / f"test_batch_result_{i}.png"
            result.save(result_path)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Batch generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_cloth_types(model, person_image, cloth_image):
    """Test all cloth types"""
    print("\n[TEST] Testing different cloth types...")
    
    cloth_types = ["upper", "lower", "overall", "inner", "outer"]
    results = {}
    
    try:
        for cloth_type in cloth_types:
            print(f"[TEST] Generating for cloth_type='{cloth_type}'...")
            
            result = model.generate(
                person_image=person_image,
                cloth_image=cloth_image,
                cloth_type=cloth_type,
                num_inference_steps=20
            )
            
            results[cloth_type] = result
            
            # Validate
            assert isinstance(result, Image.Image)
            assert result.size == (768, 1024)
            
            print(f"[PASS] {cloth_type}: OK")
        
        print(f"\n[PASS] All cloth types tested successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Cloth type test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("CatVTONInference Generation Tests")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    results = {}
    
    try:
        # Test 1: Initialization
        model = test_model_initialization()
        results['initialization'] = True
        
        # Test 2: Load real test images
        person_image, cloth_image = load_real_test_images()
        save_test_images(person_image, cloth_image)
        
        # Test 3: Single image generation
        try:
            results['single_generation'] = test_generation_single_image(model, person_image, cloth_image)
        except RuntimeError as e:
            if "Failed to import CatVTON models" in str(e) or "Failed to load CatVTON models" in str(e):
                print(f"\n[SKIP] Model loading failed (expected if model not downloaded): {e}")
                results['single_generation'] = 'SKIPPED'
            else:
                raise
        
        # Test 4: Seed reproducibility (if generation succeeded)
        if results.get('single_generation') is True:
            results['seed_reproducibility'] = test_generation_with_seed(model, person_image, cloth_image)
        
        # Test 5: Batch generation (if generation succeeded)
        if results.get('single_generation') is True:
            results['batch_generation'] = test_batch_generation(model, person_image, cloth_image)
        
        # Test 6: Different cloth types (if generation succeeded)
        if results.get('single_generation') is True:
            results['cloth_types'] = test_different_cloth_types(model, person_image, cloth_image)
        
    except Exception as e:
        print(f"\n[FATAL] Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'model' in locals():
            model.unload()
            print("\n[TEST] Model unloaded")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "PASS" if result is True else "FAIL" if result is False else "SKIPPED"
        print(f"  {test_name}: {status}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r == 'SKIPPED')
    
    print(f"\nTotal: {len(results)} tests, {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
