import torch
import os
from pathlib import Path
from typing import Optional, Union, List, Tuple
from PIL import Image
import numpy as np
import sys

from huggingface_hub import snapshot_download

repo_id = "levihsu/OOTDiffusion"  # Example repository ID

root_path = Path(__file__).parents[4]
# Download the file
# Check if the directory already exists to avoid re-downloading
# if not (root_path / "external" / "ootdiffusion").exists():
#     snapshot_download(repo_id=repo_id, local_dir=root_path/"external/ootdiffusion")
#     snapshot_download(repo_id="openai/clip-vit-large-patch14", local_dir=root_path/"external/ootdiffusion/checkpoints/clip-vit-large-patch14")

# Add external OOTDiffusion to path
project_root = Path(__file__).parent.parent.parent.parent
ootd_path = project_root / "external" / "ootdiffusion"
humanparsing_path = ootd_path / "preprocess" / "humanparsing"
if str(ootd_path) not in sys.path:
    sys.path.insert(0, str(ootd_path))
if str(ootd_path / "run") not in sys.path:
    sys.path.insert(0, str(ootd_path / "run"))
# humanparsing must precede catvton on sys.path so its utils/ package
# takes priority over external/catvton/utils.py
if str(humanparsing_path) not in sys.path:
    sys.path.insert(0, str(humanparsing_path))


class OOTDiffusionInference:
    """OOTDiffusion Virtual Try-On Inference Wrapper
    
    Wrapper for the OOTDiffusion model from https://github.com/levihsu/OOTDiffusion
    Supports high-definition (HD) and detail-controllable (DC) modes.
    """
    
    def __init__(
        self,
        model_type: str = "hd",
        gpu_id: int = 0,
        device: Optional[str] = None
    ):
        """Initialize OOTDiffusion inference wrapper.
        
        Args:
            model_type: Model type - "hd" (high-definition) or "dc" (detail-controllable)
            gpu_id: GPU device ID (0, 1, 2, etc.)
            device: Device to use (if None, uses cuda:{gpu_id} if available, else cpu)
        """
        self.model_type = model_type.lower()
        if self.model_type not in ["hd", "dc"]:
            raise ValueError("model_type must be 'hd' or 'dc'")
        
        self.gpu_id = gpu_id
        
        # Set device
        if device is None:
            self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Lazy loading
        self._model = None
        self._openpose = None
        self._parsing = None
        self._category_dict = ['upperbody', 'lowerbody', 'dress']
        self._category_dict_utils = ['upper_body', 'lower_body', 'dresses']
        
    def _load_models(self):
        """Lazy load OOTDiffusion models and preprocessing models."""
        if self._model is not None:
            return  # Already loaded

        checkpoints_root = ootd_path / "checkpoints"
        required_paths = [
            checkpoints_root / "ootd" / "vae" / "config.json",
            checkpoints_root / "ootd" / "tokenizer",
            checkpoints_root / "clip-vit-large-patch14",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise RuntimeError(
                "Missing local OOTDiffusion checkpoints. Please ensure these paths exist before inference: "
                + ", ".join(missing)
            )
        
        try:
            # catvton.py registers external/catvton/utils.py as sys.modules['utils']
            # (a plain file, not a package). Humanparsing needs utils/ as a package.
            # Temporarily evict the catvton utils so the humanparsing utils/ package
            # can be imported cleanly, then restore afterwards.
            _evicted = {k: sys.modules.pop(k)
                        for k in list(sys.modules)
                        if k == 'utils' or k.startswith('utils.')}

            try:
                from preprocess.openpose.run_openpose import OpenPose
                from preprocess.humanparsing.run_parsing import Parsing
            finally:
                # Restore catvton utils so nothing else breaks
                sys.modules.update(_evicted)

            if self.model_type == "hd":
                from ootd.inference_ootd_hd import OOTDiffusionHD
            else:
                from ootd.inference_ootd_dc import OOTDiffusionDC

            from run.utils_ootd import get_mask_location
            self._get_mask_location = get_mask_location

        except ImportError as e:
            raise ImportError(
                f"Failed to import OOTDiffusion models. "
                f"Make sure external/ootdiffusion is properly set up. Error: {e}"
            )
        
        try:
            print(f"[OOTDiffusion] Loading {self.model_type.upper()} model...")
            
            # Load preprocessing models
            print("[OOTDiffusion] Loading OpenPose...")
            self._openpose = OpenPose(self.gpu_id)
            
            print("[OOTDiffusion] Loading Human Parsing...")
            self._parsing = Parsing(self.gpu_id)
            
            # Load main model
            print(f"[OOTDiffusion] Loading {self.model_type.upper()} pipeline...")
            if self.model_type == "hd":
                self._model = OOTDiffusionHD(self.gpu_id)
            else:
                self._model = OOTDiffusionDC(self.gpu_id)
            
            print("[OOTDiffusion] All models loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load OOTDiffusion models. "
                f"Make sure checkpoints are downloaded. Error: {e}"
            )
    
    def generate(
        self,
        person_image: Union[Image.Image, str],
        cloth_image: Union[Image.Image, str],
        category: Union[int, str] = "upperbody",
        num_inference_steps: int = 10,
        guidance_scale: float = 2.0,
        num_samples: int = 1,
        seed: int = -1,
        return_mask: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """Generate virtual try-on result.
        
        Args:
            person_image: PIL Image or path to person/model image
            cloth_image: PIL Image or path to garment image
            category: Clothing category:
                     - "upperbody" / 0: Upper body clothing (shirts, jackets)
                     - "lowerbody" / 1: Lower body clothing (pants, skirts)  
                     - "dress" / 2: Full body dress
            num_inference_steps: Number of denoising steps (default 20)
            guidance_scale: Classifier-free guidance scale (default 2.0)
            num_samples: Number of samples to generate (default 1)
            seed: Random seed for reproducibility (-1 for random)
            return_mask: If True, return (result, mask) tuple
        
        Returns:
            Generated try-on image(s) as PIL Image or list of Images
            If return_mask=True, returns tuple of (result, masked_image)
        
        Note:
            - HD model only supports upperbody category
            - Images will be resized to 768x1024
        """
        self._load_models()
        
        # Convert category
        if isinstance(category, str):
            category = category.lower()
            if category == "upperbody" or category == "upper":
                category_idx = 0
            elif category == "lowerbody" or category == "lower":
                category_idx = 1
            elif category == "dress":
                category_idx = 2
            else:
                raise ValueError(f"Invalid category: {category}. Use 'upperbody', 'lowerbody', or 'dress'")
        else:
            category_idx = category
            if category_idx not in [0, 1, 2]:
                raise ValueError(f"Category index must be 0, 1, or 2, got {category_idx}")
        
        # Validate HD model constraint
        if self.model_type == "hd" and category_idx != 0:
            raise ValueError("HD model only supports upperbody category (category=0)")
        
        # Load images
        if isinstance(person_image, str):
            person_image = Image.open(person_image).convert('RGB')
        if isinstance(cloth_image, str):
            cloth_image = Image.open(cloth_image).convert('RGB')
        
        # Ensure RGB mode
        if person_image.mode != 'RGB':
            person_image = person_image.convert('RGB')
        if cloth_image.mode != 'RGB':
            cloth_image = cloth_image.convert('RGB')
        
        print(f"[OOTDiffusion] Processing images (category={self._category_dict[category_idx]}, "
              f"model={self.model_type})...")
        
        # Resize images to standard size
        cloth_img = cloth_image.resize((768, 1024))
        model_img = person_image.resize((768, 1024))
        
        # Run preprocessing
        print("[OOTDiffusion] Detecting keypoints with OpenPose...")
        keypoints = self._openpose(model_img.resize((384, 512)))
        
        print("[OOTDiffusion] Parsing human segments...")
        model_parse, _ = self._parsing(model_img.resize((384, 512)))
        
        print("[OOTDiffusion] Generating mask...")
        mask, mask_gray = self._get_mask_location(
            self.model_type, 
            self._category_dict_utils[category_idx],
            model_parse,
            keypoints
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        # Create masked image
        masked_vton_img = Image.composite(mask_gray, model_img, mask)
        
        print(f"[OOTDiffusion] Running inference ({num_inference_steps} steps)...")
        # Run inference
        images = self._model(
            model_type=self.model_type,
            category=self._category_dict[category_idx],
            image_garm=cloth_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_img,
            num_samples=num_samples,
            num_steps=num_inference_steps,
            image_scale=guidance_scale,
            seed=seed,
        )
        
        print("[OOTDiffusion] Inference complete!")
        
        # Return results
        if num_samples == 1:
            result = images[0]
            if return_mask:
                return result, masked_vton_img
            return result
        else:
            if return_mask:
                return images, masked_vton_img
            return images
    
    def batch_generate(
        self,
        person_images: List[Union[Image.Image, str]],
        cloth_images: List[Union[Image.Image, str]],
        **kwargs
    ) -> List[Image.Image]:
        """Generate virtual try-on for multiple image pairs.
        
        Args:
            person_images: List of PIL Images or paths
            cloth_images: List of PIL Images or paths
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            List of generated try-on images
        """
        assert len(person_images) == len(cloth_images), \
            "Number of person and garment images must match"
        
        results = []
        for i, (person_img, cloth_img) in enumerate(zip(person_images, cloth_images)):
            print(f"[OOTDiffusion] Processing pair {i+1}/{len(person_images)}...")
            result = self.generate(person_img, cloth_img, **kwargs)
            results.append(result)
        
        return results
    
    def unload(self):
        """Unload models to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._openpose is not None:
            del self._openpose
            self._openpose = None
        
        if self._parsing is not None:
            del self._parsing
            self._parsing = None
        
        torch.cuda.empty_cache()
        print("[OOTDiffusion] Models unloaded")


# Convenience function for simple inference
def run_ootdiffusion(
    person_image_path: str,
    cloth_image_path: str,
    output_path: Optional[str] = None,
    model_type: str = "hd",
    category: Union[int, str] = "upperbody",
    **kwargs
) -> Image.Image:
    """Run OOTDiffusion inference on image paths.
    
    Args:
        person_image_path: Path to person image
        cloth_image_path: Path to garment image
        output_path: Optional path to save output image
        model_type: "hd" or "dc"
        category: "upperbody", "lowerbody", or "dress"
        **kwargs: Additional arguments passed to OOTDiffusionInference.generate()
    
    Returns:
        Generated try-on image as PIL Image
    """
    # Load images
    person_img = Image.open(person_image_path).convert('RGB')
    cloth_img = Image.open(cloth_image_path).convert('RGB')
    
    # Initialize and run inference
    ootd = OOTDiffusionInference(model_type=model_type)
    result = ootd.generate(person_img, cloth_img, category=category, **kwargs)
    
    # Save if output path provided
    if output_path:
        if isinstance(result, list):
            # Multiple samples
            for i, img in enumerate(result):
                save_path = output_path.replace('.png', f'_{i}.png')
                img.save(save_path)
                print(f"[OOTDiffusion] Output saved to {save_path}")
        else:
            result.save(output_path)
            print(f"[OOTDiffusion] Output saved to {output_path}")
    
    return result
