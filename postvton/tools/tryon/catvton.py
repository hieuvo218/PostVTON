import torch
import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import sys

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# Add external CatVTON to path
project_root = Path(__file__).parent.parent.parent.parent
catvton_path = project_root / "external" / "catvton"
if str(catvton_path) not in sys.path:
    sys.path.insert(0, str(catvton_path))
from utils import init_weight_dtype, resize_and_crop, resize_and_padding


class CatVTONInference:
    """CatVTON Virtual Try-On Inference Tool"""
    
    def __init__(
        self,
        base_model_path: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
        resume_path: str = "zhengchong/CatVTON",
        dataset_name: str = "dresscode",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize CatVTON model components
        
        Args:
            base_model_path: Base stable diffusion model path
            resume_path: CatVTON checkpoint path
            dataset_name: Dataset version (dresscode or vitonhd)
            device: Device to run inference on
            dtype: Data type for model weights
        """
        self.device = device
        self.dtype = dtype
        self.base_model_path = base_model_path
        self.resume_path = resume_path
        self.dataset_name = dataset_name
        
        # Load model components (lazy loading)
        self._pipeline = None
        self._masker = None
        
    def _load_models(self):
        """Lazy load model components"""
        if self._pipeline is not None:
            return
            
        try:
            from model.cloth_masker import AutoMasker
            from model.pipeline import CatVTONPipeline
            from huggingface_hub import snapshot_download

            self._repo_path = snapshot_download(repo_id=self.resume_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import CatVTON models. Make sure external/catvton is properly set up. Error: {e}"
            )
        
        try:
            print("[CatVTON] Loading pipeline...")
            # Load main pipeline
            self._pipeline = CatVTONPipeline(
                base_ckpt=self.base_model_path,
                attn_ckpt=self._repo_path,
                attn_ckpt_version="mix",
                weight_dtype=self.dtype,
                skip_safety_check=True,
                device=self.device,
            )
            print("[CatVTON] Pipeline loaded successfully")
            
            print("[CatVTON] Initializing AutoMasker...")
            # Load cloth masker - try local path first, then download
            try:
                self._mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
                self._masker = AutoMasker(
                    densepose_ckpt=os.path.join(self._repo_path, "DensePose"),
                    schp_ckpt=os.path.join(self._repo_path, "SCHP"),
                    device=self.device
                )
                print("[CatVTON] AutoMasker initialized successfully")
            except Exception as masker_err:
                print(f"[CatVTON] Warning: AutoMasker initialization failed: {masker_err}")
                print("[CatVTON] Will use simple mask generation instead")
                self._masker = None
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CatVTON models. Make sure all required models are downloaded. Error: {e}"
            )
    
    def generate(
        self,
        person_image: Image.Image,
        cloth_image: Image.Image,
        cloth_type: str = "upper",
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: Optional[int] = None,
        show_type: str = "result only"
    ) -> Image.Image:
        """
        Generate virtual try-on result
        
        Args:
            person_image: PIL Image of the person
            cloth_image: PIL Image of the garment
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            mask_type: Type of mask for cloth-agnostic mask generation
                     ('upper', 'lower', 'overall', 'inner', 'outer')
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Generated try-on image as PIL Image
        """
        self._load_models()
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Resize images to 1024x768
        person_image = resize_and_crop(person_image, (768, 1024))
        cloth_image = resize_and_padding(cloth_image, (768, 1024))
        
        # Generate mask if not provided
        if self._masker is not None:
            try:
                # AutoMasker takes person_image and mask_type ('upper', 'lower', 'overall', 'inner', 'outer')
                mask_result = self._masker(person_image, cloth_type)
                mask_image = mask_result['mask'] if isinstance(mask_result, dict) else mask_result
                mask_image = self._mask_processor.blur(mask_image, blur_factor=9)
            except Exception as e:
                print(f"[CatVTON] Warning: Masker failed, using fallback mask: {e}")
                # Fallback: create a simple full mask
                mask_image = Image.new('L', person_image.size, 255)
        else:
            # Fallback: create a simple full mask if masker is not available
            mask_image = Image.new('L', person_image.size, 255)
        
        # Ensure mask is PIL Image in correct format
        if isinstance(mask_image, Image.Image):
            if mask_image.mode != 'L':
                mask_image = mask_image.convert('L')
        else:
            # If mask is a tensor or array, convert to PIL Image
            if isinstance(mask_image, np.ndarray):
                mask_image = Image.fromarray(mask_image.astype(np.uint8), mode='L')
            else:
                raise ValueError(f"Mask must be PIL Image or numpy array, got {type(mask_image)}")
        
        # Set up generator if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Run inference
        try:
            print("[CatVTON] Running inference...")
            with torch.no_grad():
                result = self._pipeline(
                    image=person_image,
                    condition_image=cloth_image,
                    mask=mask_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=1024,
                    width=768,
                    generator=generator,
                )
            print("[CatVTON] Inference completed successfully")
        except Exception as e:
            raise RuntimeError(
                f"CatVTON pipeline inference failed: {str(e)}\n"
                f"Input shapes - person: {person_image.size}, garment: {cloth_image.size}, mask: {mask_image.size}"
            )
        
        # Pipeline returns images directly as list/tuple or with .images attribute
        if isinstance(result, (list, tuple)):
            output_image = result[0]
        elif hasattr(result, 'images'):
            output_image = result.images[0]
        else:
            output_image = result
        
        return output_image
    
    def batch_generate(
        self,
        person_images: list[Image.Image],
        cloth_images: list[Image.Image],
        **kwargs
    ) -> list[Image.Image]:
        """
        Generate try-on results for multiple image pairs
        
        Args:
            person_images: List of person PIL Images
            cloth_images: List of garment PIL Images
            **kwargs: Additional arguments passed to generate()
            
        Returns:
            List of generated try-on images
        """
        assert len(person_images) == len(cloth_images), \
            "Number of person and garment images must match"
        
        results = []
        for person_img, cloth_img in zip(person_images, cloth_images):
            result = self.generate(person_img, cloth_img, **kwargs)
            results.append(result)
        
        return results
    
    def unload(self):
        """Unload models to free memory"""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        
        if self._masker is not None:
            del self._masker
            self._masker = None
            
        torch.cuda.empty_cache()


# Convenience function
def run_catvton(
    person_image_path: str,
    cloth_image_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> Image.Image:
    """
    Run CatVTON inference on image paths
    
    Args:
        person_image_path: Path to person image
        cloth_image_path: Path to garment image
        output_path: Optional path to save output
        **kwargs: Additional arguments passed to CatVTONInference.generate()
        
    Returns:
        Generated try-on image
    """
    # Load images
    person_img = Image.open(person_image_path).convert('RGB')
    cloth_img = Image.open(cloth_image_path).convert('RGB')
    
    # Initialize and run inference
    catvton = CatVTONInference()
    result = catvton.generate(person_img, cloth_img, **kwargs)
    
    # Save if output path provided
    if output_path:
        result.save(output_path)
    
    return result