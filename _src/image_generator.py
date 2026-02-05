"""
Image Generator - Core Stable Diffusion wrapper

Simple, clean interface for generating images with Stable Diffusion.
Supports both text-to-image and image-to-image generation.

Design principles:
- Single Responsibility: Only handles image generation
- Simple API: Minimal configuration needed
- Clear separation: txt2img and img2img are distinct operations
"""

import torch
from PIL import Image
from typing import Optional
from dataclasses import dataclass


@dataclass
class ImageGenerationConfig:
    """Configuration for image generation"""
    # Model settings
    model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generation defaults
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    negative_prompt: str = "blurry, bad quality, distorted, ugly"


class ImageGenerator:
    """
    Generates images using Stable Diffusion.
    
    Handles both text-to-image (first frame) and image-to-image (subsequent frames).
    
    Usage:
        config = ImageGenerationConfig()
        generator = ImageGenerator(config)
        
        # Generate first frame from text
        image = generator.generate_from_text("a beautiful sunset")
        
        # Generate next frame from previous image
        next_image = generator.generate_from_image(
            image, 
            "a beautiful sunset with clouds",
            strength=0.6
        )
    """
    
    def __init__(self, config: ImageGenerationConfig = None):
        """
        Initialize the generator.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or ImageGenerationConfig()
        self._txt2img_pipe = None
        self._img2img_pipe = None
        
        print(f"🎨 Image Generator")
        print(f"   Device: {self.config.device}")
        print(f"   Model: {self.config.model_id}")
        
        # Load img2img pipeline (most common)
        self._load_img2img_pipeline()
    
    def _load_txt2img_pipeline(self):
        """Load text-to-image pipeline (lazy loading)"""
        if self._txt2img_pipe is not None:
            return
        
        from diffusers import StableDiffusionPipeline, AutoencoderKL
        
        print("   Loading txt2img pipeline...")
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            self.config.vae_id,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
        )
        
        # Load pipeline
        self._txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            vae=vae,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self._txt2img_pipe = self._txt2img_pipe.to(self.config.device)
        
        # Enable optimizations
        if self.config.device == "cuda":
            self._txt2img_pipe.enable_attention_slicing()
            try:
                self._txt2img_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        
        print("   ✓ txt2img ready")
    
    def _load_img2img_pipeline(self):
        """Load image-to-image pipeline"""
        if self._img2img_pipe is not None:
            return
        
        from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
        
        print("   Loading img2img pipeline...")
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            self.config.vae_id,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
        )
        
        # Load pipeline
        self._img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.model_id,
            vae=vae,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self._img2img_pipe = self._img2img_pipe.to(self.config.device)
        
        # Enable optimizations
        if self.config.device == "cuda":
            self._img2img_pipe.enable_attention_slicing()
            try:
                self._img2img_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        
        print("   ✓ img2img ready")
    
    def generate_from_text(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = None,
        height: int = None,
        guidance_scale: float = None,
        num_inference_steps: int = None,
        seed: int = None
    ) -> Image.Image:
        """
        Generate image from text prompt (text-to-image).
        
        Use this for the first frame of an animation.
        
        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid (uses default if None)
            width: Image width (uses config default if None)
            height: Image height (uses config default if None)
            guidance_scale: CFG scale (uses config default if None)
            num_inference_steps: Denoising steps (uses config default if None)
            seed: Random seed (None for random)
            
        Returns:
            Generated PIL Image
        """
        self._load_txt2img_pipeline()
        
        # Use config defaults if not specified
        negative_prompt = negative_prompt or self.config.negative_prompt
        width = width or self.config.width
        height = height or self.config.height
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        
        # Create generator for seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        # Generate image
        result = self._txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        
        return result.images[0]
    
    def generate_from_image(
        self,
        init_image: Image.Image,
        prompt: str,
        strength: float = 0.6,
        negative_prompt: str = None,
        guidance_scale: float = None,
        num_inference_steps: int = None,
        seed: int = None
    ) -> Image.Image:
        """
        Generate image from existing image (image-to-image).
        
        Use this for subsequent frames in an animation.
        
        Args:
            init_image: Input image to transform
            prompt: Text description of desired output
            strength: How much to change (0.0=no change, 1.0=complete change)
            negative_prompt: What to avoid (uses default if None)
            guidance_scale: CFG scale (uses config default if None)
            num_inference_steps: Denoising steps (uses config default if None)
            seed: Random seed (None for random)
            
        Returns:
            Generated PIL Image
        """
        self._load_img2img_pipeline()
        
        # Use config defaults if not specified
        negative_prompt = negative_prompt or self.config.negative_prompt
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        
        # Create generator for seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        # Generate image
        result = self._img2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        
        return result.images[0]
    
    def cleanup(self):
        """Free GPU memory"""
        if self._txt2img_pipe is not None:
            del self._txt2img_pipe
            self._txt2img_pipe = None
        
        if self._img2img_pipe is not None:
            del self._img2img_pipe
            self._img2img_pipe = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("   ✓ Cleaned up GPU memory")
