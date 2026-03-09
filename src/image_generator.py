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

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

#TODO check to see GOLD Standard for dataclass --> and diffefence with enums
@dataclass
class ImageGenerationConfig:
    """Configuration for image generation"""
    # Model settings
    model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ControlNet settings
    enable_controlnet: bool = False
    controlnet_type: str = "canny"
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny"
    controlnet_default_scale: float = 0.85
    controlnet_guess_mode: bool = False
    controlnet_conditioning_start: float = 0.0
    controlnet_conditioning_end: float = 1.0
    
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
        self._controlnet_img2img_pipe = None
        self._warned_controlnet_family_mismatch = False
        
        print(f"🎨 Image Generator")
        print(f"   Device: {self.config.device}")
        print(f"   Model: {self.config.model_id}")
        if self.config.enable_controlnet:
            print(f"   ControlNet: {self.config.controlnet_model_id} ({self.config.controlnet_type})")
            self._warn_controlnet_family_mismatch()
        
        # Preload the most likely image-to-image pipeline.
        if self.config.enable_controlnet:
            self._load_controlnet_img2img_pipeline()
        else:
            self._load_img2img_pipeline()

    def _torch_dtype(self):
        return torch.float16 if self.config.device == "cuda" else torch.float32

    def _load_vae(self):
        return AutoencoderKL.from_pretrained(
            self.config.vae_id,
            torch_dtype=self._torch_dtype(),
        )

    def _optimize_pipeline(self, pipe):
        if self.config.device != "cuda":
            return
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    @staticmethod
    def _infer_model_family(model_id: str) -> str:
        model = (model_id or "").lower()
        if "sdxl" in model or "xl" in model:
            return "sdxl"
        if "2.1" in model or "2-1" in model or "v2" in model:
            return "sd2"
        return "sd1"

    def _warn_controlnet_family_mismatch(self):
        if self._warned_controlnet_family_mismatch:
            return
        base_family = self._infer_model_family(self.config.model_id)
        control_family = self._infer_model_family(self.config.controlnet_model_id)
        if base_family != control_family:
            print(
                "⚠️  Model family mismatch detected: "
                f"base='{self.config.model_id}' ({base_family}) vs "
                f"controlnet='{self.config.controlnet_model_id}' ({control_family}). "
                "Generation may fail or produce poor results."
            )
            self._warned_controlnet_family_mismatch = True
    
    def _load_txt2img_pipeline(self):
        """Load text-to-image pipeline (lazy loading)"""
        if self._txt2img_pipe is not None:
            return

        
        print("   Loading txt2img pipeline...")
        
        # Load VAE
        vae = self._load_vae()
        
        # Load pipeline
        self._txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            vae=vae,
            torch_dtype=self._torch_dtype(),
            safety_checker=None,
            requires_safety_checker=False
        )
        self._txt2img_pipe = self._txt2img_pipe.to(self.config.device)
        
        # Enable optimizations
        self._optimize_pipeline(self._txt2img_pipe)
        
        print("txt2img ready")
    
    def _load_img2img_pipeline(self):
        """Load image-to-image pipeline"""
        if self._img2img_pipe is not None:
            return
        
        print("Loading img2img pipeline...")
        
        # Load VAE
        vae = self._load_vae()
        
        # Load pipeline
        self._img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.model_id,
            vae=vae,
            torch_dtype=self._torch_dtype(),
            safety_checker=None,
            requires_safety_checker=False
        )
        self._img2img_pipe = self._img2img_pipe.to(self.config.device)
        
        # Enable optimizations
        self._optimize_pipeline(self._img2img_pipe)
        
        print("   ✓ img2img ready")

    def _load_controlnet_img2img_pipeline(self):
        """Load image-to-image pipeline with ControlNet."""
        if self._controlnet_img2img_pipe is not None:
            return
        if self.config.controlnet_type != "canny":
            raise ValueError(
                f"Unsupported controlnet_type='{self.config.controlnet_type}'. "
                "Only 'canny' is supported in this implementation."
            )

        print("Loading ControlNet img2img pipeline...")
        self._warn_controlnet_family_mismatch()

        vae = self._load_vae()
        controlnet = ControlNetModel.from_pretrained(
            self.config.controlnet_model_id,
            torch_dtype=self._torch_dtype(),
        )

        self._controlnet_img2img_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.config.model_id,
            vae=vae,
            controlnet=controlnet,
            torch_dtype=self._torch_dtype(),
            safety_checker=None,
            requires_safety_checker=False,
        )
        self._controlnet_img2img_pipe = self._controlnet_img2img_pipe.to(self.config.device)
        self._optimize_pipeline(self._controlnet_img2img_pipe)
        print("   ✓ ControlNet img2img ready")
    
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
        seed: int = None,
        control_image: Optional[Image.Image] = None,
        controlnet_conditioning_scale: Optional[float] = None,
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
        # Use config defaults if not specified
        negative_prompt = negative_prompt or self.config.negative_prompt
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        
        # Create generator for seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(seed)

        if self.config.enable_controlnet:
            if control_image is None:
                raise ValueError(
                    "ControlNet is enabled but no control_image was provided. "
                    "Pass a conditioning image (for example, canny edges)."
                )
            self._load_controlnet_img2img_pipeline()
            conditioning_scale = (
                self.config.controlnet_default_scale
                if controlnet_conditioning_scale is None
                else float(controlnet_conditioning_scale)
            )
            result = self._controlnet_img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                control_image=control_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guess_mode=self.config.controlnet_guess_mode,
                controlnet_conditioning_scale=conditioning_scale,
                control_guidance_start=float(self.config.controlnet_conditioning_start),
                control_guidance_end=float(self.config.controlnet_conditioning_end),
            )
        else:
            self._load_img2img_pipeline()
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

        if self._controlnet_img2img_pipe is not None:
            del self._controlnet_img2img_pipe
            self._controlnet_img2img_pipe = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("   ✓ Cleaned up GPU memory")
