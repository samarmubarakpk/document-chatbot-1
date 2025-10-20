# advanced_image_generator.py - MULTI-MODEL IMAGE GENERATION WITH ZERO ERROR TOLERANCE
"""
Advanced Image Generation System with:
1. Multiple AI image generation models (DALL-E 3, Stable Diffusion, Google Imagen)
2. Fallback mechanism for reliability
3. Uses top-tier visual analysis for perfect reconstruction
4. Zero error tolerance - exhaustive prompt engineering
"""

from typing import Dict, Any, Optional, List, Literal
import requests
import base64
import io
from PIL import Image
from openai import AsyncOpenAI, OpenAI
import asyncio
import os
from datetime import datetime
import uuid

class AdvancedImageGenerator:
    """
    Multi-model image generator with zero error tolerance.
    Supports: DALL-E 3, Stable Diffusion, Google Imagen (when available)
    """
    
    def __init__(self, openai_api_key: str, stability_api_key: Optional[str] = None):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.openai_async_client = AsyncOpenAI(api_key=openai_api_key)
        self.stability_api_key = stability_api_key
        
        # Model priorities (try in order)
        self.model_priority = [
            "dalle3",          # OpenAI DALL-E 3
            "stable-diffusion",  # Stability AI SDXL
            # "google-imagen"  # Google Imagen (when API available)
        ]
    
    async def generate_diagram_from_analysis(
        self,
        visual_analysis: Dict[str, Any],
        query_context: str = "",
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🎯 MAIN METHOD: Generate diagram from comprehensive visual analysis
        
        Args:
            visual_analysis: The comprehensive analysis from document_processor_universal.py
            query_context: User's query for additional context
            preferred_model: Force specific model (or None for auto)
        
        Returns:
            {
                "image_bytes": bytes,
                "image_path": str (saved path),
                "model_used": str,
                "generation_prompt": str (the enhanced prompt used),
                "success": bool,
                "error": Optional[str]
            }
        """
        print(f"\n{'='*80}")
        print(f"🎨 ADVANCED IMAGE GENERATION")
        print(f"{'='*80}")
        
        # Step 1: Enhance the reconstruction prompt
        enhanced_prompt = await self._enhance_reconstruction_prompt(
            visual_analysis, 
            query_context
        )
        
        print(f"📝 Enhanced prompt created ({len(enhanced_prompt)} chars)")
        print(f"Preview: {enhanced_prompt[:200]}...\n")
        
        # Step 2: Try models in priority order
        models_to_try = [preferred_model] if preferred_model else self.model_priority
        
        for model in models_to_try:
            print(f"🔄 Attempting generation with: {model.upper()}")
            
            try:
                if model == "dalle3":
                    result = await self._generate_with_dalle3(enhanced_prompt)
                elif model == "stable-diffusion":
                    result = await self._generate_with_stable_diffusion(enhanced_prompt)
                # elif model == "google-imagen":
                #     result = await self._generate_with_google_imagen(enhanced_prompt)
                else:
                    continue
                
                if result["success"]:
                    print(f"✅ Successfully generated with {model.upper()}")
                    result["model_used"] = model
                    result["generation_prompt"] = enhanced_prompt
                    return result
                else:
                    print(f"❌ {model.upper()} failed: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"❌ {model.upper()} error: {str(e)}")
                continue
        
        # All models failed
        print(f"{'='*80}")
        print(f"❌ ALL MODELS FAILED")
        print(f"{'='*80}\n")
        
        return {
            "success": False,
            "error": "All image generation models failed",
            "image_bytes": None,
            "image_path": None,
            "model_used": None,
            "generation_prompt": enhanced_prompt
        }
    
    async def _enhance_reconstruction_prompt(
        self, 
        visual_analysis: Dict[str, Any],
        query_context: str
    ) -> str:
        """
        🚀 CRITICAL: Enhance the reconstruction prompt using AI
        
        Takes the reconstruction_prompt from visual analysis and makes it PERFECT
        for image generation with exhaustive detail.
        """
        base_prompt = visual_analysis.get("reconstruction_prompt", "")
        
        if not base_prompt:
            # Fallback: build from other analysis components
            base_prompt = f"""
Visual Type: {', '.join(visual_analysis.get('visual_types', []))}
Domain: {visual_analysis.get('domain', 'general')}

Description: {visual_analysis.get('comprehensive_description', '')}

Spatial Layout: {visual_analysis.get('spatial_layout', '')}

Colors and Styles: {visual_analysis.get('colors_and_styles', '')}

Text Content: {visual_analysis.get('text_content', '')}

Key Elements:
{chr(10).join(f'- {elem}' for elem in visual_analysis.get('key_elements', []))}
"""
        
        # Use GPT-4 to enhance the prompt for image generation
        enhancement_request = f"""You are a prompt engineering expert for AI image generation.

Take this visual description and create a PERFECT prompt for an AI image generator (DALL-E 3, Stable Diffusion, etc.) that will recreate the visual with ZERO error.

Original Description:
{base_prompt}

User Context: {query_context if query_context else 'None'}

REQUIREMENTS FOR THE ENHANCED PROMPT:
1. Start with clear art style directive: "Professional technical documentation style" or similar
2. Specify exact layout: "Top to bottom arrangement" or "Left to right flow"
3. List EVERY component with precise descriptions
4. Include ALL text labels EXACTLY as they appear
5. Specify all colors explicitly
6. Describe line styles (solid, dashed, thickness)
7. Include shapes (rectangles, circles, arrows, etc.)
8. Specify spatial relationships precisely
9. Add technical drawing specifications if needed
10. Include background color and overall style

OUTPUT FORMAT:
Create a single, comprehensive prompt of 300-500 words that an image generator can use to create a pixel-perfect recreation.

Enhanced Prompt:"""
        
        try:
            response = await self.openai_async_client.chat.completions.create(
                model="gpt-4o",  # Use best model for prompt enhancement
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating detailed prompts for AI image generation. Your prompts must be exhaustive, precise, and result in perfect visual recreation."
                    },
                    {
                        "role": "user",
                        "content": enhancement_request
                    }
                ],
                max_tokens=1500,
                temperature=0.3  # Some creativity, but mostly precise
            )
            
            enhanced_prompt = response.choices[0].message.content.strip()
            
            # Add technical drawing boilerplate if not present
            if "professional" not in enhanced_prompt.lower():
                enhanced_prompt = f"Professional technical documentation style. Clean, precise, high-quality. {enhanced_prompt}"
            
            return enhanced_prompt
            
        except Exception as e:
            print(f"⚠️  Prompt enhancement failed, using base prompt: {str(e)}")
            return base_prompt
    
    async def _generate_with_dalle3(self, prompt: str) -> Dict[str, Any]:
        """
        Generate image using DALL-E 3
        """
        try:
            # DALL-E 3 has 4000 char limit - truncate if needed
            if len(prompt) > 4000:
                print(f"⚠️  Prompt too long ({len(prompt)} chars), truncating to 4000")
                prompt = prompt[:3997] + "..."
            
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="hd",  # Use HD quality
                n=1,
            )
            
            # Download image
            image_url = response.data[0].url
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            image_bytes = image_response.content
            
            # Save to outputs
            image_filename = f"diagram_{uuid.uuid4()}.png"
            image_path = f"/mnt/user-data/outputs/{image_filename}"
            
            os.makedirs("/mnt/user-data/outputs", exist_ok=True)
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            return {
                "success": True,
                "image_bytes": image_bytes,
                "image_path": image_path,
                "image_filename": image_filename,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "image_bytes": None,
                "image_path": None,
                "error": str(e)
            }
    
    async def _generate_with_stable_diffusion(self, prompt: str) -> Dict[str, Any]:
        """
        Generate image using Stability AI's Stable Diffusion XL
        Requires STABILITY_API_KEY environment variable
        """
        if not self.stability_api_key:
            return {
                "success": False,
                "error": "Stability AI API key not configured"
            }
        
        try:
            # Stability AI API endpoint
            url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
            
            headers = {
                "authorization": f"Bearer {self.stability_api_key}",
                "accept": "image/*"
            }
            
            # Prepare request
            data = {
                "prompt": prompt,
                "output_format": "png",
                "mode": "text-to-image",
                "aspect_ratio": "1:1",
                "model": "sd3-large"  # Best quality model
            }
            
            # Make request
            response = requests.post(url, headers=headers, files={"none": ''}, data=data, timeout=60)
            response.raise_for_status()
            
            image_bytes = response.content
            
            # Save to outputs
            image_filename = f"diagram_{uuid.uuid4()}.png"
            image_path = f"/mnt/user-data/outputs/{image_filename}"
            
            os.makedirs("/mnt/user-data/outputs", exist_ok=True)
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            return {
                "success": True,
                "image_bytes": image_bytes,
                "image_path": image_path,
                "image_filename": image_filename,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "image_bytes": None,
                "image_path": None,
                "error": str(e)
            }
    
    async def generate_simple_diagram(
        self,
        description: str,
        style: str = "technical",
        model: str = "dalle3"
    ) -> Dict[str, Any]:
        """
        Simple diagram generation from text description (for queries without visual analysis)
        """
        print(f"\n{'='*80}")
        print(f"🎨 SIMPLE DIAGRAM GENERATION")
        print(f"{'='*80}")
        
        # Create basic visual analysis structure
        visual_analysis = {
            "reconstruction_prompt": f"""Create a professional {style} diagram.

{description}

Style: Clean, professional, technical documentation quality
Background: White
Lines: Clear, precise
Text: Readable labels
Layout: Organized and clear"""
        }
        
        return await self.generate_diagram_from_analysis(
            visual_analysis,
            query_context=description,
            preferred_model=model
        )


class ImageGenerationConfig:
    """Configuration for image generation"""
    
    # Model priorities and fallbacks
    MODEL_PRIORITY = [
        "dalle3",
        "stable-diffusion"
    ]
    
    # Quality settings
    DALLE3_QUALITY = "hd"  # "standard" or "hd"
    DALLE3_SIZE = "1024x1024"  # "1024x1024", "1792x1024", "1024x1792"
    
    SD_MODEL = "sd3-large"  # "sd3-large", "sd3-medium", "sdxl-1.0"
    SD_ASPECT_RATIO = "1:1"  # "1:1", "16:9", "9:16", etc.
    
    # Prompt enhancement
    ENHANCE_PROMPTS = True  # Use GPT-4 to enhance prompts
    MAX_PROMPT_LENGTH = 4000  # For DALL-E 3
    
    # Output settings
    OUTPUT_FORMAT = "png"
    OUTPUT_DIRECTORY = "/mnt/user-data/outputs"
    
    # Retry settings
    MAX_RETRIES = 2
    RETRY_DELAY = 2  # seconds


# Example usage
"""
# Initialize
generator = AdvancedImageGenerator(
    openai_api_key="your-key",
    stability_api_key="your-key"  # optional
)

# Generate from visual analysis
result = await generator.generate_diagram_from_analysis(
    visual_analysis=comprehensive_visual_analysis,
    query_context="User wants a network diagram"
)

if result["success"]:
    print(f"Image saved to: {result['image_path']}")
    # Serve via HTTP: http://localhost:8000/images/{result['image_filename']}
else:
    print(f"Generation failed: {result['error']}")
"""