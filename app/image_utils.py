# app/image_utils.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

def add_watermark(image_array: np.ndarray, text: str, confidence: float) -> Image.Image:
    """
    Adds a large, prominent, and highly readable text overlay to an image.
    """
    image = Image.fromarray(image_array).convert("RGB")
    draw = ImageDraw.Draw(image)

    is_pneumonia = (text == "PNEUMONIA")
    text_color = (220, 53, 69) if is_pneumonia else (25, 135, 84) # Solid Red or Green
    shadow_color = (0, 0, 0)
    
    # --- FONT FIX: Use a bundled font file for reliability ---
    font_size = int(image.height / 6) # Make it very large
    font_path = Path(__file__).parent / "Roboto-Bold.ttf" # Assumes font is in the same 'app' folder
    
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except IOError:
        print(f"Font at '{font_path}' not found, falling back to default.")
        font = ImageFont.load_default()

    # Text to draw
    text_to_draw = f"{text}\n{confidence:.1%}"
    
    # --- POSITIONING: Center the text block ---
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text_to_draw, font=font, align="center")
        text_width = right - left
        text_height = bottom - top
    except Exception:
        # Fallback for older Pillow
        text_width, text_height = draw.textsize(text_to_draw, font=font)
        
    x = (image.width - text_width) / 2
    y = (image.height - text_height) / 2
    
    # Draw shadow/outline for contrast
    offset = font_size // 20
    draw.text((x-offset, y-offset), text_to_draw, font=font, fill=shadow_color, align="center")
    draw.text((x+offset, y-offset), text_to_draw, font=font, fill=shadow_color, align="center")
    draw.text((x-offset, y+offset), text_to_draw, font=font, fill=shadow_color, align="center")
    draw.text((x+offset, y+offset), text_to_draw, font=font, fill=shadow_color, align="center")

    # Draw main text
    draw.text((x, y), text_to_draw, font=font, fill=text_color, align="center")
    
    return image
