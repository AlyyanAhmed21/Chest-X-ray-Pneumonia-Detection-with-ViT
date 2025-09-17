# app/image_utils.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

def add_watermark(image_array: np.ndarray, text: str, confidence: float) -> Image.Image:
    """
    Adds a large, prominent, and professional-looking text banner to the top of an image.
    """
    image = Image.fromarray(image_array).convert("RGBA")
    
    # Create a transparent overlay layer
    txt_overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_overlay)

    # --- DEFINE BANNER AND TEXT PROPERTIES ---
    is_pneumonia = (text == "PNEUMONIA")
    banner_color = (220, 53, 69, 210) if is_pneumonia else (25, 135, 84, 210) # Red/Green
    text_color = (255, 255, 255)
    
    # --- SIZING FIX: Banner height is 1/8th of the image height ---
    banner_height = int(image.height / 8)
    
    # --- FONT FIX: Use a bundled font and scale it to fit the banner ---
    font_path = Path(__file__).parent / "Roboto-Bold.ttf"
    try:
        # Start with a font size that's a good proportion of the banner height
        font_size = int(banner_height * 0.6)
        font = ImageFont.truetype(str(font_path), font_size)
    except IOError:
        print(f"Font at '{font_path}' not found, using default.")
        font = ImageFont.load_default()

    # Text to draw
    text_to_draw = f"{text} | Confidence: {confidence:.1%}"
    
    # --- DRAW THE BANNER ---
    draw.rectangle([0, 0, image.width, banner_height], fill=banner_color)

    # --- POSITIONING: Center the text within the banner ---
    try:
        # getbbox is the modern, accurate way to measure text
        _, top, _, bottom = draw.textbbox((0, 0), text_to_draw, font=font)
        text_w = font.getlength(text_to_draw)
        text_h = bottom - top
    except Exception:
        # Fallback for older Pillow
        text_w = font.getsize(text_to_draw)[0]
        text_h = font.getsize(text_to_draw)[1]
        
    text_x = (image.width - text_w) / 2
    text_y = (banner_height - text_h) / 2
    
    # Draw text
    draw.text((text_x, text_y), text_to_draw, font=font, fill=text_color)
    
    # Composite the overlay onto the original image
    watermarked_image = Image.alpha_composite(image, txt_overlay)
    
    return watermarked_image.convert("RGB")
