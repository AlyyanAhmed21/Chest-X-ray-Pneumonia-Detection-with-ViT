# app/image_utils.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

def add_watermark(image_array: np.ndarray, text: str, confidence: float) -> Image.Image:
    """
    Adds a large, prominent, and consistently sized text banner to the top of an image.
    """
    image = Image.fromarray(image_array).convert("RGBA")
    
    txt_overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_overlay)

    is_pneumonia = (text == "PNEUMONIA")
    banner_color = (220, 53, 69, 215) if is_pneumonia else (25, 135, 84, 215)
    text_color = (255, 255, 255)
    
    # --- SIZING FIX: Banner is 1/7th of the image height ---
    banner_height = int(image.height / 7)
    
    # --- FONT FIX: Start large and shrink to fit ---
    font_path = Path(__file__).parent / "Roboto-Bold.ttf"
    font_size = int(banner_height * 0.75) # Start with a large font
    
    text_to_draw = f"{text} | {confidence:.1%}"
    
    # Shrink font until the text fits within 90% of the image width
    while True:
        try:
            font = ImageFont.truetype(str(font_path), font_size)
            text_width = font.getlength(text_to_draw)
            if text_width < image.width * 0.9:
                break
            font_size -= 2
        except IOError:
            print(f"Font at '{font_path}' not found, using default.")
            font = ImageFont.load_default()
            break # Exit loop if font fails
            
    # Draw banner
    draw.rectangle([0, 0, image.width, banner_height], fill=banner_color)

    # Center the text within the banner
    text_height = font.getbbox(text_to_draw)[3]
    text_x = (image.width - text_width) / 2
    text_y = (banner_height - text_height) / 2
    
    # Draw text
    draw.text((text_x, text_y), text_to_draw, font=font, fill=text_color)
    
    watermarked_image = Image.alpha_composite(image, txt_overlay)
    
    return watermarked_image.convert("RGB")
