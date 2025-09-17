# app/image_utils.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def add_watermark(image_array: np.ndarray, text: str, confidence: float) -> Image.Image:
    """
    Adds a large, prominent, and professional-looking watermark to an image.
    """
    image = Image.fromarray(image_array).convert("RGBA")
    
    txt_overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_overlay)

    # Define watermark properties
    is_pneumonia = (text == "PNEUMONIA")
    box_color = (220, 53, 69, 200) if is_pneumonia else (25, 135, 84, 200) # Red/Green with higher opacity
    text_color = (255, 255, 255, 255)
    
    # --- FONT FIX: Make font size proportional and larger ---
    try:
        # On many systems (including HF Spaces), Arial is available.
        font_path = "arialbd.ttf"
        font_size = int(image.height / 7) # Made font significantly larger
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font '{font_path}' not found, using default. Watermark quality may be lower.")
        # Fallback if font is not on the system
        font_size = int(image.height / 7)
        try:
            # Try a generic sans-serif font
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

    # Define text
    text_to_draw = f"{text}\n{confidence:.1%}"
    
    # --- POSITIONING FIX: Center the watermark ---
    try:
        # Use getbbox for modern Pillow versions to get precise text dimensions
        left, top, right, bottom = draw.textbbox((0, 0), text_to_draw, font=font)
        text_width = right - left
        text_height = bottom - top
    except AttributeError:
        # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(text_to_draw, font=font)
        
    x = (image.width - text_width) / 2
    y = (image.height - text_height) / 2
    position = (x, y)
    
    # Add padding to the background box
    padding = font_size / 4
    box_position = [
        x - padding,
        y - padding,
        x + text_width + padding,
        y + text_height + padding
    ]

    # Draw the rectangle and the text
    draw.rectangle(box_position, fill=box_color)
    draw.text(position, text_to_draw, font=font, fill=text_color, align="center")
    
    watermarked_image = Image.alpha_composite(image, txt_overlay)
    
    return watermarked_image.convert("RGB")
