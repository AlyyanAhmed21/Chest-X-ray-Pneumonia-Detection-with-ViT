# app/image_utils.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def add_watermark(image_array: np.ndarray, text: str, confidence: float) -> Image.Image:
    """
    Adds a large, prominent, and highly readable watermark to an image.
    """
    image = Image.fromarray(image_array).convert("RGBA")
    
    txt_overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_overlay)

    is_pneumonia = (text == "PNEUMONIA")
    box_color = (220, 53, 69, 210) if is_pneumonia else (25, 135, 84, 210)
    text_color = (255, 255, 255, 255)
    shadow_color = (0, 0, 0, 180)
    
    # --- FONT FIX: Try multiple common system fonts ---
    font_size = int(image.height / 6) # Even larger font size
    font_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf", "/System/Library/Fonts/Arial Bold.ttf"]
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except IOError:
            continue
    if not font:
        print("No suitable bold font found, using default. Watermark quality will be lower.")
        font = ImageFont.load_default()

    # Define text and calculate its position
    text_to_draw = f"{text}\n{confidence:.1%}"
    try:
        _, top, _, bottom = draw.textbbox((0, 0), text_to_draw, font=font)
        text_height = bottom - top
    except Exception:
        text_height = font_size * 2 # Fallback
        
    y = 15
    
    # Draw background box
    draw.rectangle([0, y - 10, image.width, y + text_height + 10], fill=box_color)
    
    # Draw text with shadow for readability
    for line in text_to_draw.split('\n'):
        try:
            _, _, text_width, _ = draw.textbbox((0,0), line, font=font)
        except Exception:
            text_width = len(line) * font_size / 2 # Fallback
            
        x = (image.width - text_width) / 2
        # Draw shadow
        draw.text((x+2, y+2), line, font=font, fill=shadow_color)
        # Draw text
        draw.text((x, y), line, font=font, fill=text_color)
        y += text_height / 2 + 5
    
    watermarked_image = Image.alpha_composite(image, txt_overlay)
    return watermarked_image.convert("RGB")
