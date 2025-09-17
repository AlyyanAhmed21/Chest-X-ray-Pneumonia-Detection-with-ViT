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

    # Define watermark properties
    is_pneumonia = (text == "PNEUMONIA")
    box_color = (220, 53, 69, 210) if is_pneumonia else (25, 135, 84, 210) # Red/Green with high opacity
    text_color = (255, 255, 255, 255)
    shadow_color = (0, 0, 0, 180)
    
    # --- FONT FIX: Use a reliable font path for Linux/HF Spaces ---
    font_size = int(image.height / 7) # Large font size
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font at '{font_path}' not found, falling back to default.")
        font = ImageFont.load_default() # Fallback

    # Text to draw
    text_to_draw = f"{text}\n{confidence:.1%}"
    
    # --- POSITIONING: Center the watermark ---
    try:
        # getbbox is the modern, accurate way to measure text
        _, top, _, bottom = draw.textbbox((0, 0), text_to_draw, font=font, align="center")
        text_height = bottom - top
    except Exception:
        text_height = font_size * 2.2 # Estimate height if getbbox fails

    # Position the entire block at the top with a margin
    y_start = image.height * 0.05 
    
    # Draw background box across the full width
    draw.rectangle([0, y_start, image.width, y_start + text_height + 20], fill=box_color)
    
    # Draw each line of text centered horizontally
    y = y_start + 10
    for i, line in enumerate(text_to_draw.split('\n')):
        try:
            _, _, line_width, _ = draw.textbbox((0,0), line, font=font)
        except Exception:
            line_width = len(line) * font_size / 2 # Fallback
            
        x = (image.width - line_width) / 2
        
        # Draw shadow for contrast
        draw.text((x+3, y+3), line, font=font, fill=shadow_color, align="center")
        # Draw main text
        draw.text((x, y), line, font=font, fill=text_color, align="center")
        
        # Adjust y position for the next line
        y += font_size * 1.1 if i == 0 else font_size
    
    watermarked_image = Image.alpha_composite(image, txt_overlay)
    
    return watermarked_image.convert("RGB")
