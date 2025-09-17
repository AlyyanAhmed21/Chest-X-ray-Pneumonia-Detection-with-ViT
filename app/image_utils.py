# app/image_utils.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

def add_watermark(image_array: np.ndarray, text: str, confidence: float) -> Image.Image:
    """
    Adds a large, prominent, and consistently sized text overlay to an image.
    """
    image = Image.fromarray(image_array).convert("RGBA")
    txt_overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_overlay)

    is_pneumonia = (text == "PNEUMONIA")
    box_color = (220, 53, 69, 210) if is_pneumonia else (25, 135, 84, 210)
    text_color = (255, 255, 255, 255)
    shadow_color = (0, 0, 0, 180)
    
    # --- FONT FIX: Use a bundled font file for reliability ---
    font_path = Path(__file__).parent / "Roboto-Bold.ttf"
    
    # --- SIZING FIX: Make the box a fixed proportion of the image width ---
    box_width = int(image.width * 0.4) # Box will be 40% of the image width
    box_height = int(box_width * 0.4)  # Maintain an aspect ratio for the box
    
    # Start with a large font size and shrink it until the text fits in the box
    font_size = int(box_height / 2.5)
    font = ImageFont.truetype(str(font_path), font_size)
    
    text_to_draw = f"{text}\n{confidence:.1%}"
    
    # Adjust font size until text fits within the box width
    while font.getbbox(text_to_draw, align="center")[2] > box_width * 0.9:
        font_size -= 2
        font = ImageFont.truetype(str(font_path), font_size)

    # --- POSITIONING: Place box in the top-left corner ---
    box_position = [10, 10, 10 + box_width, 10 + box_height]
    draw.rectangle(box_position, fill=box_color)

    # Center the text within the box
    text_bbox = draw.textbbox((0, 0), text_to_draw, font=font, align="center")
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = box_position[0] + (box_width - text_width) / 2
    text_y = box_position[1] + (box_height - text_height) / 2
    
    # Draw text with shadow
    draw.text((text_x + 2, text_y + 2), text_to_draw, font=font, fill=shadow_color, align="center")
    draw.text((text_x, text_y), text_to_draw, font=font, fill=text_color, align="center")
    
    watermarked_image = Image.alpha_composite(image, txt_overlay)
    return watermarked_image.convert("RGB")
