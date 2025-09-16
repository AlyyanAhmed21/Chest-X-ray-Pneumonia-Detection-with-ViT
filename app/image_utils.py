# app/image_utils.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def add_watermark(image_array: np.ndarray, text: str, confidence: float) -> Image.Image:
    """
    Adds a translucent watermark to an image with the prediction result and confidence.

    Args:
        image_array: The input image as a NumPy array.
        text: The prediction text (e.g., "NORMAL" or "PNEUMONIA").
        confidence: The confidence score of the prediction.

    Returns:
        A PIL Image object with the watermark applied.
    """
    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_array).convert("RGBA")
    
    # Create a transparent overlay for the text
    txt_overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_overlay)

    # Define watermark properties
    is_pneumonia = (text == "PNEUMONIA")
    box_color = (220, 53, 69, 180) if is_pneumonia else (25, 135, 84, 180) # Red for Pneumonia, Green for Normal
    text_color = (255, 255, 255, 255)
    
    # Define font (uses a default if a specific .ttf is not found)
    try:
        font_size = int(image.height / 8)
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        print("Arial Bold font not found, using default. Watermark quality may be lower.")
        font_size = int(image.height / 8)
        font = ImageFont.load_default()

    # Define text and box position
    text_to_draw = f"{text}\n{confidence:.1%}"
    
    # Get text size
    try:
        # Use getbbox for modern Pillow versions
        _, _, text_width, text_height = draw.textbbox((0, 0), text_to_draw, font=font)
    except AttributeError:
        # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(text_to_draw, font=font)
        
    position = (20, 20) # Top-left corner with some padding
    box_position = [
        position[0] - 10,
        position[1] - 10,
        position[0] + text_width + 10,
        position[1] + text_height + 10
    ]

    # Draw the semi-transparent rectangle and the text
    draw.rectangle(box_position, fill=box_color)
    draw.text(position, text_to_draw, font=font, fill=text_color)
    
    # Combine the overlay with the original image
    watermarked_image = Image.alpha_composite(image, txt_overlay)
    
    return watermarked_image.convert("RGB")