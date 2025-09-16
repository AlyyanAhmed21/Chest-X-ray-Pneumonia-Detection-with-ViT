# app/prediction.py

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Union

# Define a type hint for the input, which can be a path or bytes
ImageType = Union[str, Path, bytes]

class PredictionPipeline:
    def __init__(self, model_path: Path = Path("artifacts/model_training/model")):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, image_sources: List[ImageType]) -> Dict[str, Union[str, float]]:
        if not image_sources:
            return {"prediction": "Error", "confidence": 0.0, "details": "No images provided."}

        all_logits = []
        for source in image_sources:
            try:
                # --- THIS IS THE FIX ---
                # The Image.open() function can handle both paths and byte streams.
                # No special handling is needed.
                image = Image.open(source).convert("RGB")
                
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    all_logits.append(outputs.logits)
            except Exception as e:
                print(f"Skipping a corrupted or invalid image file. Error: {e}")
                continue
        
        if not all_logits:
             return {"prediction": "Error", "confidence": 0.0, "details": "All provided images were invalid."}

        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        probabilities = torch.nn.functional.softmax(avg_logits, dim=-1)
        confidence_score, predicted_class_idx = torch.max(probabilities, dim=-1)
        predicted_label = self.id2label[predicted_class_idx.item()]
        
        return {
            "prediction": predicted_label,
            "confidence": confidence_score.item()
        }