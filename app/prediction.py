# app/prediction.py

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Union, Any
from .image_utils import add_watermark

ImageType = Union[str, Path, bytes, np.ndarray]

class PredictionPipeline:
    def __init__(self, model_path: Path = Path("artifacts/model_training/model")):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, image_sources: List[ImageType]) -> Dict[str, Any]:
        if not image_sources:
            return {"error": "No images provided."}

        individual_results = []
        all_logits = []
        valid_images_as_np = []

        for source in image_sources:
            try:
                if isinstance(source, np.ndarray):
                    image = Image.fromarray(source).convert("RGB")
                else:
                    image = Image.open(source).convert("RGB")
                
                valid_images_as_np.append(np.array(image))
                
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    all_logits.append(logits)
                    
                    # --- NEW: Calculate individual prediction ---
                    ind_probs = torch.nn.functional.softmax(logits, dim=-1)
                    ind_conf, ind_idx = torch.max(ind_probs, dim=-1)
                    individual_results.append({
                        "prediction": self.id2label[ind_idx.item()],
                        "confidence": ind_conf.item()
                    })

            except Exception as e:
                print(f"Skipping a corrupted or invalid image file. Error: {e}")
                individual_results.append({"prediction": "Error", "confidence": 0})
                continue
        
        if not all_logits:
             return {"error": "All images were invalid."}

        # --- Aggregate Prediction (same as before) ---
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        probabilities = torch.nn.functional.softmax(avg_logits, dim=-1)
        confidence_score, predicted_class_idx = torch.max(probabilities, dim=-1)
        
        final_prediction = self.id2label[predicted_class_idx.item()]
        final_confidence = confidence_score.item()

        # --- NEW: Watermark images with their INDIVIDUAL results ---
        watermarked_images = [
            add_watermark(img_np, res["prediction"], res["confidence"])
            for img_np, res in zip(valid_images_as_np, individual_results)
            if res["prediction"] != "Error"
        ]
        
        return {
            "final_prediction": final_prediction,
            "final_confidence": final_confidence,
            "individual_results": individual_results,
            "watermarked_images": watermarked_images
        }