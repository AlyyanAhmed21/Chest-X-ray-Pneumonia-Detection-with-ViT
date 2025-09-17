# app/prediction.py

import torch
from transformers import ViTImageProcessor, ViTForImageClassification, AutoImageProcessor, ResNetForImageClassification
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Union, Any
from .image_utils import add_watermark

ImageType = Union[str, Path, bytes, np.ndarray]

class PredictionPipeline:
    def __init__(self, model_path: Path = Path("artifacts/model_training/model")):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Pneumonia Model (our fine-tuned model) ---
        self.pneumonia_processor = ViTImageProcessor.from_pretrained(model_path)
        self.pneumonia_model = ViTForImageClassification.from_pretrained(model_path).to(self.device)
        self.pneumonia_model.eval()
        self.id2label = self.pneumonia_model.config.id2label

        # --- Sanity Check Model (general purpose) ---
        # This model knows what many things are, including X-rays.
        self.sanity_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.sanity_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(self.device)
        self.sanity_model.eval()

    def is_likely_xray(self, image: Image.Image) -> bool:
        """
        Uses the general-purpose ResNet-50 model to check if the image
        is likely a chest X-ray.
        """
        with torch.no_grad():
            inputs = self.sanity_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.sanity_model(**inputs)
            logits = outputs.logits
            
            # Get the top 5 predicted classes
            top5_probs, top5_indices = torch.topk(logits.softmax(-1), 5)
            
            # The model's labels are in its config. We look for 'x-ray' or 'chest'.
            for idx in top5_indices[0]:
                label = self.sanity_model.config.id2label[idx.item()].lower()
                if "x-ray" in label or "chest" in label or "radiograph" in label:
                    print(f"Sanity check passed: Image classified as '{label}'")
                    return True
        
        print("Sanity check failed: Image is not classified as an X-ray.")
        return False

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
                
                # --- NEW: Perform the sanity check first! ---
                if not self.is_likely_xray(image):
                    raise ValueError("Image does not appear to be a chest X-ray.")

                valid_images_as_np.append(np.array(image))
                
                inputs = self.pneumonia_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.pneumonia_model(**inputs)
                    logits = outputs.logits
                    all_logits.append(logits)
                    
                    ind_probs = torch.nn.functional.softmax(logits, dim=-1)
                    ind_conf, ind_idx = torch.max(ind_probs, dim=-1)
                    individual_results.append({
                        "prediction": self.id2label[ind_idx.item()],
                        "confidence": ind_conf.item()
                    })

            except Exception as e:
                print(f"Skipping an invalid image file. Error: {e}")
                individual_results.append({"prediction": "Error", "confidence": 0})
                continue
        
        if not all_logits:
             return {"error": "Invalid Image", "details": "All uploaded files were invalid or did not appear to be chest X-rays. Please upload a clear, frontal chest X-ray image."}

        # ... (Aggregate prediction and watermarking are the same) ...
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        probabilities = torch.nn.functional.softmax(avg_logits, dim=-1)
        confidence_score, predicted_class_idx = torch.max(probabilities, dim=-1)
        
        final_prediction = self.id2label[predicted_class_idx.item()]
        final_confidence = confidence_score.item()

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
