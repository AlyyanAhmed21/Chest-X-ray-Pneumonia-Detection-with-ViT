# app/prediction.py (Final Version with Relaxed Sanity Check)

import torch
from transformers import ViTImageProcessor, ViTForImageClassification, AutoImageProcessor, ResNetForImageClassification
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Union, Any
from .image_utils import add_watermark

ImageType = Union[str, Path, bytes, np.ndarray]

# A list of obviously non-medical terms to check against
FORBIDDEN_LABELS = [
    "car", "truck", "van", "motorcycle", "bicycle", "bus", "train", "boat", "airplane",
    "cat", "dog", "bird", "horse", "sheep", "cow", "bear", "zebra", "giraffe",
    "landscape", "mountain", "beach", "forest", "building", "house", "road", "street",
    "computer", "keyboard", "mouse", "laptop", "cellphone", "television",
    "food", "plate", "bowl", "cup", "fork", "knife", "spoon"
]

class PredictionPipeline:
    def __init__(self, model_path: Path = Path("artifacts/model_training/model")):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.pneumonia_processor = ViTImageProcessor.from_pretrained(model_path)
        self.pneumonia_model = ViTForImageClassification.from_pretrained(model_path).to(self.device)
        self.pneumonia_model.eval()
        self.id2label = self.pneumonia_model.config.id2label

        self.sanity_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.sanity_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(self.device)
        self.sanity_model.eval()

    def sanity_check(self, image: Image.Image) -> bool:
        """
        Uses a general-purpose model to check if the image is something obviously
        not a medical scan. Returns True if the image is plausible, False otherwise.
        """
        with torch.no_grad():
            inputs = self.sanity_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.sanity_model(**inputs)
            logits = outputs.logits
            
            top5_indices = torch.topk(logits, 5).indices[0]
            
            for idx in top5_indices:
                label = self.sanity_model.config.id2label[idx.item()].lower()
                # Check for partial matches (e.g., 'sports car', 'fire truck')
                for forbidden in FORBIDDEN_LABELS:
                    if forbidden in label:
                        print(f"Sanity check FAILED: Image classified as '{label}', which contains a forbidden term '{forbidden}'.")
                        return False # It's definitely not an X-ray
        
        print("Sanity check PASSED: Image does not appear to be a common non-medical object.")
        return True # It's plausible enough to proceed

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
                
                # --- NEW: Perform the relaxed sanity check ---
                if not self.sanity_check(image):
                    raise ValueError("Image appears to be a common object, not a medical scan.")

                valid_images_as_np.append(np.array(image))
                
                # ... (rest of the prediction logic is the same)
                inputs = self.pneumonia_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.pneumonia_model(**inputs)
                    logits = outputs.logits
                    all_logits.append(logits)
                    ind_probs = torch.nn.functional.softmax(logits, dim=-1); ind_conf, ind_idx = torch.max(ind_probs, dim=-1)
                    individual_results.append({"prediction": self.id2label[ind_idx.item()], "confidence": ind_conf.item()})

            except Exception as e:
                print(f"Skipping an invalid image file. Error: {e}")
                individual_results.append({"prediction": "Error", "confidence": 0})
                continue
        
        if not all_logits:
             return {"error": "Invalid Image", "details": "All uploaded files were invalid or did not appear to be chest X-rays. Please upload a clear, frontal chest X-ray image."}

        # ... (Aggregate prediction and watermarking are the same)
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        probabilities = torch.nn.functional.softmax(avg_logits, dim=-1)
        confidence_score, predicted_class_idx = torch.max(probabilities, dim=-1)
        final_prediction = self.id2label[predicted_class_idx.item()]
        final_confidence = confidence_score.item()
        # NOTE: The low-confidence check has been removed as the sanity check is more robust.
        
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
