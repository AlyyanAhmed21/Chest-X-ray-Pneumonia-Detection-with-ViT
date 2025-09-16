# prediction.py

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import argparse
import os
from pathlib import Path

class PredictionPipeline:
    def __init__(self, model_path: str = "artifacts/model_training/model"):
        """
        Initializes the prediction pipeline by loading the trained model and processor.
        
        Args:
            model_path (str): The path to the directory containing the saved model and processor.
        """
        # Set the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the processor and model from the specified path
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path).to(self.device)
        self.model.eval() # Set the model to evaluation mode
        
        # Get the label mappings from the model's configuration
        self.id2label = self.model.config.id2label

    def predict(self, image_path: str):
        """
        Makes a prediction on a single image.
        
        Args:
            image_path (str): The file path of the image to be classified.
            
        Returns:
            dict: A dictionary containing the predicted label and its confidence score.
        """
        try:
            # Open the image using PIL (Python Imaging Library)
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return {"error": f"Image not found at path: {image_path}"}
        except Exception as e:
            return {"error": f"Failed to open image: {e}"}

        # Preprocess the image using the ViTImageProcessor
        # This handles resizing, normalization, and conversion to a tensor
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Make a prediction
        with torch.no_grad(): # Disable gradient calculation for faster inference
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get the predicted class index
        predicted_class_idx = logits.argmax(-1).item()
        
        # Get the human-readable label
        predicted_label = self.id2label[predicted_class_idx]
        
        # Calculate the confidence score using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence_score = probabilities[0][predicted_class_idx].item()
        
        result = {
            "predicted_label": predicted_label,
            "confidence_score": f"{confidence_score:.4f}"
        }
        
        return result

if __name__ == '__main__':
    # --- How to run this script from the command line ---
    # Example 1 (Pneumonia):
    # python prediction.py --image "artifacts/data_ingestion/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"
    
    # Example 2 (Normal):
    # python prediction.py --image "artifacts/data_ingestion/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

    # Set up argument parser to accept image path from the command line
    parser = argparse.ArgumentParser(description="Chest X-ray Pneumonia Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Create an instance of the pipeline
    pipeline = PredictionPipeline()
    
    # Make a prediction
    result = pipeline.predict(args.image)
    
    # Print the result
    print("\n--- Prediction Result ---")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"The model predicts this is a '{result['predicted_label']}' case.")
        print(f"Confidence: {result['confidence_score']}")
    print("-------------------------\n")