# src/vitClassifier/components/model_evaluation.py

import mlflow
import mlflow.pytorch
import torch
import json
from pathlib import Path
from datasets import load_from_disk
from transformers import (ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments, DefaultDataCollator)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from vitClassifier.entity.config_entity import EvaluationConfig
from vitClassifier.utils.common import read_yaml # Keep this if you need it, but it's not used here
from vitClassifier import logger

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate(self):
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the best model from the training stage and move it to the correct device
        model_path = str(self.config.path_of_model)
        model = ViTForImageClassification.from_pretrained(model_path).to(device)
        
        # Load the pre-processed test dataset
        test_data = load_from_disk(str(self.config.test_dataset_path))
        
        # We DO NOT need transforms here because the data is already processed
        # test_data.set_transform(...) # REMOVED

        # Use the default collator which handles 'pixel_values' and 'label'
        data_collator = DefaultDataCollator()

        # Dummy trainer for running predictions
        eval_args = TrainingArguments(
            output_dir="./eval_output", # Temporary directory
            per_device_eval_batch_size=self.config.batch_size,
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            args=eval_args,
            data_collator=data_collator
        )

        # --- Run Predictions ---
        logger.info("Running final evaluation on the test set...")
        outputs = trainer.predict(test_data)
        y_true = outputs.label_ids
        y_pred = outputs.predictions.argmax(1)

        # --- Calculate Metrics ---
        scores = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average='macro'),
            "precision": precision_score(y_true, y_pred, average='macro'),
            "recall": recall_score(y_true, y_pred, average='macro')
        }
        logger.info(f"Test Set Metrics: {scores}")

        # --- Save Metrics to a JSON file ---
        metrics_path = Path(self.config.metrics_file_name) 
        
        # Now create the directory
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(scores, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # --- Log to MLflow ---
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Pneumonia-ViT-Classification")

        with mlflow.start_run():
            logger.info("Logging parameters and metrics to MLflow...")
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(scores)
            
            # --- THIS IS THE FINAL FIX ---
            # Instead of logging the model object, log the directory where the
            # trained model was already saved by the Trainer.
            # `mlflow.log_artifact` is a simple upload and will not cause registry errors.
            model_dir_path = str(self.config.path_of_model)
            mlflow.log_artifact(model_dir_path, artifact_path="model")

            logger.info("Successfully logged artifacts to MLflow.")