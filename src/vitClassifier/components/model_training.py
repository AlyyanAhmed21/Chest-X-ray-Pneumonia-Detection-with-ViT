# src/vitClassifier/components/model_training.py

import torch
from datasets import load_from_disk
from transformers import (ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, DefaultDataCollator)
from vitClassifier.entity.config_entity import TrainingConfig
from vitClassifier import logger
import evaluate

class ModelTraining:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self):
        # --- NEW: Explicitly define the device ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # --- Load datasets (no change) ---
        train_data = load_from_disk(str(self.config.train_dataset_path))
        val_data = load_from_disk(str(self.config.val_dataset_path))
        
        id2label = {i: label for i, label in enumerate(train_data.features['label'].names)}
        label2id = {label: i for i, label in id2label.items()}
        
        model = ViTForImageClassification.from_pretrained(
            self.config.model_name, num_labels=len(id2label), id2label=id2label,
            label2id=label2id, ignore_mismatched_sizes=True
        )

        # --- NEW: Move the model to the correct device ---
        model.to(device)

        # --- TrainingArguments (no change) ---
        args = TrainingArguments(
            output_dir=str(self.config.root_dir),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            save_strategy='epoch',
            eval_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            report_to="none"
        )
        
        metric = evaluate.load("accuracy")
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            return metric.compute(predictions=predictions, references=labels)
        
        data_collator = DefaultDataCollator()
        processor = ViTImageProcessor.from_pretrained(self.config.model_name)

        trainer = Trainer(
            model, # The model is now already on the GPU
            args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor,
        )

        logger.info("Starting model fine-tuning with validation...")
        trainer.train()
        trainer.save_model(str(self.config.trained_model_path))
        logger.info("Model fine-tuning complete and best model saved.")