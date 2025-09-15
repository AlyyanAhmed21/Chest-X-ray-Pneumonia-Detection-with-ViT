# src/vitClassifier/components/data_transformation.py

import pandas as pd
from datasets import Dataset, Image, ClassLabel
from imblearn.over_sampling import RandomOverSampler
from vitClassifier.entity.config_entity import DataTransformationConfig
from vitClassifier import logger
# --- NEW IMPORTS ---
from transformers import ViTImageProcessor
from torchvision.transforms import (Compose, Resize, ToTensor, Normalize, RandomRotation, RandomHorizontalFlip)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig, random_state: int, model_name: str):
        self.config = config
        self.random_state = random_state
        self.model_name = model_name # <-- Need model_name to load the correct processor

    def transform_data(self):
        # --- 1. Load DataFrames and apply Oversampling (same as before) ---
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)
        val_df = pd.read_csv(self.config.val_data_path)
        
        y = train_df[['label']]
        X = train_df.drop(['label'], axis=1)
        ros = RandomOverSampler(random_state=self.random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        train_df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
        
        train_dataset = Dataset.from_pandas(train_df_balanced).cast_column("image", Image())
        test_dataset = Dataset.from_pandas(test_df).cast_column("image", Image())
        val_dataset = Dataset.from_pandas(val_df).cast_column("image", Image())
        
        # --- 2. Label Encoding (same as before) ---
        labels_list = train_df_balanced['label'].unique().tolist()
        class_labels = ClassLabel(num_classes=len(labels_list), names=labels_list)

        def map_label2id(example):
            example['label'] = class_labels.str2int(example['label'])
            return example

        train_dataset = train_dataset.map(map_label2id, batched=True).cast_column('label', class_labels)
        test_dataset = test_dataset.map(map_label2id, batched=True).cast_column('label', class_labels)
        val_dataset = val_dataset.map(map_label2id, batched=True).cast_column('label', class_labels)

        # --- 3. THE NEW LOGIC: Preprocess images with .map() ---
        logger.info("Starting image preprocessing with .map(). This may take a few minutes...")
        processor = ViTImageProcessor.from_pretrained(self.model_name)
        image_mean, image_std = processor.image_mean, processor.image_std
        size = processor.size["height"]
        normalize = Normalize(mean=image_mean, std=image_std)

        # Define transforms
        _train_transforms = Compose([Resize((size, size)), RandomRotation(15), RandomHorizontalFlip(), ToTensor(), normalize])
        _val_test_transforms = Compose([Resize((size, size)), ToTensor(), normalize])
        
        def apply_train_transforms(examples):
            examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
            return examples

        def apply_val_test_transforms(examples):
            examples['pixel_values'] = [_val_test_transforms(image.convert("RGB")) for image in examples['image']]
            return examples
            
        # Use .map() to apply transforms and create 'pixel_values' column
        train_dataset = train_dataset.map(apply_train_transforms, batched=True)
        test_dataset = test_dataset.map(apply_val_test_transforms, batched=True)
        val_dataset = val_dataset.map(apply_val_test_transforms, batched=True)

        # Remove the original 'image' column to save space
        train_dataset = train_dataset.remove_columns(['image'])
        test_dataset = test_dataset.remove_columns(['image'])
        val_dataset = val_dataset.remove_columns(['image'])

        # --- 4. Save the fully processed datasets ---
        train_dataset.save_to_disk(str(self.config.train_dataset_path))
        test_dataset.save_to_disk(str(self.config.test_dataset_path))
        val_dataset.save_to_disk(str(self.config.val_dataset_path))
        
        logger.info("Data Transformation complete. Fully preprocessed datasets saved.")