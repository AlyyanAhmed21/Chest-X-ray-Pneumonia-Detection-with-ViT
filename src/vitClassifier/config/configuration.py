# src/vitClassifier/config/configuration.py

from vitClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH # <-- THIS IMPORT IS THE FIX
from vitClassifier.utils.common import read_yaml, create_directories
from vitClassifier.entity.config_entity import (DataIngestionConfig,
                                                  DataTransformationConfig,
                                                  TrainingConfig,
                                                  EvaluationConfig)
from pathlib import Path
import os

class ConfigurationManager:
    def __init__(self, config_filepath=None, params_filepath=None):
        
        # If no path is provided when creating an instance, use the imported constants
        if config_filepath is None:
            config_filepath = CONFIG_FILE_PATH
        if params_filepath is None:
            params_filepath = PARAMS_FILE_PATH

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_kaggle_dataset_id=config.source_kaggle_dataset_id,
            unzip_dir=Path(config.unzip_dir),
            train_df_path=Path(config.train_df_path),
            test_df_path=Path(config.test_df_path),
            val_df_path=Path(config.val_df_path)
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            val_data_path=Path(config.val_data_path),
            train_dataset_path=Path(config.train_dataset_path),
            test_dataset_path=Path(config.test_dataset_path),
            val_dataset_path=Path(config.val_dataset_path)
        )
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.model_training
        params = self.params
        create_directories([Path(training.root_dir)])
        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            model_name=training.model_name,
            train_dataset_path=Path(training.train_dataset_path),
            val_dataset_path=Path(training.val_dataset_path),
            learning_rate=params.LEARNING_RATE,
            batch_size=params.BATCH_SIZE,
            epochs=params.EPOCHS,
            weight_decay=params.WEIGHT_DECAY,
            warmup_steps=params.WARMUP_STEPS,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = self.config.model_evaluation
        return EvaluationConfig(
            path_of_model=Path(eval_config.model_path),
            test_dataset_path=Path(eval_config.test_dataset_path),
            mlflow_uri=eval_config.mlflow_uri,
            all_params=self.params,
            batch_size=self.params.BATCH_SIZE,
            metrics_file_name=Path(eval_config.metrics_file_name) # <--- MAKE SURE THIS LINE EXISTS
        )