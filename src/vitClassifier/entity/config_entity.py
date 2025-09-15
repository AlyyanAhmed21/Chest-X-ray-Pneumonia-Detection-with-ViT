from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_kaggle_dataset_id: str
    unzip_dir: Path
    train_df_path: Path # New
    test_df_path: Path  # New
    val_df_path: Path   # New

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path # New
    test_data_path: Path  # New
    val_data_path: Path   # New
    train_dataset_path: Path
    test_dataset_path: Path
    val_dataset_path: Path # New

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    model_name: str
    train_dataset_path: Path # New
    val_dataset_path: Path   # New
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    warmup_steps: int

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    test_dataset_path: Path
    mlflow_uri: str
    all_params: dict
    batch_size: int
    metrics_file_name: Path