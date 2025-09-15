import os
import pandas as pd
from pathlib import Path
from vitClassifier import logger
from vitClassifier.entity.config_entity import DataIngestionConfig
import kaggle

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_dataset(self):
        try:
            # ... (download logic remains exactly the same)
            logger.info("Authenticating with Kaggle API...")
            kaggle.api.authenticate()
            logger.info("Authentication successful.")

            dataset_id = self.config.source_kaggle_dataset_id
            download_path = self.config.unzip_dir
            
            expected_data_folder = download_path / "chest_xray"
            if expected_data_folder.exists():
                logger.info(f"Dataset already exists at {expected_data_folder}. Skipping download.")
                return

            logger.info(f"Downloading dataset '{dataset_id}' to '{download_path}'...")
            kaggle.api.dataset_download_files(
                dataset=dataset_id, path=download_path, unzip=True, quiet=False
            )
            logger.info("Dataset downloaded and unzipped successfully.")

        except Exception as e:
            logger.error(f"Failed to download dataset from Kaggle. Error: {e}")
            raise e

    def create_dataframes(self):
        """
        Scans train, test, and val directories and creates separate DataFrames.
        """
        source_root = self.config.unzip_dir / "chest_xray"
        
        # Helper function to create a dataframe for a given split (train/test/val)
        def _create_df_for_split(split_name: str, save_path: Path):
            split_path = source_root / split_name
            file_names, labels = [], []
            
            # Using .glob to find all .jpeg files in NORMAL and PNEUMONIA subfolders
            for file in sorted(split_path.glob('*/*.jpeg')):
                label = file.parent.name # NORMAL or PNEUMONIA
                labels.append(label)
                file_names.append(str(file))
            
            df = pd.DataFrame({"image": file_names, "label": labels})
            df.to_csv(save_path, index=False)
            logger.info(f"Created and saved {split_name} DataFrame to {save_path}")

        # Create DataFrames for each split
        _create_df_for_split("train", self.config.train_df_path)
        _create_df_for_split("test", self.config.test_df_path)
        _create_df_for_split("val", self.config.val_df_path)

    def ingest_data(self):
        logger.info("Starting data ingestion process.")
        self.download_dataset()
        self.create_dataframes()
        logger.info("Data ingestion process completed.")