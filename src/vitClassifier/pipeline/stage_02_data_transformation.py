from vitClassifier.config.configuration import ConfigurationManager
from vitClassifier.components.data_transformation import DataTransformation
from vitClassifier import logger

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        params = config_manager.params
        # Get model_name from the training config section
        model_name = config_manager.config.model_training.model_name
        
        data_transformation = DataTransformation(
            config=data_transformation_config,
            random_state=params.RANDOM_STATE,
            model_name=model_name # Pass the model name
        )
        data_transformation.transform_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e