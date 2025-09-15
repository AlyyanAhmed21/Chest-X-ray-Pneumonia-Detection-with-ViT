from vitClassifier import logger
from vitClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from vitClassifier.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from vitClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from vitClassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from dotenv import load_dotenv
load_dotenv()

def run_pipeline(stage_name, pipeline_class):
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline = pipeline_class()
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == '__main__':
    run_pipeline("Data Ingestion stage", DataIngestionTrainingPipeline)
    run_pipeline("Data Transformation stage", DataTransformationTrainingPipeline)
    run_pipeline("Model Training stage", ModelTrainingPipeline)
    run_pipeline("Model Evaluation stage", ModelEvaluationPipeline)