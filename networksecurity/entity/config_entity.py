from datetime import datetime
import os
from networksecurity.constants import training_pipeline

print(training_pipeline.FILE_NAME)

# all the constants are stored in training pipeline 
class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m/%d/%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        # we're making the artifact dir for each timestamp (its path only)
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)
        self.timestamp : str=timestamp


# all the paths tht'll be used in data ingestion
class DataInjestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        # ingestion directory is going to be in artifact directory and name is given from the constant
        self.data_ingestion_dir: str =os.path.join(
            training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIRECTORY_NAME
        )

        # rest of the folders and files are going to be in the data_ingestion_dir (jus made)
        self.feature_store_file_path: str =os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )
        self.training_file_path: str =os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME, training_pipeline.TRAIN_FILE_NAME
        )
        self.testing_file_path: str =os.path.join(
            self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR_NAME, training_pipeline.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE
class DataValidationConfig:
    def __init__(self,training_pipeline_config =TrainingPipelineConfig):
        # validation directory is going to be in artifact directory and name is given from the constant
        self.data_validation_dir: str= os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_VALIDATION_DIR)

        # rest of the files/folders in validation directory
        self.valid_data_dir: str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_INVALID_DIR)

        # now inside these valid and invlaid dir, we'll have train test file for both
        self.valid_test_file: str = os.path.join(self.valid_data_dir,training_pipeline.TEST_FILE_NAME)
        self.valid_train_file: str = os.path.join(self.valid_data_dir,training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file: str = os.path.join(self.invalid_data_dir,training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file: str = os.path.join(self.invalid_data_dir,training_pipeline.TRAIN_FILE_NAME)

        # report file
        self.drift_report_file_path: str =os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

class DatatransformationConfig:
    def __init__(self,training_pipeline_config =TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,training_pipeline.TRAIN_FILE_NAME.replace("csv","npy"),)
        self.transformed_test_file_path: str=os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,training_pipeline.TEST_FILE_NAME.replace("csv","npy"),)
        self.transformed_object_file_path: str=os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config = TrainingPipelineConfig):
        self.model_trainer_dir : str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.model_trainer_trained_model_file_path: str =os.path.join(self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR_NAME, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_FILE_NAME)
        self.expected_accuracy: float= training_pipeline.MODEL_TRAINER_TRAINED_MODEL_EXPECTED_SCORE
        self.overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVERFITTING_AND_UNDERFITTING_THRESHOLD
        