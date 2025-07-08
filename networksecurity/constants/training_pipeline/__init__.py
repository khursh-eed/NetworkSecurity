import os
import pandas as pd
import numpy as np
import sys

# all dataingestion related constants start w DATA_INGESTION

DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE: str ="Khursheed"
DATA_INGESTION_DIRECTORY_NAME: str ="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str ="feature_store"
DATA_INGESTION_INGESTED_DIR_NAME: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# common constants
TARGET_COLUMN: str ="Result"
PIPELINE_NAME: str= "NetworkSecurity"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str ="phishing_converted.csv"

TRAIN_FILE_NAME: str ="train.csv"
TEST_FILE_NAME: str ="test.csv"

SCHEMA_FILE_PATH: str = os.path.join("data_schema","schema.yaml")
PREPROCESSING_OBJECT_FILE_NAME: str ="preprocessing.pkl"

# data validation constants
DATA_VALIDATION_DIR:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalidated"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"

# data transformation constants
DATA_TRANSFORMATION_DIR_NAME:str ="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str ="transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str="transformed_object"
DATA_TRANFORMATION_INPUT_PARAMS: dict={
    "missing_values" : np.nan,
    "n_neighbors" : 3,
    "weights" : "uniform",
}

# model trainer constants
MODEL_TRAINER_DIR_NAME: str="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR_NAME:str ="trained_model"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_TRAINED_MODEL_EXPECTED_SCORE: float =0.6
MODEL_TRAINER_OVERFITTING_AND_UNDERFITTING_THRESHOLD: float = 0.05
SAVED_MODEL_DIR: str =os.path.join("saved_models")


