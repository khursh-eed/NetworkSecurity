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
FILE_NAME: str ="NetworkData.csv"

TRAIN_FILE_NAME: str ="train.csv"
TEST_FILE_NAME: str ="test.csv"

