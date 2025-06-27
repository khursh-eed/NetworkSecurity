from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# dataingestion config also required to actually perform the stuff

from networksecurity.entity.config_entity import DataInjestionConfig
# artifact will be the output form here
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from  sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("mongo_db_url")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataInjestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    # function to read data form mongo db
    def export_collection_as_df(self):
        try:
            db_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            # there'll be one default column from mongodb
            if "_id" in df.columns.tolist():
                df =df.drop(columns=["_id"],axis=1)

            df.replace({"na":np.nan},inplace =True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    # function for saving the data to feature dir
    def export_data_to_featuredir(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False,header=True)
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    # fucntion to split the dataframe that we have and store it in specified folder 
    def split_train_test(self, dataframe:pd.DataFrame):
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info(f"performed train test split on the dataframe")
            logging.info("Exited split_train_test method of data ingestion class")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Exporting train and test file path")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index =False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index =False, header=True)
            logging.info(f"Exporting completed")
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_df()
            dataframe = self.export_data_to_featuredir(dataframe)
            self.split_train_test(dataframe) 

            data_ingestion_artifcat = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifcat
        except Exception as e:
            raise NetworkSecurityException(e,sys)
