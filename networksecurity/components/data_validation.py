from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact

from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
import pandas as pd
import numpy as np
import os,sys
from scipy.stats import ks_2samp #this is for the kolmogorpv's test

from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file



# step 1:read the data from the input to this, thats the data ingestion artifact
# step 2: 

class DataValidation:
    def __init__(self,data_ingestion_artifact=DataIngestionArtifact, data_valdiation_config = DataValidationConfig):
        try:
            self.data_ingestion_artifact =data_ingestion_artifact
            self.data_validation_config = data_valdiation_config
            # we need a dif funtion to read this yaml file (created in util)
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    @staticmethod
    def read_file(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            # our entire schem is already present in self._schema_config
            number_of_columns = len(self._schema_config)
            logging.info(f"required number of Columns: {number_of_columns}")
            logging.info(f"Dataframe has {len(dataframe.columns)} columns")
            if len(dataframe.columns)==number_of_columns:
                return True
            else :return False
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def detect_data_drift(self, base_df, current_df, threshold =0.05)->bool:
        try:
            status =True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_samp =ks_2samp(d1,d2)

                if threshold <= is_samp.pvalue:
                    is_found = False
                else: 
                    is_found = True
                    status = False

                report.update({column:{
                    "p_value": float(is_samp.pvalue),
                    "drift_status": is_found
                }})

                drift_report_file_path = self.data_validation_config.drift_report_file_path

                # now we make this dir
                dir_path = os.path.dirname(drift_report_file_path)
                os.makedirs(dir_path, exist_ok=True)
                
                # we'll need to write in the file we made (use a util function)

                write_yaml_file(file_path=drift_report_file_path,content=report)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    
    def intitiate_data_validation (self)->DataValidationArtifact:
        try: 
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # read data from train and test

            train_data = DataValidation.read_file(train_file_path)
            test_data = DataValidation.read_file(test_file_path)

            # validate number of columns
            status = self.validate_number_of_columns(dataframe=train_data)
            if not status: 
                error_message =f"Train dataframe does not contain all columns. \n"

            status = self.validate_number_of_columns(dataframe=test_data)
            if not status: 
                error_message =f"Test dataframe does not contain all columns. \n"

            # check datadrift

            status = self.detect_data_drift(base_df=train_data,current_df=test_data)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file)
            os.makedirs(dir_path, exist_ok=True)

            # we save the train file to valid train file folder

            train_data.to_csv(self.data_validation_config.valid_train_file, index=False, header=True)
            test_data.to_csv(self.data_validation_config.valid_test_file, index=False, header=True)

            # we're not doing the status validation here, rather we return it in the artificat
            
            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path =None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)