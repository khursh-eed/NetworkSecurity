from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DatatransformationConfig
import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
# we need to drop the target columns so far that -> 
from networksecurity.constants.training_pipeline import TARGET_COLUMN, DATA_TRANFORMATION_INPUT_PARAMS
from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object


class DataTransformation:
    def __init__(self,data_transformation_config =DatatransformationConfig, data_validation_artifact = DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path) 
        except Exception as e:
            raise NetworkSecurityException(e,sys)   
    
    def get_data_transformer_object(self)->Pipeline:
        # KNN imputer used, along w parameters already specified from constants
        logging.info("Entered get_data_transformer_object method of DatsTrasformationn")
        try:
            # inititlaisng KNN imputer with parameter
            # double star to show key value pair
            imputer:KNNImputer= KNNImputer(**DATA_TRANFORMATION_INPUT_PARAMS)
            logging.info("initiased KNN imputer")
            # processor is of pipeline type, and we name the imputer as "imputer"
            processor:Pipeline=Pipeline([("imputer",imputer)])      

            return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    

    def initiate_data_transformation (self)->DataTransformationArtifact:
        try:
            logging.info("Entered the initiation of Data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # training dataframe
            input_feature_train_df =train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_featute_train_df = train_df[TARGET_COLUMN]
            # need to replace the -1 w 0's in the target as this is a classification problem
            target_featute_train_df =target_featute_train_df.replace(-1,0)

            # same for test
            input_feature_test_df =test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_featute_test_df = test_df[TARGET_COLUMN]
            # need to replace the -1 w 0's in the target as this is a classification problem
            target_featute_test_df =target_featute_test_df.replace(-1,0)

            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_feature_train = preprocessor_object.transform(input_feature_train_df)
            transformed_input_feature_test = preprocessor_object.transform(input_feature_test_df)

            # these are arrays so we need to save them total (
            # input features (array) + target (not array)

            # c_ (combines)
            train_arr =np.c_[transformed_input_feature_train, np.array(target_featute_train_df)]
            test_arr =np.c_[transformed_input_feature_test, np.array(target_featute_test_df)]

            # saving the numpy array

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            # creates a pickle file
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)

            # preparingn artificats
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_test_file_path= self.data_transformation_config.transformed_test_file_path,
                transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
            )

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)    
