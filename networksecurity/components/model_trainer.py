from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact

from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
import pandas as pd
import numpy as np
import os,sys

from networksecurity.utils.main_utils.utils import save_object,load_object , evaluate_models
from networksecurity.utils.main_utils.utils import load_numpy_array_data
from networksecurity.utils.ml_metric.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_metric.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier

import mlflow
import dagshub
dagshub.init(repo_owner='khursh-eed', repo_name='NetworkSecurity', mlflow=True)

class Trainer:
    def __init__(self,model_trainer_config = ModelTrainerConfig,data_transformation_artificat= DataTransformationArtifact):


        try:
            self.model_trainer_config = model_trainer_config
            self.data_trans_artifact =data_transformation_artificat
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
            f1_score= classification_metric.f1_score
            recall_score = classification_metric.recall_score
            precision_score = classification_metric.precision_Score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.sklearn.log_model(best_model,"model")

        
    def train_model(self,x_train, y_train, x_test, y_test):
        models = { 
            "Random Forest" : RandomForestClassifier(verbose =1),
            "Decision Tree" : DecisionTreeClassifier(),
            "Gradient Boosting" : GradientBoostingClassifier(verbose=1),
            "Logistic Regression" : LogisticRegression(verbose =1),
            "AdaBoostClassifier" : AdaBoostClassifier(),
        }
        params={
            "Decision Tree": {
                'criterion' : ['gini','entropy','log_loss'],
                # 'splitter' : ['best','random'],
                # 'max_features' : ['sqrt','log2']
            },
            "Random Forest":{
                # 'criterion' : ['gini','entropy','log_loss'],
                # 'max_features' : ['sqrt','log2'],
                'n_estimators' :[8,16,32,64,128,256]
            },
            "Gradient Boosting": {
                # 'loss' : ['log_loss','exponential'],
                'learning_rate' : [.1,.01,.05,.001],
                'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                # 'criterion' : ['gini','entropy','log_loss'],
                # 'max_features' : ['sqrt','log2'],
                'n_estimators' :[8,16,32,64,128,256],

            },
            "Logistic Regression" : {},
            "AdaBoostClassifier" : {
                'learning_rate' : [.1,.01,.05,.001],
                'n_estimators' :[8,16,32,64,128,256],

            }
        }

        # model_report:dict =evaluate_models(X_train=x_train,X_test = x_test,Y_train =y_train,Y_test=y_test,models=models, param= params)
        model_report,best_model_score,  best_model,best_model_name = evaluate_models(
                X_train=x_train,
                X_test=x_test,
                Y_train=y_train,
                Y_test=y_test,
                models=models,
                param=params
                )   
        # best_model_score =max(sorted(model_report.values()))
        print(f"Best model: {best_model_name} with score: {best_model_score}")
        # from this getting the best model score

        

        # best_model_name =list(model_report.keys())[list(model_report.values()).index(best_model_score)]

        # best_model = models[best_model_name]

        y_train_predict = best_model.predict(x_train)

        classification_train_metric = get_classification_score(y_true= y_train,y_pred = y_train_predict)
        
        # tracking experiments w  mlflow
        self.track_mlflow(best_model,classification_train_metric)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true= y_test,y_pred = y_test_pred)

        self.track_mlflow(best_model,classification_test_metric)

        # KNN imputer
        preprocessor = load_object(file_path = self.data_trans_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.model_trainer_trained_model_file_path)
        os.makedirs(model_dir_path ,exist_ok=True)

        Network_Model =NetworkModel(preprocessor=preprocessor, model=best_model)

        save_object(self.model_trainer_config.model_trainer_trained_model_file_path,obj=Network_Model)

        # Model trainer artifiact

        model_trainer_Artiefcat = ModelTrainerArtifact(model_trainer_file_path= self.model_trainer_config.model_trainer_trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric,
                             )
        
        logging.info(f"Model trainer artiefact {model_trainer_Artiefcat}")

        return model_trainer_Artiefcat


        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            # getting the path to the files
            train_file_path = self.data_trans_artifact.transformed_train_file_path
            test_file_path = self.data_trans_artifact.transformed_test_file_path

            # loading training and testign array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # creating the xtraintest, adn ytriantest
            x_train,x_test,y_train,y_test = (
                # taking all the columns except last (-1 gives the last one)
                train_arr[: ,:-1],
                test_arr[: ,:-1],
                # jus taking the last columns
                train_arr[: ,-1],
                test_arr[: ,-1],

            )

            model = self.train_model(x_train, y_train, x_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e,sys)