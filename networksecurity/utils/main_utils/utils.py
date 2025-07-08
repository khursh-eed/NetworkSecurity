import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import dill
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path:str)-> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys) from e

def write_yaml_file(file_path:str, content:object, replace:bool =False)->bool:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok =True)
        with open(file_path , "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_numpy_array_data(file_path:str ,array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_object(file_path:str ,obj: object):
    try:
        logging.info("Entered the save_object function of MainUtils")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file) 
        logging.info("Exited the save_object function of MainUtils")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_object(file_path:str ,)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"the file path {file_path} does not exist")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_numpy_array_data(file_path:str ) ->np.array:
    try:
       with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

def evaluate_models(X_train, X_test, Y_train, Y_test, models,param):
    try:
        report ={}
        b_model = None
        b_model_name = None
        b_score = float('-inf')

        # go thru the list of all models

        for model_name in models:
            # take all params
            model = models[model_name]
            para = param.get(model_name,{})
            

            # we pass all the params to gs
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)

            # model.set_params(**gs.best_params_)
            # model.fit(X_train, Y_train)

            # returns the best model
            best_model = gs.best_estimator_
            # we fit on it
            best_model.fit(X_train, Y_train)

            # we're only fitting the best_model, so we predict only for tht one
            y_train_pred =best_model.predict(X_train)

            y_test_pred =best_model.predict(X_test)

            train_model_score = r2_score(Y_train,y_train_pred)
            test_model_score = r2_score(Y_test,y_test_pred)

            report[model_name] = test_model_score

            # the best model is dif comapred to the ones sent, so we need to return this separetly
            # we need to store the best one
            if test_model_score > b_score:
                b_score = test_model_score
                b_model = best_model
                b_model_name = model_name

        return report,b_score, b_model,b_model_name
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

    #  for i in range(len(list(models))):
    #         # take all params
    #         model = list(models.values())[i]
    #         para = param[list(models.keys())[i]]
            

    #         # we pass all the params to gs
    #         gs = GridSearchCV(model,para,cv=3)
    #         gs.fit(X_train,Y_train)

    #         # model.set_params(**gs.best_params_)
    #         # model.fit(X_train, Y_train)

    #         # returns the best model
    #         best_model = gs.best_estimator_
    #         # we fit on it
    #         best_model.fit(X_train, Y_train)





