import os
import json
import sys

from dotenv import load_dotenv
load_dotenv()

mongo_db_url =os.getenv("mongo_db_url")
print (mongo_db_url)

import certifi
ca= certifi.where()

import pandas as pd
import numpy as np
import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    # initialising the function
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    # now aim : read all the data from the csv and convert to json file 

    def csv_to_json_convert(self,file_path):
        # raise exception if file path not found
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_data_to_mogodb(self,records,database,collection):
        try: 
            self.database = database
            self.records =records
            self.collection =collection

            self.mongo_client = pymongo.MongoClient(mongo_db_url)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
if __name__ =='__main__':
    FILE_PATH = "Network_data/phishing_converted.csv"
    DATABASE = "Khursheed"
    Collection = "NetworkData"
    networkobj =NetworkDataExtract()
    records = networkobj.csv_to_json_convert(file_path=FILE_PATH)
    print(records)
    no_of_record = networkobj.insert_data_to_mogodb(records,DATABASE,Collection)
    print(no_of_record)
    
     
    

