from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataInjestionConfig
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == '__main__':
    try: 
        # initiating our data ingestion
        # in our dataingestion we had to provide the congif, but we cant pass it directly, we need to initialse it and everything it need to sbe inititalsied
        trainingpipelineconfig = TrainingPipelineConfig()
        # training pipeline needs datetime - takes automatically
        datainjestionconfig =DataInjestionConfig(training_pipeline_config=trainingpipelineconfig)

        dataingestion = DataIngestion(data_ingestion_config=datainjestionconfig)
        logging.info("inititating dataingestion")
        # calling the last func in dataingestion which initiates data ingestion n return dataingestion artifact
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        logging.info("data initiation completed")
        print(dataingestionartifact)

        
        data_valdiation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,data_valdiation_config)
        logging.info("initiation datavalidation")

        data_validation_artifact = data_validation.intitiate_data_validation()
        print(data_validation_artifact)

        logging.info("data validation completed")
        
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)

