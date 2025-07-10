import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongodb_url = os.getenv("mongo_db_url")
print(mongodb_url)

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

# fastapi libraries

from fastapi.middleware.cors import CORSMiddleware
from fastapi import File,FastAPI,requests,UploadFile
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

client = pymongo.MongoClient(mongodb_url)

from networksecurity.constants.training_pipeline import DATA_INGESTION_DATABASE, DATA_INGESTION_COLLECTION_NAME
database = client[DATA_INGESTION_DATABASE]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins =["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins =origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers =["*"],
)


# homepage

@app.get("/",tags = ["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        # pipeline runs now()
        return Response("Training is successful")

    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

if __name__ == "__main__":
    app_run(app, host="local host", port=8000)


