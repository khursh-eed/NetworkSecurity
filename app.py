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
from fastapi import Request
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_metric.model.estimator import NetworkModel

client = pymongo.MongoClient(mongodb_url)

from networksecurity.constants.training_pipeline import DATA_INGESTION_DATABASE, DATA_INGESTION_COLLECTION_NAME
database = client[DATA_INGESTION_DATABASE]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# idk why im doing this :) (read later ab fastapi)

app = FastAPI()
origins =["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins =origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers =["*"],
)
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")


# homepage

@app.get("/",tags = ["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        # pipeline runs now
        return Response("Training is successful")

    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
@app.post("/predict")
async def predict_route(request:Request,file:UploadFile=File(...) ):
    try:
        # our test data or the file we'll be uploading
        df =pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        # we'll use the network model (in estimator.py) to predict
        network_model =NetworkModel(preprocessor=preprocessor,model=model)
        y_pred = network_model.predict(df)
        print("Predictions:", y_pred)
        print("Type of predictions:", type(y_pred))
        print("Length of predictions:", len(y_pred))
        df['predicted_column'] = y_pred
        print("DataFrame shape after adding prediction:", df.shape)
        print("First few rows with prediction:\n", df.head())
        
        df.to_csv('Predicted_output/output.csv', index=False)

        
        # converting it into html
        table_html = df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request":request, "table":table_html})
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

if __name__ == "__main__":
    app_run(app, host="local host", port=8000)


