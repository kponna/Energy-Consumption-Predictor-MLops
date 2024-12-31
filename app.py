import os
import sys
import certifi
ca=certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url=os.getenv("PYMONGO_URI")
print(mongo_db_url)
from energygeneration.constant import SCHEMA_FILE_PATH 
import pymongo
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from energygeneration.utils.main_utils.utils import add_cyclic_features,prepare_batch_input
from energygeneration.utils.ml_utils.model.batch_estimator import EnergyModel
from energygeneration.pipelines.training_pipeline import TrainingPipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from energygeneration.utils.main_utils.utils import load_object

client=pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)

from energygeneration.constant import DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME

database=client[DATA_INGESTION_DATABASE_NAME]
collection=client[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/",tags=['authentication'])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful.")
    except Exception as e:
        raise EnergyGenerationException(e,sys)
    
@app.post("/predict") 
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        print(df.head())
        df = add_cyclic_features(data=df,schema_file=SCHEMA_FILE_PATH)
        X = prepare_batch_input(df=df,time_steps=6)
        scaler = load_object("final_model/scaler.pkl")
        final_model = load_object("final_model/model.keras")
        energy_model = EnergyModel(model=final_model)
        print(X[:2])
        time_steps=6
        y_pred = energy_model.predictor(X)
        y_pred_test_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        df['predicted_column'] = [None] * (time_steps - 1) + list(y_pred_test_rescaled)
        print(df['predicted_column'])
        df.to_csv("prediction_output/output.csv", index=False)
        table_html = df.to_html(classes="table table-striped")
        # print(table_html)
        return templates.TemplateResponse("table.html",{"request":request,"table":table_html})
    except Exception as e:
        raise EnergyGenerationException(e,sys)
     
if __name__ == "__main__":
    
    app_run(app,host="0.0.0.0",port=8000) #while runnig aws ec2

    # app_run(app,host="localhost",port=8000) # while running fastapi