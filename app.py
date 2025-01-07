import os
import sys
import certifi
import pymongo
import pandas as pd
from dotenv import load_dotenv 

from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates

from energygeneration.logging.logger import logging
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.utils.main_utils.utils import load_object
from energygeneration.utils.main_utils.utils import add_cyclic_features,prepare_batch_input
from energygeneration.utils.ml_utils.model.batch_estimator import EnergyModel
from energygeneration.pipelines.training_pipeline import TrainingPipeline
from energygeneration.constant import SCHEMA_FILE_PATH,DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DATABASE_NAME,TIME_STEPS

 
# Load environment variables
load_dotenv()
mongo_db_url = os.getenv("PYMONGO_URI")
print(mongo_db_url)
if not mongo_db_url:
    raise ValueError("MongoDB URL is not set in environment variables.")

# MongoDB client setup
ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca) 
database=client[DATA_INGESTION_DATABASE_NAME]
collection=client[DATA_INGESTION_COLLECTION_NAME]

# FastAPI app initialization
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Template configuration
templates = Jinja2Templates(directory="./templates")

@app.get("/",tags=['authentication'])
async def index():
    """
    Redirect to the FastAPI documentation.
    """
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    """
    Triggers the training pipeline.

    Returns:
        Response: A success message if the training completes successfully.
    """
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful.")
    except Exception as e:
        raise EnergyGenerationException(e,sys)
    
@app.post("/predict") 
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    Predict outcomes using the uploaded dataset. 
    Args:
        request (Request): The incoming HTTP request.
        file (UploadFile): The uploaded CSV file containing data. 
    Returns:
        TemplateResponse: Rendered HTML table with predictions.
    """
    try:
        # Read uploaded CSV file
        df = pd.read_csv(file.file)
        print(df.head())

        # Add cyclic features and prepare batch input
        df = add_cyclic_features(data=df,schema_file=SCHEMA_FILE_PATH)
        X = prepare_batch_input(df=df,time_steps=TIME_STEPS)

        # Load scaler and model
        scaler = load_object("final_model/scaler.pkl")
        final_model = load_object("final_model/model.keras")
        energy_model = EnergyModel(model=final_model)
        print(X[:2])

        # Generate predictions 
        y_pred = energy_model.predictor(X)
        y_pred_test_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Add predictions to DataFrame
        time_steps=TIME_STEPS
        df['predicted_column'] = [None] * (time_steps - 1) + list(y_pred_test_rescaled)
        logging.info(f"Predicted values added to DataFrame:\n{df['predicted_column']}")
        print(df['predicted_column'])
        # Save predictions and render table
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)
        table_html = df.to_html(classes="table table-striped") 
        return templates.TemplateResponse("table.html",{"request":request,"table":table_html})
    except Exception as e:
        logging.error("Error during prediction.", exc_info=True)
        raise EnergyGenerationException(e,sys)
     
if __name__ == "__main__":
    
    # app_run(app,host="0.0.0.0",port=8080) # AWS EC2 deployment

    app_run(app,host="localhost",port=8000) # Local deployment