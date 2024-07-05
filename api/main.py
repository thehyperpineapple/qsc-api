from typing import Union
from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from model_api import model_predictions
from model_api import model_predictions, text_classification
import asyncio
from io import BytesIO
import pandas as pd
import openpyxl 
import csv 
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'], #Allow all origins
    allow_credentials=True,
    allow_methods = ['*'],
    allow_headers=['*']
)

@app.get("/")
async def read_root():
    return {"Hello":"World"}

class SinglePrediction(BaseModel):
    auth: int
    file_and_search: int 
    impersonation: int
    threats: int

@app.post("/single_response/")
async def return_single_response(item: SinglePrediction):
    print("Model API: Received payload with contents ", item)
    input_list = [item.auth, item.impersonation, item.threats, item.file_and_search]
    # int_input_list = [eval(i) for i in input_list]
    results = model_predictions.single_predictions(input_list)
    return results

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    # Debugging: Print the file name and first few bytes
    print(f"Model API: Received file {file.filename} of file size: {len(contents)} bytes")

   # Read the file into a DataFrame

    def read_csv_with_multiple_encodings(filepath):
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                return pd.read_csv(filepath, encoding=encoding)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("Unable to read the file with the given encodings")

    try:
        df = read_csv_with_multiple_encodings(BytesIO(contents))
    except Exception as e:
        return {"error": str(e)}
    print(f"Model API: Converted to Dataframe")

    print(f"Model API: Classifying Descriptions ")
    new_df = text_classification.bulk_classify_text(df)
    print("Model API: Classified Descriptions")

    print("Model API: Predicting using ML algorithms")
    new_df = model_predictions.bulk_prediction(new_df)
    print("Model API: Finished ML Predictions")

    print("Model API: Predicting using MLP regressor")
    new_df = model_predictions.bulk_deep_learning(new_df)
    print("Model API: Finished DL Predictions")

    print("Model API: Adjusting Scores")
    new_df = model_predictions.adjust_df(new_df)
    print("Model API: Returning Dataframe")


    # Save the DataFrame back to a file
    output = BytesIO()
    if file.filename.endswith('.xlsx'):
        new_df.to_excel(output, index=False)
    else:
        new_df.to_csv(output, index=False)
    
    output.seek(0)
    print("Model API: Returning File")
    return StreamingResponse(output, media_type='application/octet-stream', headers={"Content-Disposition": f"attachment; filename={file.filename}"})

# Post the file to the FastAPI server
# files = {"file": uploaded_file.getvalue()}
# response = requests.post("http://127.0.0.1:8000/uploadfile/", files=files)